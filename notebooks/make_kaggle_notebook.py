"""Generate notebooks/kaggle_train.ipynb — the all-in-one Kaggle T4 notebook.

The notebook is generated from this script (single source of truth lives in
src/radsum/). Regenerate after changing the pipeline:
    python notebooks/make_kaggle_notebook.py
"""

import json
from pathlib import Path

REPO_URL = "https://github.com/habib-analyst/radsum-qlora"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md(
        "# radsum-qlora — QLoRA fine-tune of Qwen2-VL-2B for radiology summarization\n"
        "\n"
        "End-to-end: install → data (IU X-ray / OpenI) → QLoRA train (~8–12h on T4) → "
        "clinical eval (CheXbert F1 + finding-hallucination rate) → baselines.\n"
        "Budget for the whole notebook: ~14 GPU-hours (one Kaggle week)."
    ),
    code(
        "# 0. GPU check\n"
        "import torch, subprocess\n"
        "print(torch.cuda.get_device_name(0))\n"
        "print(subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader'],"
        " capture_output=True, text=True).stdout)"
    ),
    code(
        "# 1. Dependencies (run once)\n"
        "%pip install -q transformers peft bitsandbytes accelerate qwen-vl-utils datasets "
        "huggingface_hub pandas pyarrow pillow scikit-learn pyyaml tqdm"
    ),
    code(
        "# 2. Code (single source of truth = the GitHub repo)\n"
        f"!git clone {REPO_URL} 2>/dev/null || (cd radsum-qlora && git pull)\n"
        "%cd radsum-qlora\n"
        "!pip install -q -e . 2>/dev/null || pip install -q -r requirements.txt\n"
        "import sys; sys.path.insert(0, 'src')"
    ),
    code(
        "# 3. Data: IU X-ray mirror (~2.1 GB) + CheXbert weights (~1.3 GB)\n"
        "!mkdir -p data/raw weights\n"
        "!python scripts/download_openi.py --out data/raw\n"
        "!curl -sL --retry 3 -o weights/chexbert.pth "
        "https://huggingface.co/StanfordAIMI/RRG_scorers/resolve/main/chexbert.pth\n"
        "!ls -la weights/chexbert.pth"
    ),
    code(
        "# 4. Build manifest + study-disjoint splits\n"
        "!python -m radsum.data_build --raw data/raw --out data/processed\n"
        "# point configs at the Kaggle paths\n"
        "!sed -i 's|data/processed/manifest.parquet|/kaggle/working/radsum-qlora/data/processed/manifest.parquet|' "
        "configs/train.yaml configs/eval.yaml\n"
        "!sed -i 's|weights/chexbert.pth|/kaggle/working/radsum-qlora/weights/chexbert.pth|' configs/eval.yaml"
    ),
    code(
        "# 5. TRAIN — QLoRA, 3 epochs (~8-12h on T4). Logs stream to train_log.jsonl\n"
        "!python -m radsum.train --config configs/train.yaml --out_dir results/qlora_run 2>&1 | tee results/train_console.log"
    ),
    code(
        "# 6. EVAL main model: generate test impressions + CheXbert F1 + FHR\n"
        "!python -m radsum.evaluate --config configs/eval.yaml --split test "
        "--adapter results/qlora_run/adapter_final --out results/eval_qlora.json"
    ),
    code(
        "# 7. BASELINES (each ~1-3h)\n"
        "# 7a. zero-shot base 2B\n"
        "!python -m radsum.evaluate --config configs/eval.yaml --split test "
        "--out results/eval_zeroshot2b.json\n"
        "# 7b. text-only ablation: fine-tuned model, blank image at inference\n"
        "#     (proves the image matters; expect FHR to rise)\n"
        "!python -m radsum.evaluate --config configs/eval.yaml --split test "
        "--adapter results/qlora_run/adapter_final --blank_image "
        "--out results/eval_textonly.json"
    ),
    code(
        "# 7c. zero-shot 7B (bigger zero-shot reference; 4-bit fits on T4, ~15 GB download)\n"
        "#     subset of 200 to bound eval time (~2h)\n"
        "!sed 's|Qwen/Qwen2-VL-2B-Instruct|Qwen/Qwen2-VL-7B-Instruct|' configs/eval.yaml > configs/eval_7b.yaml\n"
        "!python -m radsum.evaluate --config configs/eval_7b.yaml --split test "
        "--max_samples 200 --out results/eval_zeroshot7b.json"
    ),
    code(
        "# 8. Validate the FHR metric (perturbation / negation / agreement)\n"
        "!python -m radsum.validate_metric --weights weights/chexbert.pth "
        "--manifest data/processed/manifest.parquet --n 200 --out results/metric_validation.json"
    ),
    code(
        "# 9. Summary table\n"
        "import json, glob\n"
        "for f in sorted(glob.glob('results/eval_*.json')):\n"
        "    r = json.load(open(f))\n"
        "    c = r['config']\n"
        "    print(f\"\\n== {f} ==\")\n"
        "    print('  micro F1 (14):', round(r['chexbert_f1_all14']['micro_f1'], 4))\n"
        "    print('  macro F1 (14):', round(r['chexbert_f1_all14']['macro_f1'], 4))\n"
        "    print('  micro F1 (key5):', round(r['chexbert_f1_key5']['micro_f1'], 4))\n"
        "    print('  FHR macro:', round(r['finding_hallucination_rate']['macro_fhr'], 4))\n"
        "    print('  ROUGE-L:', round(r['rouge_l_f1_mean'], 4))\n"
        "print('\\nmetric validation:')\n"
        "print(open('results/metric_validation.json').read())"
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path(__file__).parent / "kaggle_train.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
