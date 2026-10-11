# radsum-qlora

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/radsum-qlora.svg)](https://github.com/habib-analyst/radsum-qlora)

**QLoRA fine-tuning of a 2B vision-language model for radiology finding summarization — and a new metric for what BLEU/ROUGE can't see.** Given a chest X-ray plus its findings section, write a concise radiology impression. The proposed **finding-hallucination rate (FHR)** measures the fraction of generated positive findings unsupported by the source findings — exposing hallucinations that n-gram metrics hide.

## Status: CPU prep done, GPU training queued

The data pipeline is complete and validated on CPU: IU X-ray (OpenI) collection downloaded, cleaned, and split into **3,689 samples**. The CheXbert clinical labeler and the FHR metric are implemented and tested. The QLoRA training run (~16–24 GPU-hours on a T4×2) is deferred to the weekly Kaggle quota reset — there are **no training results yet**, so this README has no results table. Numbers will be posted here when the run finishes.

## Quickstart (CPU, no GPU needed)

```bash
git clone https://github.com/habib-analyst/radsum-qlora.git
cd radsum-qlora
pip install -r requirements.txt
python scripts/download_openi.py --out data/raw
python -m radsum.data_build --raw data/raw --out data/processed   # → 3,689 samples
python -m radsum.validate_metric --weights weights/chexbert.pth \
  --pairs data/processed/val.jsonl
```

`validate_metric` scores sample findings→impression pairs with CheXbert F1 and FHR — this is the CPU-runnable part you can try today.

## Training (GPU, when quota allows)

```bash
python -m radsum.train --config configs/train.yaml --out_dir results/qlora_run      # Qwen/Qwen2-VL-2B-Instruct, QLoRA 4-bit, r=64
python -m radsum.evaluate --config configs/eval.yaml --split test \
  --out_dir results/eval                                                            # CheXbert F1 + FHR + ROUGE-L
```

A ready-to-run Kaggle notebook is in `notebooks/kaggle_train.ipynb` (regenerable via `notebooks/make_kaggle_notebook.py`).

## How it works

The claim being tested: a 2B model fine-tuned on public data matches or exceeds the clinical F1 of much larger zero-shot VLMs, while FHR exposes hallucinations that n-gram metrics hide.

- **Data** (`src/radsum/data.py`, `data_build.py`) — IU X-ray findings→impression pairs, deduped and split; 3,689 samples.
- **Metric** (`src/radsum/metrics.py`, `chexbert_labeler.py`) — CheXbert 14-label clinical F1 plus FHR: generated positive findings are checked against the source findings section; unsupported ones count as hallucinations.
- **Training** (`src/radsum/train.py`) — QLoRA (4-bit, r=64) on `Qwen/Qwen2-VL-2B-Instruct`.
- **Evaluation** (`src/radsum/evaluate.py`) — CheXbert F1 (micro-14 and key-5), FHR, ROUGE-L on the held-out test split.

## Limitations

- No training results exist yet — any FHR/F1 claim is a hypothesis until the queued run completes.
- The metric's CheXbert dependency requires `weights/chexbert.pth`, which you must supply (not in the repo).

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
