# radsum-qlora

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/radsum-qlora.svg)](https://github.com/habib-analyst/radsum-qlora)

**QLoRA fine-tuning of a 2B vision-language model for radiology finding summarization — and a new metric for what BLEU/ROUGE can't see.** Given a chest X-ray plus its findings section, write a concise radiology impression. The proposed **finding-hallucination rate (FHR)** measures the fraction of generated positive findings unsupported by the source findings — exposing hallucinations that n-gram metrics hide.

## Status: pipeline + metric validated on CPU; GPU training queued

The data pipeline is complete: the IU X-ray (OpenI) collection is downloaded, cleaned, and split into **3,689 samples**. The FHR metric and its CheXbert labeler are implemented, tested (17/17 tests pass), and **validated on real report text** — see [Results](#results). The QLoRA training run (~16–24 GPU-hours on a T4×2) is deferred to the weekly Kaggle quota reset, so **no model results exist yet**: there is no trained adapter and no model ROUGE/F1 to report. The only numbers in this README are measurements of the dataset, the metric, and the test suite.

## Results

### Dataset (verified from the built manifest)

Source: `ykumards/open-i` (IU X-ray / OpenI, CC-BY-ND-4.0), 3,851 report rows. After cleaning and image validation, the manifest holds **3,689 samples**, split study-disjointly by report-uid hash (seed 42):

| Split | Samples |
|---|---|
| Train | 2,564 |
| Val | 402 |
| Test | 723 |

### FHR metric validation — CheXbert labeler (n = 200 reports, seed 0)

FHR is validated three automated ways (`python -m radsum.validate_metric`; full output in [`results/metric_validation.json`](results/metric_validation.json)). An earlier n = 100 run gave the same picture and is kept as [`results/metric_validation_n100_prior.json`](results/metric_validation_n100_prior.json).

| Test | Result |
|---|---|
| Perturbation: inject k false findings into reference impressions | macro FHR **0.317 → 0.884 → 0.916 → 0.929** for k = 0, 1, 2, 4 (strictly monotonic) |
| Negation stress: inject negated findings ("No pneumothorax is seen.") | macro FHR 0.317 → **0.337** (Δ +0.020; the metric largely ignores negations, inheriting CheXbert's residual negation errors) |
| Agreement with an independent negation-aware keyword matcher | keyword macro FHR 0.557 vs CheXbert 0.317; Spearman ρ = **0.556** (n = 37 aligned samples) |

Read the k = 0 baseline honestly: reference impressions score FHR 0.317, not 0 — radiologists state conclusions not literally present in the findings section, and CheXbert adds label noise. FHR is a *comparative* metric: the delta between systems is the signal, not the absolute level.

### CPU evaluation — keyword matcher (test split, 760 report pairs)

`scripts/run_cpu_eval.py` repeats the validation with the keyword matcher alone (no CheXbert weights needed) on the held-out test split ([`results/cpu_eval.json`](results/cpu_eval.json)):

| Measurement | Value |
|---|---|
| Gold-impression keyword FHR (macro / micro) | 0.666 / 0.697 (204 of 760 impressions had ≥1 keyword-detectable finding) |
| Perturbation, macro FHR for k = 0, 1, 2, 4 | 0.666 → 0.834 → 0.879 → 0.890 (monotonic) |
| Negation stress, macro FHR | 0.666 (Δ 0.000) |
| ROUGE-L, impression vs findings (mean) | 0.096 |

The keyword matcher is the weaker instrument: it detects findings in only ~27% of impressions and its absolute FHR (0.666) runs far above CheXbert's (0.317) on the same kind of text. It is a cross-check on the metric's *behaviour*, not a substitute labeler.

### Model results

None yet. The comparison table this project is built around — QLoRA 2B vs zero-shot 2B vs zero-shot 7B vs blank-image ablation, scored with CheXbert F1 (micro-14, key-5), FHR, omission rate, and ROUGE-L on the 723-sample test split — will be filled from the Kaggle run (`notebooks/kaggle_train.ipynb`) when quota allows.

## Quickstart (CPU, no GPU needed)

```bash
git clone https://github.com/habib-analyst/radsum-qlora.git
cd radsum-qlora
pip install -r requirements.txt
python scripts/download_openi.py --out data/raw
python -m radsum.data_build --raw data/raw --out data/processed   # → 3,689 samples
python scripts/run_cpu_eval.py --raw data/raw --out results/cpu_eval.json
```

With CheXbert weights supplied, the full metric validation is:

```bash
python -m radsum.validate_metric --weights weights/chexbert.pth \
  --manifest data/processed/manifest.parquet --n 200
```

## Usage example

Score your own generated impressions against their source findings (FHR + clinical F1), once you have CheXbert weights:

```python
from radsum.chexbert_labeler import CheXbertLabeler
from radsum.metrics import chexbert_f1, finding_hallucination_rate

labeler = CheXbertLabeler("weights/chexbert.pth")
gen_labels = labeler.label(generated_impressions)   # your model's output
ref_labels = labeler.label(reference_impressions)   # gold impressions
src_labels = labeler.label(findings_sections)       # source findings

print(chexbert_f1(gen_labels, ref_labels))                  # clinical F1
print(finding_hallucination_rate(gen_labels, src_labels))   # FHR
```

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
- **Training** (`src/radsum/train.py`) — QLoRA (4-bit, r=64) on `Qwen/Qwen2-VL-2B-Instruct`, loss on response tokens only.
- **Evaluation** (`src/radsum/evaluate.py`) — CheXbert F1 (micro-14 and key-5), FHR, omission rate, ROUGE-L on the held-out test split.

## Limitations

- No trained model yet — any FHR/F1 claim about the *model* is a hypothesis until the queued run completes. Everything measured so far concerns the data and the metric.
- IU X-ray is small (3,689 samples) and single-institution; impressions are short and formulaic, so results may not transfer to other institutions or report styles.
- CheXbert is an imperfect labeler; its errors (notably on negation) propagate into both F1 and FHR. FHR validation is automated — no human-annotator agreement study has been run.
- The metric's CheXbert dependency requires `weights/chexbert.pth`, which you must supply (not in the repo; from `StanfordAIMI/RRG_scorers`).

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
