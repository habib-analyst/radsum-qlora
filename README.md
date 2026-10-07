# radsum-qlora

**QLoRA fine-tuning of a 2B vision-language model for radiology finding
summarization — and a new metric for what BLEU/ROUGE can't see.**

Fine-tunes `Qwen/Qwen2-VL-2B-Instruct` with QLoRA (4-bit, r=64) on the IU
X-ray (OpenI) chest X-ray collection for the task *findings → impression*:
given the image plus its findings section, write a concise radiology
impression. Evaluated with CheXbert clinical-label F1 plus a proposed
**finding-hallucination rate (FHR)** — the fraction of generated positive
findings unsupported by the source findings section.

The claim: a 2B model fine-tuned on public data matches or exceeds the
clinical F1 of much larger zero-shot VLMs, while FHR exposes hallucinations
that n-gram metrics hide.

## Results (IU X-ray held-out test)

| Model | CheXbert F1 (micro, 14) | CheXbert F1 (key 5) | FHR ↓ | ROUGE-L |
|---|---|---|---|---|
| QLoRA 2B (ours) | *TBD* | *TBD* | *TBD* | *TBD* |
| Zero-shot 2B base | *TBD* | *TBD* | *TBD* | *TBD* |
| Zero-shot 7B | *TBD* | *TBD* | *TBD* | *TBD* |
| QLoRA 2B, blank image (text-only ablation) | *TBD* | *TBD* | *TBD* | *TBD* |

Full numbers, configs, and the writeup: [`PAPER.md`](PAPER.md).

## The finding-hallucination rate

For each generated impression, CheXbert extracts positive findings; a finding
is *hallucinated* if the source findings section marks it negative/blank
(uncertain counts as supported — conservative by design). FHR is the
macro/micro mean over samples. Validated by (1) a perturbation test —
injecting false findings raises FHR monotonically, (2) a negation stress
test — negated findings do *not* raise it, (3) agreement with an
independent negation-aware keyword matcher.

```
python -m radsum.validate_metric --weights weights/chexbert.pth \
    --manifest data/processed/manifest.parquet
```

## Quickstart (Kaggle T4, ~14 GPU-h total)

```bash
git clone https://github.com/habib-analyst/radsum-qlora && cd radsum-qlora
pip install -r requirements.txt

# 1. data: download IU X-ray mirror + build manifest + splits (~2.1 GB)
python scripts/download_openi.py --out data/raw
python -m radsum.data_build --raw data/raw --out data/processed

# 2. train QLoRA (~8-12h on T4)
python -m radsum.train --config configs/train.yaml --out_dir results/qlora_run

# 3. evaluate (CheXbert F1 + FHR + ablations, ~2-4h)
python -m radsum.evaluate --config configs/eval.yaml --split test \
    --adapter results/qlora_run/adapter_final --out results/eval_qlora.json
```

Or run the all-in-one Kaggle notebook: [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb).

## Layout

```
src/radsum/
  data.py              IU X-ray manifest + study-disjoint splits
  train.py             QLoRA training loop (loss on response tokens only)
  evaluate.py          generation + clinical eval harness
  chexbert_labeler.py  CheXbert inference from local weights
  metrics.py           CheXbert F1, finding-hallucination rate, omission, ROUGE-L
  validate_metric.py   perturbation / negation / agreement validation of FHR
configs/               train.yaml, eval.yaml
```

## Reproducibility

- Deterministic study-disjoint 70/10/20 splits (hash of report uid, seed 42).
- All configs, seeds, and the training log committed under `results/`.
- CheXbert weights: `StanfordAIMI/RRG_scorers` (`chexbert.pth`), loaded locally.

## License

MIT. Data: IU X-ray via the `ykumards/open-i` mirror (CC-BY-ND-4.0).
