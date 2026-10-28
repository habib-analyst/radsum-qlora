"""CPU-only evaluation that needs neither a GPU nor CheXbert weights.

What this measures (and what it does not):

* It validates the *metric code* of this project on real IU X-ray report
  text, using the independent negation-aware keyword matcher from
  `radsum.validate_metric` as the labeler. The CheXbert-based validation
  in `validate_metric.py` additionally needs `weights/chexbert.pth`;
  the QLoRA model's own metrics need the GPU training run. Neither is
  replaced by this script -- it is the honest CPU-runnable subset.

Measurements on the held-out test split (repo's uid-hash split):

1. GOLD-IMPRESSION FHR: keyword-matcher FHR of the *reference* impression
   against its own findings section. Real radiologists sometimes state
   conclusions not literally present in the findings text, so this is a
   descriptive human-text baseline, not zero by construction.
2. PERTURBATION TEST: inject k in {0,1,2,4} false-positive finding
   sentences (the repo's INJECTED_FINDINGS bank) into the reference
   impressions; keyword FHR must rise monotonically with k.
3. NEGATION STRESS: append the repo's NEGATED_FINDINGS sentences; FHR
   must not materially rise (negations are not findings).
4. ROUGE-L of reference impression vs findings (lexical overlap stat).

Data: one parquet shard of the ykumards/open-i HF mirror (CC-BY-ND-4.0),
text columns only; rows filtered to non-empty findings AND impression
(the same text filter as `radsum.data.build_manifest`, minus its image
step). Deterministic: seed 0 everywhere; if the test split exceeds --n
rows, a seeded sample is taken.

Run:
    python scripts/download_openi.py --out data/raw   # or fetch one shard
    python scripts/run_cpu_eval.py --raw data/raw --n 300 \
        --out results/cpu_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from radsum.data import _clean, assign_split
from radsum.metrics import rouge_l_f1
from radsum.validate_metric import (
    INJECTED_FINDINGS,
    NEGATED_FINDINGS,
    keyword_fhr,
)


def load_test_pairs(raw_dir: str, n: int, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    shards = sorted(Path(raw_dir).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {raw_dir}")
    frames = [
        pd.read_parquet(s, columns=["uid", "findings", "impression"])
        for s in shards
    ]
    df = pd.concat(frames, ignore_index=True)
    n_shard_rows = len(df)
    df["findings"] = df["findings"].map(_clean)
    df["impression"] = df["impression"].map(_clean)
    keep = (df["findings"].str.len() > 0) & (df["impression"].str.len() > 0)
    df = df[keep].reset_index(drop=True)
    df["split"] = df["uid"].map(lambda u: assign_split(int(u)))
    test = df[df["split"] == "test"].reset_index(drop=True)
    n_test = len(test)
    if n_test > n:
        test = test.sample(n, random_state=seed).reset_index(drop=True)
    provenance = {
        "shards": [s.name for s in shards],
        "rows_in_shards": int(n_shard_rows),
        "rows_after_text_filter": int(len(df)),
        "test_split_rows": int(n_test),
        "n_evaluated": int(len(test)),
    }
    return test, provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cpu_eval.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    test, provenance = load_test_pairs(args.raw, args.n, args.seed)
    findings = test["findings"].tolist()
    impressions = test["impression"].tolist()
    print(f"evaluating {len(test)} test-split pairs", flush=True)

    # 1. gold-impression FHR (keyword matcher)
    gold = keyword_fhr(impressions, findings)

    # 2. perturbation test (mirrors validate_metric.py, keyword labeler)
    perturbation = {}
    for k in [0, 1, 2, 4]:
        perturbed = []
        for imp in impressions:
            conds = rng.sample(sorted(INJECTED_FINDINGS), k) if k else []
            perturbed.append(imp + " " + " ".join(INJECTED_FINDINGS[c] for c in conds))
        r = keyword_fhr(perturbed, findings)
        perturbation[f"k={k}"] = {
            "macro_fhr": r["macro_fhr"],
            "micro_fhr": r["micro_fhr"],
        }
        print(f"  injected k={k}: macro FHR = {r['macro_fhr']:.3f}", flush=True)

    # 3. negation stress
    negated = [
        imp + " " + " ".join(NEGATED_FINDINGS[c] for c in NEGATED_FINDINGS)
        for imp in impressions
    ]
    neg = keyword_fhr(negated, findings)
    print(f"  negated injections: macro FHR = {neg['macro_fhr']:.3f}", flush=True)

    # 4. ROUGE-L impression vs findings
    rouge = [rouge_l_f1(imp, find) for imp, find in zip(impressions, findings)]

    ks = [perturbation[f"k={k}"]["macro_fhr"] for k in [0, 1, 2, 4]]
    results = {
        "labeler": "negation-aware keyword matcher (radsum.validate_metric.keyword_fhr)",
        "provenance": provenance,
        "gold_impression_fhr": {
            "macro_fhr": gold["macro_fhr"],
            "micro_fhr": gold["micro_fhr"],
            "n_scored": gold["n"],
        },
        "perturbation_test": perturbation,
        "perturbation_monotonic": all(b >= a for a, b in zip(ks, ks[1:])),
        "negation_stress_macro_fhr": neg["macro_fhr"],
        "rouge_l_impression_vs_findings_mean": sum(rouge) / len(rouge) if rouge else 0.0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
