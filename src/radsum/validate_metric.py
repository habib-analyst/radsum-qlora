"""Validation of the finding-hallucination rate (FHR) metric.

No human annotators are available in this pipeline, so FHR is validated
three ways (all automated, all honest about what they show):

1. PERTURBATION TEST: take reference impressions, inject k random
   false-positive finding sentences ("There is a pneumothorax.", ...),
   and check that FHR rises monotonically with k. A metric that does not
   respond to injected hallucinations is useless.
2. NEGATION STRESS: inject negated findings ("No pneumothorax.") and check
   FHR does NOT rise (the metric must not confuse negations for findings).
3. AGREEMENT: compare CheXbert-based FHR against an independent,
   negation-aware keyword matcher on the same samples; report Spearman
   correlation of per-sample FHR. High agreement = the metric is not an
   artifact of one labeler.

Run (needs CheXbert weights + a labeled sample of impressions):
    python -m radsum.validate_metric --weights weights/chexbert.pth \\
        --manifest data/processed/manifest.parquet --n 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from .chexbert_labeler import CheXbertLabeler
from .metrics import CHEXBERT_CONDITIONS, finding_hallucination_rate

# False-positive finding sentences, one per CheXbert condition (index-aligned
# with CHEXBERT_CONDITIONS; index 13 = No Finding, skipped).
INJECTED_FINDINGS = {
    0: "The cardiomediastinal silhouette is enlarged.",
    1: "Cardiomegaly is present.",
    2: "There is a lung opacity in the right lung.",
    3: "A lung lesion is seen.",
    4: "Pulmonary edema is present.",
    5: "There is consolidation in the left lower lobe.",
    6: "Findings are consistent with pneumonia.",
    7: "There is atelectasis at the lung base.",
    8: "A pneumothorax is present.",
    9: "There is a pleural effusion.",
    10: "Pleural thickening is noted.",
    11: "A rib fracture is seen.",
    12: "A support device is in place.",
}

NEGATED_FINDINGS = {
    8: "No pneumothorax is seen.",
    9: "There is no pleural effusion.",
    1: "The heart size is normal; no cardiomegaly.",
}

# Naive keyword sets per condition for the independent matcher.
KEYWORDS = {
    0: ["cardiomediastinal", "mediastin"],
    1: ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    2: ["opacity", "opacities"],
    3: ["lesion", "nodule", "mass"],
    4: ["edema"],
    5: ["consolidation"],
    6: ["pneumonia"],
    7: ["atelectasis"],
    8: ["pneumothorax"],
    9: ["pleural effusion", "effusion"],
    10: ["pleural thickening", "pleural"],
    11: ["fracture"],
    12: ["support device", "catheter", "line", "tube", "pacemaker"],
}

NEGATION_CUES = ["no ", "without ", "negative for ", "free of ", "absence of "]


def keyword_labels(texts):
    """Negation-aware keyword matcher -> per-text set of condition indices."""
    out = []
    for t in texts:
        low = " " + t.lower() + " "
        found = set()
        for c, kws in KEYWORDS.items():
            for kw in kws:
                pos = low.find(kw)
                if pos == -1:
                    continue
                # check a 40-char window before the keyword for negation cues
                window = low[max(0, pos - 40) : pos]
                if any(cue in window for cue in NEGATION_CUES):
                    continue
                found.add(c)
                break
        out.append(found)
    return out


def keyword_fhr(generated_texts, source_texts):
    """FHR computed with the keyword matcher instead of CheXbert."""
    gen_sets = keyword_labels(generated_texts)
    src_sets = keyword_labels(source_texts)
    per_sample, micro_h, micro_g, n = [], 0, 0, 0
    for g, s in zip(gen_sets, src_sets):
        h = len(g - s)
        if g:
            per_sample.append(h / len(g))
            n += 1
        micro_h += h
        micro_g += len(g)
    return {
        "macro_fhr": sum(per_sample) / len(per_sample) if per_sample else 0.0,
        "micro_fhr": micro_h / micro_g if micro_g else 0.0,
        "n": n,
        "per_sample": per_sample,
    }


def spearman(a, b):
    import math

    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/metric_validation.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    manifest = pd.read_parquet(args.manifest)
    sample = manifest.sample(min(args.n, len(manifest)), random_state=args.seed)
    impressions = sample["impression"].tolist()
    findings = sample["findings"].tolist()

    print("loading CheXbert ...", flush=True)
    labeler = CheXbertLabeler(args.weights)
    src_labels = labeler.label(findings)

    # --- 1. perturbation test -----------------------------------------
    pert_results = {}
    for k in [0, 1, 2, 4]:
        perturbed = []
        for imp in impressions:
            conds = rng.sample(sorted(INJECTED_FINDINGS), k) if k else []
            perturbed.append(
                imp + " " + " ".join(INJECTED_FINDINGS[c] for c in conds)
            )
        gen_labels = labeler.label(perturbed)
        fhr = finding_hallucination_rate(gen_labels, src_labels)
        pert_results[f"k={k}"] = {
            "macro_fhr": fhr["macro_fhr"],
            "micro_fhr": fhr["micro_fhr"],
        }
        print(f"  injected k={k}: macro FHR = {fhr['macro_fhr']:.3f}", flush=True)

    # --- 2. negation stress -------------------------------------------
    negated = [
        imp + " " + " ".join(NEGATED_FINDINGS[c] for c in NEGATED_FINDINGS)
        for imp in impressions
    ]
    neg_labels = labeler.label(negated)
    neg_fhr = finding_hallucination_rate(neg_labels, src_labels)
    print(f"  negated injections: macro FHR = {neg_fhr['macro_fhr']:.3f}", flush=True)

    # --- 3. agreement with keyword matcher -----------------------------
    gen_labels_plain = labeler.label(impressions)
    chex_fhr = finding_hallucination_rate(gen_labels_plain, src_labels)
    kw_fhr = keyword_fhr(impressions, findings)
    # Per-sample FHR from both methods, aligned on samples where BOTH
    # methods see at least one generated finding.
    gen_sets_kw = keyword_labels(impressions)
    src_sets_kw = keyword_labels(findings)
    per_chex, per_kw = [], []
    for g, s, gk, sk in zip(
        gen_labels_plain, src_labels, gen_sets_kw, src_sets_kw
    ):
        gen_c = {c for c in range(13) if g[c] == 1}
        src_c = {c for c in range(13) if s[c] in (1, 3)}
        if gen_c and gk:
            per_chex.append(len(gen_c - src_c) / len(gen_c))
            per_kw.append(len(gk - sk) / len(gk))
    rho = spearman(per_chex, per_kw) if len(per_chex) > 2 else 0.0
    print(f"  keyword-matcher FHR = {kw_fhr['macro_fhr']:.3f}, "
          f"spearman rho = {rho:.3f} (n={len(per_chex)})", flush=True)

    results = {
        "n_samples": len(impressions),
        "perturbation_test": pert_results,
        "negation_stress_macro_fhr": neg_fhr["macro_fhr"],
        "agreement": {
            "chexbert_macro_fhr": chex_fhr["macro_fhr"],
            "keyword_macro_fhr": kw_fhr["macro_fhr"],
            "spearman_rho": rho,
            "n_aligned_samples": len(per_chex),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
