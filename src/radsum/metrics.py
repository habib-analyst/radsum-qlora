"""Clinical metrics for radiology impression generation.

Two families:
1. CheXbert clinical-label F1: disease labels extracted (by the CheXbert
   labeler) from generated vs reference impressions, compared as multilabel
   classification. Reported as micro-F1 and macro-F1 over the 14 conditions,
   plus F1 over the 5 key conditions.
2. Finding-hallucination rate (FHR) -- the novel metric of this project:
   the fraction of *generated positive findings* that are NOT supported by
   the source findings section. BLEU/ROUGE reward n-gram overlap and can stay
   high while a model invents a pneumothorax; FHR measures exactly that
   failure mode.

Label convention (CheXbert): per condition, class in
{0: blank, 1: positive, 2: negative, 3: uncertain}; the 14th condition
("No Finding") uses {0: no, 1: yes}.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

CHEXBERT_CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

# The 5 conditions most commonly reported in RRG literature.
KEY_CONDITIONS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

N_FINDING_CONDITIONS = 13  # all except "No Finding"

# CheXbert class ids
BLANK, POSITIVE, NEGATIVE, UNCERTAIN = 0, 1, 2, 3


def to_binary(labels: Sequence[int]) -> List[int]:
    """Map CheXbert 4-class labels to binary presence (positive -> 1).

    Uncertain is mapped to 0 (conservative: only asserted findings count).
    "No Finding" (index 13) is passed through as-is (already 0/1).
    """
    out = []
    for i, lab in enumerate(labels):
        if i == 13:
            out.append(1 if lab == 1 else 0)
        else:
            out.append(1 if lab == POSITIVE else 0)
    return out


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def chexbert_f1(
    pred_labels: Sequence[Sequence[int]],
    ref_labels: Sequence[Sequence[int]],
    conditions: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Multilabel F1 of predicted vs reference CheXbert condition labels.

    Args:
        pred_labels: list of 14-class-id vectors (one per generated impression).
        ref_labels: list of 14-class-id vectors (one per reference impression).
        conditions: subset of condition names to score; default = all 14.

    Returns dict with micro/macro precision, recall, F1 over the positive class.
    """
    idx = (
        list(range(14))
        if conditions is None
        else [CHEXBERT_CONDITIONS.index(c) for c in conditions]
    )
    f1s, precisions, recalls = [], [], []
    micro_tp = micro_fp = micro_fn = 0
    for j in idx:
        tp = fp = fn = 0
        for p, r in zip(pred_labels, ref_labels):
            pb = to_binary(p)[j]
            rb = to_binary(r)[j]
            if pb == 1 and rb == 1:
                tp += 1
            elif pb == 1 and rb == 0:
                fp += 1
            elif pb == 0 and rb == 1:
                fn += 1
        p_, r_, f_ = _prf(tp, fp, fn)
        precisions.append(p_)
        recalls.append(r_)
        f1s.append(f_)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    mp, mr, mf = _prf(micro_tp, micro_fp, micro_fn)
    n = len(idx)
    # Macro average is taken over conditions with support (tp+fp+fn > 0);
    # conditions absent from both predictions and references carry no signal
    # and would otherwise drag the mean toward zero on small samples.
    scored = [
        (p, r, f)
        for (p, r, f, j) in zip(
            precisions, recalls, f1s, idx
        )
        if _has_support(pred_labels, ref_labels, j)
    ]
    if scored:
        macro_p = sum(p for p, _, _ in scored) / len(scored)
        macro_r = sum(r for _, r, _ in scored) / len(scored)
        macro_f = sum(f for _, _, f in scored) / len(scored)
    else:
        macro_p = macro_r = macro_f = 0.0
    return {
        "micro_precision": mp,
        "micro_recall": mr,
        "micro_f1": mf,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "n_conditions": n,
        "n_scored_conditions": len(scored),
    }


def _has_support(
    pred_labels: Sequence[Sequence[int]],
    ref_labels: Sequence[Sequence[int]],
    j: int,
) -> bool:
    for p, r in zip(pred_labels, ref_labels):
        if to_binary(p)[j] == 1 or to_binary(r)[j] == 1:
            return True
    return False


def finding_hallucination_rate(
    gen_labels: Sequence[Sequence[int]],
    src_labels: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """Finding-hallucination rate (FHR): fraction of generated positive findings
    unsupported by the source findings section.

    A generated finding for condition c is *hallucinated* when CheXbert marks
    it positive in the generated impression while the source findings section
    marks it anything other than positive or uncertain (i.e. negative/blank).
    Uncertain in the source counts as supported (conservative against the
    metric -- we only flag clear inventions).

    "No Finding" is excluded: it is not a finding.

    Returns macro mean (per-sample, skipping samples with no generated
    findings), micro rate (global counts), and supporting counts.
    """
    per_sample = []
    micro_h, micro_g = 0, 0
    n_scored = 0
    for g, s in zip(gen_labels, src_labels):
        hallucinated = 0
        generated = 0
        for c in range(N_FINDING_CONDITIONS):
            if g[c] == POSITIVE:
                generated += 1
                if s[c] not in (POSITIVE, UNCERTAIN):
                    hallucinated += 1
        if generated > 0:
            per_sample.append(hallucinated / generated)
            n_scored += 1
        micro_h += hallucinated
        micro_g += generated
    return {
        "macro_fhr": sum(per_sample) / len(per_sample) if per_sample else 0.0,
        "micro_fhr": micro_h / micro_g if micro_g > 0 else 0.0,
        "n_scored_samples": n_scored,
        "n_samples": len(gen_labels),
        "total_hallucinated": micro_h,
        "total_generated": micro_g,
    }


def omission_rate(
    pred_labels: Sequence[Sequence[int]],
    ref_labels: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """Fraction of reference positive findings missing from the prediction
    (the complementary failure mode to hallucination)."""
    per_sample = []
    micro_m, micro_r = 0, 0
    for p, r in zip(pred_labels, ref_labels):
        pb, rb = to_binary(p), to_binary(r)
        missed = sum(1 for c in range(N_FINDING_CONDITIONS) if rb[c] == 1 and pb[c] == 0)
        total = sum(1 for c in range(N_FINDING_CONDITIONS) if rb[c] == 1)
        if total > 0:
            per_sample.append(missed / total)
        micro_m += missed
        micro_r += total
    return {
        "macro_omission": sum(per_sample) / len(per_sample) if per_sample else 0.0,
        "micro_omission": micro_m / micro_r if micro_r > 0 else 0.0,
    }


def _lcs_len(a: List[str], b: List[str]) -> int:
    # Memory-efficient LCS on token lists.
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l_f1(pred: str, ref: str) -> float:
    """ROUGE-L F1 (pure-Python, no extra deps) as a lexical sanity metric."""
    pt, rt = pred.lower().split(), ref.lower().split()
    if not pt or not rt:
        return 0.0
    lcs = _lcs_len(pt, rt)
    p = lcs / len(pt)
    r = lcs / len(rt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
