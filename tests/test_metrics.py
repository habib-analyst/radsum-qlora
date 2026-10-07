"""Unit tests for metrics.py — pure-Python, no weights needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radsum.metrics import (
    chexbert_f1,
    finding_hallucination_rate,
    omission_rate,
    rouge_l_f1,
    to_binary,
)

# CheXbert class ids: 0 blank, 1 positive, 2 negative, 3 uncertain.
# Condition order: [..., 7 Atelectasis, 8 Pneumothorax, 9 Pleural Effusion, ...]


def _vec(**kw):
    v = [0] * 14
    for cond_idx, cls in kw.items():
        v[int(cond_idx)] = cls
    return v


def test_to_binary_positive_only():
    assert to_binary(_vec(**{"7": 1, "9": 3}))[7] == 1  # positive -> 1
    assert to_binary(_vec(**{"7": 1, "9": 3}))[9] == 0  # uncertain -> 0
    assert to_binary(_vec(**{"7": 2}))[7] == 0  # negative -> 0


def test_chexbert_f1_perfect():
    p = [_vec(**{"7": 1}), _vec(**{"9": 1})]
    r = [_vec(**{"7": 1}), _vec(**{"9": 1})]
    m = chexbert_f1(p, r)
    assert m["micro_f1"] == 1.0
    assert m["macro_f1"] == 1.0


def test_chexbert_f1_zero():
    p = [_vec(**{"7": 1})]
    r = [_vec(**{"9": 1})]
    m = chexbert_f1(p, r)
    assert m["micro_f1"] == 0.0


def test_fhr_no_hallucination():
    # generated positive also positive in source -> FHR 0
    g = [_vec(**{"7": 1})]
    s = [_vec(**{"7": 1})]
    m = finding_hallucination_rate(g, s)
    assert m["macro_fhr"] == 0.0
    assert m["micro_fhr"] == 0.0


def test_fhr_full_hallucination():
    # generated positive, source blank -> hallucinated
    g = [_vec(**{"8": 1})]
    s = [_vec(**{"8": 0})]
    m = finding_hallucination_rate(g, s)
    assert m["macro_fhr"] == 1.0
    assert m["micro_fhr"] == 1.0


def test_fhr_uncertain_source_counts_as_supported():
    g = [_vec(**{"9": 1})]
    s = [_vec(**{"9": 3})]  # uncertain -> supported, not hallucinated
    m = finding_hallucination_rate(g, s)
    assert m["macro_fhr"] == 0.0


def test_fhr_partial():
    # 2 generated findings, 1 hallucinated -> 0.5
    g = [_vec(**{"7": 1, "8": 1})]
    s = [_vec(**{"7": 1, "8": 0})]
    m = finding_hallucination_rate(g, s)
    assert m["macro_fhr"] == 0.5
    assert m["total_hallucinated"] == 1
    assert m["total_generated"] == 2


def test_fhr_skips_empty_generations():
    g = [_vec(), _vec(**{"8": 1})]
    s = [_vec(), _vec(**{"8": 0})]
    m = finding_hallucination_rate(g, s)
    assert m["n_scored_samples"] == 1
    assert m["n_samples"] == 2
    assert m["macro_fhr"] == 1.0


def test_omission_rate():
    p = [_vec(**{"7": 0})]
    r = [_vec(**{"7": 1, "9": 1})]
    m = omission_rate(p, r)
    assert m["micro_omission"] == 1.0  # missed both


def test_rouge_l_identical():
    assert rouge_l_f1("no acute findings", "no acute findings") == 1.0


def test_rouge_l_empty():
    assert rouge_l_f1("", "something") == 0.0
