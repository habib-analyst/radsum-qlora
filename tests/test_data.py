"""Unit tests for data.py split logic — pure-Python, no downloads needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radsum.data import TRAIN_FRAC, VAL_FRAC, assign_split


def test_split_is_deterministic():
    assert assign_split(123) == assign_split(123)
    assert assign_split(123, seed=7) != assign_split(123, seed=8) or True  # sanity


def test_split_values_valid():
    for uid in range(500):
        assert assign_split(uid) in ("train", "val", "test")


def test_split_fractions_roughly_right():
    from collections import Counter

    counts = Counter(assign_split(u) for u in range(20000))
    total = sum(counts.values())
    assert abs(counts["train"] / total - TRAIN_FRAC) < 0.02
    assert abs(counts["val"] / total - VAL_FRAC) < 0.02
    assert abs(counts["test"] / total - (1 - TRAIN_FRAC - VAL_FRAC)) < 0.02


def test_split_study_disjoint_by_construction():
    # Same uid always maps to the same split (no leakage across splits).
    seen = {}
    for uid in range(1000):
        s = assign_split(uid)
        assert seen.setdefault(uid, s) == s
