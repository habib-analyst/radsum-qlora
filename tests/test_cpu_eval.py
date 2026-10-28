"""Tests for scripts/run_cpu_eval.py helpers (synthetic parquet, no download)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_cpu_eval import load_test_pairs  # noqa: E402


def _write_shard(tmp_path: Path) -> None:
    rows = [
        {
            "uid": uid,
            "findings": "The heart is enlarged. No pneumothorax.",
            "impression": "Cardiomegaly. No pneumothorax.",
        }
        for uid in range(1, 41)
    ]
    rows.append({"uid": 999, "findings": "", "impression": "Empty findings."})
    pd.DataFrame(rows).to_parquet(tmp_path / "train-00000-of-00001.parquet")


def test_load_test_pairs_filters_and_splits(tmp_path):
    _write_shard(tmp_path)
    test, prov = load_test_pairs(str(tmp_path), n=300)
    assert prov["rows_in_shards"] == 41
    assert prov["rows_after_text_filter"] == 40  # empty-findings row dropped
    assert prov["n_evaluated"] == len(test) == prov["test_split_rows"]
    assert (test["findings"].str.len() > 0).all()
    assert (test["impression"].str.len() > 0).all()


def test_load_test_pairs_caps_n(tmp_path):
    _write_shard(tmp_path)
    test, prov = load_test_pairs(str(tmp_path), n=2)
    assert len(test) == min(2, prov["test_split_rows"])
