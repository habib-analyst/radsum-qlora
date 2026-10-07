"""IU X-ray (OpenI) data pipeline: findings -> impression summarization.

Source: the `ykumards/open-i` Hugging Face mirror of the Indiana University
chest X-ray collection (CC-BY-ND-4.0). Each row carries report text
(findings / impression / indication / comparison) plus frontal and lateral
PNG bytes.

Pipeline:
1. `build_manifest(raw_dir, out_dir)` -- read the parquet shards, keep rows
   with non-empty findings AND impression, write one PNG per frontal image
   to disk, and emit a manifest parquet with one row per study-image.
2. `make_splits(manifest, seed)` -- deterministic study-disjoint 70/10/20
   train/val/test split by hashing the report uid.

Task framing: input = frontal chest X-ray + findings text (+ optional
indication); target = impression text.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

TRAIN_FRAC, VAL_FRAC = 0.70, 0.10
SPLIT_SEED = 42


def _clean(text) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def load_raw_shards(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    shards = sorted(raw_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {raw_dir}")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def build_manifest(raw_dir: str | Path, out_dir: str | Path) -> Path:
    """Build the image+text manifest. Returns path to manifest parquet."""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_shards(raw_dir)
    df["findings"] = df["findings"].map(_clean)
    df["impression"] = df["impression"].map(_clean)
    df["indication"] = df["indication"].map(_clean)
    df["comparison"] = df["comparison"].map(_clean)

    keep = (df["findings"].str.len() > 0) & (df["impression"].str.len() > 0)
    df = df[keep].reset_index(drop=True)

    # The stored bytes are already encoded images (JPEG in this mirror):
    # write them raw (no PIL re-encode). Validity is checked by magic
    # bytes; a random sample is verified with PIL in verify_images().
    rows = []
    for i, row in df.iterrows():
        img_bytes = row.get("img_frontal")
        if img_bytes is None:
            continue
        raw = bytes(img_bytes)
        if raw[:2] == b"\xff\xd8":
            ext = "jpg"
        elif raw[:8] == b"\x89PNG\r\n\x1a\n":
            ext = "png"
        else:
            continue
        name = f"iu_{int(row['uid']):06d}_{i}.{ext}"
        (img_dir / name).write_bytes(raw)
        rows.append(
            {
                "sample_id": f"iu_{int(row['uid']):06d}_{i}",
                "uid": int(row["uid"]),
                "image_path": str(img_dir / name),
                "findings": row["findings"],
                "impression": row["impression"],
                "indication": row["indication"],
                "comparison": row["comparison"],
            }
        )
    manifest = pd.DataFrame(rows)
    out_path = out_dir / "manifest.parquet"
    manifest.to_parquet(out_path, index=False)
    return out_path


def verify_images(manifest: pd.DataFrame, n: int = 100, seed: int = 0) -> dict:
    """Open a random sample of manifest images with PIL to catch corruption."""
    from PIL import Image

    sample = manifest.sample(min(n, len(manifest)), random_state=seed)
    ok, bad = 0, []
    for _, row in sample.iterrows():
        try:
            with Image.open(row["image_path"]) as im:
                im.verify()
            ok += 1
        except Exception:
            bad.append(row["sample_id"])
    return {"checked": len(sample), "ok": ok, "bad": bad}


def assign_split(uid: int, seed: int = SPLIT_SEED) -> str:
    """Deterministic study-disjoint split from the report uid."""
    h = int(hashlib.md5(f"{seed}:{uid}".encode()).hexdigest(), 16) % 1000
    if h < int(TRAIN_FRAC * 1000):
        return "train"
    if h < int((TRAIN_FRAC + VAL_FRAC) * 1000):
        return "val"
    return "test"


def make_splits(manifest_path: str | Path, seed: int = SPLIT_SEED) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    manifest["split"] = manifest["uid"].map(lambda u: assign_split(int(u), seed))
    manifest.to_parquet(manifest_path, index=False)
    return manifest


def split_counts(manifest: pd.DataFrame) -> dict:
    return manifest["split"].value_counts().to_dict()
