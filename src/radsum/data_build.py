"""CLI: build the IU X-ray manifest + study-disjoint splits.

Run:
    python -m radsum.data_build --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse

from .data import build_manifest, make_splits, split_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir with OpenI parquet shards")
    ap.add_argument("--out", required=True, help="output dir for manifest + images")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest_path = build_manifest(args.raw, args.out)
    manifest = make_splits(manifest_path, seed=args.seed)
    counts = split_counts(manifest)
    print(f"manifest: {manifest_path}")
    print(f"samples: {len(manifest)}, splits: {counts}")


if __name__ == "__main__":
    main()
