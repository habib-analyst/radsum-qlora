"""Download the IU X-ray (OpenI) parquet shards from the Hugging Face mirror.

Source: ykumards/open-i (CC-BY-ND-4.0), 5 parquet shards, ~2.1 GB total.

Run:
    python scripts/download_openi.py --out data/raw
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DATASET = "ykumards/open-i"
API = f"https://huggingface.co/api/datasets/{DATASET}"


def shard_files() -> list[str]:
    import json
    import urllib.request

    with urllib.request.urlopen(API, timeout=60) as r:
        meta = json.load(r)
    return sorted(
        s["rfilename"]
        for s in meta["siblings"]
        if s["rfilename"].endswith(".parquet")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in shard_files():
        dest = out / Path(f).name
        if dest.exists():
            print(f"exists, skipping: {dest.name}")
            continue
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{f}"
        print(f"downloading {Path(f).name} ...", flush=True)
        rc = subprocess.run(
            ["curl", "-sL", "--retry", "3", "-o", str(dest), url]
        ).returncode
        if rc != 0:
            print(f"FAILED: {f}", file=sys.stderr)
            sys.exit(1)
        print(f"  ok ({dest.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
