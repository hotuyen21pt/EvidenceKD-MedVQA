"""Build ``data/processed`` from the sibling VinDR-CXR-VQA repository.

Runs ``vindr_cxr_vqa.build_dataset`` with ``--disjoint-test`` (design §5.2:
train identical to the paper reproduction, test carved out of val so that
thresholds can be chosen on val) and records the split manifest hash.

    python scripts/build_data.py [--vindr ../VinDR-CXR-VQA] [--out data/processed]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vindr", default=os.path.join(REPO, "..", "VinDR-CXR-VQA"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "processed"))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)

    vindr = os.path.abspath(args.vindr)
    if not os.path.isdir(os.path.join(vindr, "vindr_cxr_vqa")):
        print(f"vindr_cxr_vqa package not found under {vindr}", file=sys.stderr)
        return 2
    cmd = [args.python, "-m", "vindr_cxr_vqa.build_dataset",
           "--config", os.path.join(vindr, "configs", "default.yaml"),
           "--disjoint-test", "--out", os.path.abspath(args.out)]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=vindr, check=True)

    from evidencekd.data.splits import split_manifest
    from evidencekd.data.vindr_processed import load_images, splits_from_images

    images = load_images(args.out)
    splits = splits_from_images(images)
    manifest = split_manifest(splits)
    manifest["source"] = "vindr_cxr_vqa.build_dataset --disjoint-test (seed 42, val_ratio 0.15, test_size 300)"
    os.makedirs(os.path.join(REPO, "reports"), exist_ok=True)
    path = os.path.join(REPO, "reports", "split_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"split manifest -> {path}: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
