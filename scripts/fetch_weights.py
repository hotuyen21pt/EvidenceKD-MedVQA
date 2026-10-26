"""Download the COCO-pretrained Faster R-CNN weights for offline Kaggle use.

Kaggle training sessions have no internet, and torchvision would otherwise
try to fetch these on first use. Run once on a machine with internet, then
``scripts/make_kaggle_bundle.py --part weights`` packs the file.

    python scripts/fetch_weights.py [--out weights/]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

URL = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
SHA256_PREFIX = "258fb6c6"  # torchvision's hash prefix, in the filename


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="weights")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, os.path.basename(args.url))
    if os.path.isfile(dest) and os.path.getsize(dest) > 100 * 2**20:
        print(f"already present: {dest} ({os.path.getsize(dest) / 2**20:.0f} MB)")
    else:
        print(f"downloading {args.url} -> {dest}")

        def hook(blocks, bs, total):
            done = blocks * bs
            if total > 0 and blocks % 200 == 0:
                print(f"  {done / 2**20:6.0f} / {total / 2**20:.0f} MB", flush=True)

        urllib.request.urlretrieve(args.url, dest, reporthook=hook)
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    ok = digest.startswith(SHA256_PREFIX)
    print(f"sha256 {digest[:16]}...  {'OK' if ok else 'MISMATCH — delete and retry'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
