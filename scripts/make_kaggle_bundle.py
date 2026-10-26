"""Package this repository for Kaggle.

    python scripts/make_kaggle_bundle.py --part code      # ~45 MB: src + vindr_cxr_vqa + configs + scripts + notebooks + data/processed
    python scripts/make_kaggle_bundle.py --part images    # the 4,394 VQA PNGs + CSVs (~360 MB); --all-images for all 15,000 (~1.3 GB)
    python scripts/make_kaggle_bundle.py --part weights   # COCO Faster R-CNN weights (160 MB)
    python scripts/make_kaggle_bundle.py --part all       # everything in one zip
    python scripts/make_kaggle_bundle.py --part repo      # snapshot of the repository itself (no data/outputs)

Upload each zip as a Kaggle Dataset; Kaggle unpacks it, and the notebooks'
setup cell walks /kaggle/input to find whichever layout results. The images
part is layout-compatible with the `vpa-repo` dataset already on Kaggle, so
it only needs uploading when that dataset is not attached.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VINDR = os.path.normpath(os.path.join(REPO, "..", "VinDR-CXR-VQA"))

CODE_ITEMS = ("src", "configs", "scripts", "notebooks", "docs", "pyproject.toml", "README.md", "reports")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ipynb_checkpoints", "*.egg-info")
PROCESSED_FILES = ("train.jsonl", "val.jsonl", "test.jsonl", "images.json", "build_summary.json")
DATASET_FILES = ("annotations_train.csv", "image_labels_train.csv")


def copy_code(stage: str) -> None:
    for item in CODE_ITEMS:
        s = os.path.join(REPO, item)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(stage, item), ignore=IGNORE, dirs_exist_ok=True)
        elif os.path.isfile(s):
            shutil.copy2(s, stage)
    # the sibling package: data readers + the Tables 4-6 scorer
    src = os.path.join(VINDR, "vindr_cxr_vqa")
    if not os.path.isdir(src):
        raise SystemExit(f"vindr_cxr_vqa not found at {src}")
    shutil.copytree(src, os.path.join(stage, "vindr_cxr_vqa"), ignore=IGNORE, dirs_exist_ok=True)
    vcfg = os.path.join(VINDR, "configs", "default.yaml")
    os.makedirs(os.path.join(stage, "vindr_configs"), exist_ok=True)
    shutil.copy2(vcfg, os.path.join(stage, "vindr_configs", "default.yaml"))


def copy_processed(stage: str, processed_dir: str) -> set[str]:
    dst = os.path.join(stage, "data", "processed")
    os.makedirs(dst, exist_ok=True)
    ids: set[str] = set()
    for f in PROCESSED_FILES:
        p = os.path.join(processed_dir, f)
        if not os.path.isfile(p):
            raise SystemExit(f"{p} missing — run `python scripts/build_data.py` first")
        shutil.copy2(p, dst)
    with open(os.path.join(processed_dir, "images.json"), encoding="utf-8") as f:
        data = json.load(f)
    for rec in (data.values() if isinstance(data, dict) else data):
        ids.add(rec["image_id"])
    return ids


def copy_images(stage: str, dataset_dir: str, image_ids: set[str] | None) -> int:
    src_dir = os.path.join(dataset_dir, "train_data", "train")
    dst_dir = os.path.join(stage, "dataset", "train_data", "train")
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for name in os.listdir(src_dir):
        if not name.endswith(".png"):
            continue
        if image_ids is not None and name[:-4] not in image_ids:
            continue
        shutil.copy2(os.path.join(src_dir, name), dst_dir)
        n += 1
    shutil.copy2(os.path.join(src_dir, "resize_factors.csv"), dst_dir)
    for f in DATASET_FILES:
        shutil.copy2(os.path.join(dataset_dir, f), os.path.join(stage, "dataset"))
    return n


def copy_weights(stage: str, weights_dir: str) -> list[str]:
    files = [f for f in os.listdir(weights_dir) if f.endswith(".pth")] if os.path.isdir(weights_dir) else []
    if not files:
        raise SystemExit(f"no .pth under {weights_dir} — run `python scripts/fetch_weights.py`")
    os.makedirs(os.path.join(stage, "weights"), exist_ok=True)
    for f in files:
        shutil.copy2(os.path.join(weights_dir, f), os.path.join(stage, "weights"))
    return files


def write_manifest(stage: str, part: str, info: dict) -> None:
    lines = [f"# EvidenceKD-MedVQA Kaggle bundle — part `{part}`", "",
             "Attach this dataset to `notebooks/kaggle_stage0_detector.ipynb` (import it into",
             "Kaggle via File -> Import Notebook) and Run all. The setup cell finds:", "",
             "| marker | provides |", "|---|---|",
             "| `src/evidencekd` + `vindr_cxr_vqa/` | code |",
             "| `data/processed/train.jsonl` | the built dataset (disjoint test) |",
             "| `dataset/train_data/train` | VinDr PNGs + CSVs |",
             "| `weights/*.pth` | COCO Faster R-CNN init |", ""]
    for k, v in info.items():
        lines.append(f"- {k}: {v}")
    with open(os.path.join(stage, "KAGGLE_README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def zip_stage(stage: str, out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for cur, _dirs, files in os.walk(stage):
            for name in files:
                p = os.path.join(cur, name)
                arc = os.path.relpath(p, stage).replace(os.sep, "/")
                ctype = zipfile.ZIP_STORED if name.endswith((".png", ".pth", ".zip")) else zipfile.ZIP_DEFLATED
                zf.write(p, arc, compress_type=ctype)
    print(f"{os.path.getsize(out) / 2**20:8.1f} MB  {out}")


def repo_snapshot(out: str) -> None:
    """Zip the repository as git sees it (tracked + untracked, not ignored)."""
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                       cwd=REPO, capture_output=True, text=True, check=True)
    files = [f for f in r.stdout.splitlines() if f and os.path.isfile(os.path.join(REPO, f))]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(os.path.join(REPO, f), "EvidenceKD-MedVQA/" + f)
    print(f"{os.path.getsize(out) / 2**20:8.1f} MB  {out}  ({len(files)} files)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", default="code", choices=["code", "images", "weights", "all", "repo"])
    ap.add_argument("--dataset-dir", default=os.path.join(VINDR, "dataset"))
    ap.add_argument("--processed-dir", default=os.path.join(REPO, "data", "processed"))
    ap.add_argument("--weights-dir", default=os.path.join(REPO, "weights"))
    ap.add_argument("--all-images", action="store_true", help="every PNG of train_data/train, not only the 4,394 VQA ones")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-stage", action="store_true")
    args = ap.parse_args(argv)

    if args.part == "repo":
        repo_snapshot(args.out or os.path.join(REPO, "outputs", "evidencekd-repo.zip"))
        return 0

    stage = os.path.join(REPO, "outputs", f"kaggle_stage_{args.part}")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    info: dict = {}
    ids = copy_processed(stage, args.processed_dir) if args.part in ("code", "all", "images") else None
    if args.part in ("code", "all"):
        copy_code(stage)
        info["code"] = "src/evidencekd + vindr_cxr_vqa + configs + scripts + notebooks"
        info["processed"] = f"{len(ids)} images indexed in data/processed"
    if args.part in ("images", "all"):
        n = copy_images(stage, args.dataset_dir, None if args.all_images else ids)
        info["images"] = f"{n} PNG under dataset/train_data/train ({'all' if args.all_images else 'VQA only'})"
        if args.part == "images":
            shutil.rmtree(os.path.join(stage, "data"))  # images part carries no JSONL
    if args.part in ("weights", "all"):
        info["weights"] = ", ".join(copy_weights(stage, args.weights_dir))
    write_manifest(stage, args.part, info)
    suffix = {"code": "-code", "images": "-images" + ("-all" if args.all_images else ""),
              "weights": "-weights", "all": ""}[args.part]
    out = args.out or os.path.join(REPO, "outputs", f"evidencekd-kaggle{suffix}.zip")
    zip_stage(stage, out)
    if not args.keep_stage:
        shutil.rmtree(stage, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
