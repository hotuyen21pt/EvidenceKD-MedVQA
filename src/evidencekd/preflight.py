"""Check everything the stage-0 notebook needs before it spends GPU time.

Mirrors ``vindr_cxr_vqa.preflight``: packages, GPU, data, weights, and that
the sibling ``vindr_cxr_vqa`` package (the scorer) is importable. Exits
non-zero on any FAIL. Standard library only; torch is probed, not imported
at module level.

    python -m evidencekd.preflight --config configs/detector.yaml
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

from evidencekd.utils.config import load_config

REQUIRED = {"torch": "2.2", "torchvision": "0.17", "PIL": None}
OPTIONAL = {"yaml": None}


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("+")[0].split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def check_packages(report: list[tuple[str, str, str]]) -> None:
    for name, minimum in {**REQUIRED, **OPTIONAL}.items():
        required = name in REQUIRED
        try:
            mod = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            report.append(("FAIL" if required else "WARN", f"package {name}", f"not importable: {e}"))
            continue
        ver = getattr(mod, "__version__", "?")
        if minimum and ver != "?" and _version_tuple(ver) < _version_tuple(minimum):
            report.append(("FAIL", f"package {name}", f"{ver} < {minimum}"))
        else:
            report.append(("OK", f"package {name}", ver))


def check_gpu(report: list[tuple[str, str, str]]) -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        report.append(("FAIL", "gpu", "torch missing"))
        return
    if not torch.cuda.is_available():
        report.append(("WARN", "gpu", "no CUDA device — training will be very slow"))
        return
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / 2**30
    status = "OK" if gb >= 12 else "WARN"
    report.append((status, "gpu", f"{props.name}, {gb:.1f} GB, {torch.cuda.device_count()} device(s)"))


def check_data(cfg, report: list[tuple[str, str, str]]) -> None:
    proc = cfg.paths.processed_dir
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl", "images.json"):
        p = os.path.join(proc, fname)
        report.append(("OK" if os.path.isfile(p) else "FAIL", f"data {fname}", p))
    ds = cfg.paths.dataset_dir
    img_dir = os.path.join(ds, "train_data", "train")
    if not os.path.isdir(img_dir):
        report.append(("FAIL", "images", f"{img_dir} not a directory"))
        return
    n_png = sum(1 for f in os.listdir(img_dir) if f.endswith(".png"))
    report.append(("OK" if n_png > 0 else "FAIL", "images", f"{n_png} PNG under {img_dir}"))
    for fname in ("annotations_train.csv", "image_labels_train.csv"):
        p = os.path.join(ds, fname)
        report.append(("OK" if os.path.isfile(p) else "FAIL", f"dataset {fname}", p))
    rf = os.path.join(img_dir, "resize_factors.csv")
    report.append(("OK" if os.path.isfile(rf) else "FAIL", "resize_factors.csv", rf))
    # the first 50 train records must resolve to a file
    tr = os.path.join(proc, "train.jsonl")
    if os.path.isfile(tr):
        missing = 0
        with open(tr, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                rec = json.loads(line)
                if not os.path.isfile(os.path.join(ds, rec["image_path"])):
                    missing += 1
        report.append(("OK" if missing == 0 else "FAIL", "image_path resolves",
                       f"{50 - missing}/50 of the first train records"))


def check_weights(cfg, report: list[tuple[str, str, str]]) -> None:
    w = cfg.model.get("coco_weights") or ""
    if not w:
        report.append(("WARN", "coco weights", "none configured — backbone trains from scratch"))
    elif os.path.isfile(w):
        report.append(("OK", "coco weights", f"{w} ({os.path.getsize(w) / 2**20:.0f} MB)"))
    else:
        report.append(("WARN", "coco weights", f"{w} missing — backbone trains from scratch"))


def check_scorer(report: list[tuple[str, str, str]]) -> None:
    try:
        import vindr_cxr_vqa.evaluate  # noqa: F401
        import vindr_cxr_vqa.vindr  # noqa: F401
        report.append(("OK", "vindr_cxr_vqa", "importable (Tables 4-6 scorer)"))
    except Exception as e:  # noqa: BLE001
        report.append(("FAIL", "vindr_cxr_vqa", f"not importable: {e}"))


def run(cfg) -> int:
    report: list[tuple[str, str, str]] = []
    check_packages(report)
    check_gpu(report)
    check_data(cfg, report)
    check_weights(cfg, report)
    check_scorer(report)
    width = max(len(r[1]) for r in report)
    for status, what, detail in report:
        print(f"[{status:4}] {what:<{width}}  {detail}")
    fails = [r for r in report if r[0] == "FAIL"]
    print(f"\n{len(fails)} FAIL, {sum(r[0] == 'WARN' for r in report)} WARN")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/detector.yaml")
    args = ap.parse_args(argv)
    return run(load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
