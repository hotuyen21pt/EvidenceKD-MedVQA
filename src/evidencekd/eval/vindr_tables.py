"""Score any bridge prediction file with VinDr-CXR-VQA's own scorer (Tables 4-6).

    python -m evidencekd.eval.vindr_tables --config configs/decision.yaml \
        --pred reports/preds_test_template.jsonl --split test [--paper]

Runs ``python -m vindr_cxr_vqa.evaluate`` as a subprocess on a generated
config that points at this repository's data, writes ``eval_<pred>.json``
next to the prediction file and prints the three tables beside the paper's
Pretrained / Fine-tuned rows. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from evidencekd.utils.config import load_config


def write_vindr_config(cfg, out_dir: str) -> str:
    path = os.path.join(out_dir, "vindr_eval.yaml")
    lines = ["paths:", f"  dataset_dir: {cfg.paths.dataset_dir}", f"  processed_dir: {cfg.paths.processed_dir}",
             f"  output_dir: {os.path.abspath(out_dir).replace(os.sep, '/')}", "  vqa_json: data/raw/data_v1.json",
             "data:", "  box_source: fixed", "  loc_format: norm1000",
             "eval:", "  iou_threshold: 0.3", "  good_localization_iou: 0.5", "  acceptable_localization_iou: 0.3",
             "  match: any", "  max_new_tokens: 256"]
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def evaluate(cfg, *, pred_path: str, split: str, out_path: str | None = None, match: str = "any",
             gt_source: str = "raw", python: str = sys.executable) -> dict | None:
    out_dir = os.path.dirname(os.path.abspath(pred_path))
    vcfg = write_vindr_config(cfg, out_dir)
    out_path = out_path or os.path.join(out_dir, "eval_" + os.path.basename(pred_path).replace(".jsonl", ".json"))
    base_split = "val" if split == "tune" else split
    cmd = [python, "-m", "vindr_cxr_vqa.evaluate", "--config", vcfg, "--pred", pred_path, "--split", base_split,
           "--pred-space", "norm1000", "--match", match, "--gt-source", gt_source, "--out", out_path]
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-6000:])
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:]); return None
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)


def table_rows(result: dict, name: str) -> str:
    v, d, l = result["vqa"], result["detection"], result["localization"]
    return (f"| {name} | {v['f1']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | {d['precision']:.3f} | {d['recall']:.3f} | "
            f"{d['miou_tp']:.3f} | {l['iou_ge_0.5_pct']:.1f} | {l['iou_ge_0.3_pct']:.1f} |")


def render(results: dict[str, dict], *, with_paper: bool = True) -> str:
    lines = ["| System | T4 F1 | T4 P | T4 R | T5 P | T5 R | T5 mIoU(TP) | T6 IoU≥.5 % | T6 IoU≥.3 % |", "|---|---|---|---|---|---|---|---|---|"]
    if with_paper:
        try:
            from vindr_cxr_vqa.results import PAPER
            for n in ("Pretrained", "Fine-tuned"):
                lines.append(table_rows({"vqa": PAPER["vqa"][n], "detection": PAPER["detection"][n], "localization": PAPER["localization"][n]},
                                        f"Paper {n} (val-659)"))
        except ImportError:
            pass
    for name, res in results.items():
        if res:
            lines.append(table_rows(res, name))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/decision.yaml")
    ap.add_argument("--pred", required=True, nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--match", default="any", choices=["any", "greedy"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    results = {}
    for p in args.pred:
        results[os.path.basename(p).replace(".jsonl", "") + f" ({args.split})"] = evaluate(cfg, pred_path=p, split=args.split, match=args.match)
    print("\n" + render(results))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
