"""Score a stage-0 detector: mAP, Grounding@IoU (spec Table B) and — through
the bridge — VinDr's Tables 4, 5 and 6 for the B7 detector-only baseline.

    python -m evidencekd.eval.stage0 --config configs/detector.yaml \
        --detections outputs/detections_val.json --split val

Writes ``reports/stage0_{split}.json`` plus the bridge's prediction JSONL and
then calls ``vindr_cxr_vqa.evaluate`` on it (``--pred-space norm1000``), so the
three VinDr tables come from VinDr's own scorer. ``--oracle`` replays the
answer key through the same bridge: it must give VQA precision 1.0 and
localization 100 % at IoU >= 0.5, or the bridge is broken.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from evidencekd.data.vindr_processed import load_images, load_samples, splits_from_images
from evidencekd.eval.detection import mean_average_precision
from evidencekd.eval.vindr_bridge import (detector_rows, ground_truth_boxes, grounding_success,
                                          load_detections, oracle_rows, write_rows)
from evidencekd.labels import VQA_CLASSES
from evidencekd.utils.config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/detector.yaml")
    ap.add_argument("--detections", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--vindr-config", default=None,
                    help="vindr_cxr_vqa config whose paths point at this processed dir")
    ap.add_argument("--skip-vindr", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = args.out_dir or cfg.paths.get("reports_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)

    images = load_images(cfg.paths.processed_dir)
    splits = splits_from_images(images)
    samples = load_samples(cfg.paths.processed_dir, args.split)
    tag = "oracle" if args.oracle else "b7"
    report: dict = {"split": args.split, "source": tag, "n_questions": len(samples),
                    "n_images": len(splits.of(args.split))}

    if args.oracle:
        rows = oracle_rows(samples)
    else:
        if not args.detections:
            ap.error("--detections is required unless --oracle")
        dets = load_detections(args.detections)
        thr = float(cfg.eval.get("score_threshold", 0.3))
        gt = ground_truth_boxes(images, splits.of(args.split), source=cfg.eval.get("gt_source", "fused"))
        report["map"] = mean_average_precision(
            {k: [d for d in v if d.score >= 0.0] for k, v in dets.items() if k in gt}, gt, VQA_CLASSES)
        report["grounding"] = grounding_success(
            samples, dets, iou_threshold=float(cfg.eval.get("iou_threshold", 0.5)),
            score_threshold=thr)
        report["score_threshold"] = thr
        rows = detector_rows(samples, dets, score_threshold=thr,
                             max_boxes=int(cfg.eval.get("max_boxes_per_question", 1)))

    pred_path = os.path.join(out_dir, f"preds_{args.split}_{tag}.jsonl")
    write_rows(rows, pred_path)
    report["n_predictions"] = len(rows)
    report["with_loc_pct"] = round(100.0 * sum("<loc_" in r.prediction for r in rows) / max(len(rows), 1), 2)
    report["prediction_sample"] = rows[0].prediction if rows else ""

    if not args.skip_vindr:
        vindr_out = os.path.join(out_dir, f"eval_{args.split}_{tag}.json")
        # vindr evaluate reads paths from ITS config; we pass ours through a
        # generated config that points processed_dir at our data.
        vcfg = args.vindr_config or _write_vindr_config(cfg, out_dir)
        cmd = [sys.executable, "-m", "vindr_cxr_vqa.evaluate", "--config", vcfg,
               "--pred", pred_path, "--split", args.split, "--pred-space", "norm1000",
               "--match", "any", "--gt-source", "raw", "--out", vindr_out]
        print("$", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout[-6000:])
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-4000:])
            report["vindr_tables"] = None
            report["vindr_error"] = r.stderr[-2000:]
        else:
            with open(vindr_out, encoding="utf-8") as f:
                v = json.load(f)
            report["vindr_tables"] = {k: v.get(k) for k in ("format", "vqa", "detection", "localization")}

    path = os.path.join(out_dir, f"stage0_{args.split}_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _print_summary(report)
    print(f"[stage0] -> {path}")
    if args.oracle and report.get("vindr_tables"):
        vq = report["vindr_tables"]["vqa"]["precision"]
        loc = report["vindr_tables"]["localization"]["iou_ge_0.5_pct"]
        ok = abs(vq - 1.0) < 1e-6 and loc >= 99.9
        print(f"[stage0] oracle bridge check: precision {vq}, IoU>=0.5 {loc}% -> {'OK' if ok else 'BROKEN'}")
        return 0 if ok else 1
    return 0


def _write_vindr_config(cfg, out_dir: str) -> str:
    """A vindr_cxr_vqa config whose paths point at this repo's data."""
    path = os.path.join(out_dir, "vindr_eval.yaml")
    lines = [
        "paths:",
        f"  dataset_dir: {cfg.paths.dataset_dir}",
        f"  processed_dir: {cfg.paths.processed_dir}",
        f"  output_dir: {os.path.abspath(out_dir).replace(os.sep, '/')}",
        "  vqa_json: data/raw/data_v1.json",
        "data:",
        "  box_source: fixed",
        "  loc_format: norm1000",
        "eval:",
        "  iou_threshold: 0.3",
        "  good_localization_iou: 0.5",
        "  acceptable_localization_iou: 0.3",
        "  match: any",
        "  max_new_tokens: 256",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _print_summary(r: dict) -> None:
    print(f"\n=== stage 0 · {r['split']} · {r['source']} ===")
    if "map" in r:
        m = r["map"]
        print(f"mAP@.5 {m['map_50']:.4f}   mAP@[.5:.95] {m['map_50_95']:.4f}   "
              f"({m['classes_scored']} classes, {m['n_gt_boxes']} GT boxes, {m['n_detections']} detections)")
    if "grounding" in r:
        g = r["grounding"]
        print(f"Grounding@{g['iou_threshold']} {g['grounding_at_iou']:.4f}   mIoU {g['miou']:.4f}   (n={g['n_questions']})")
    print(f"predictions with <loc>: {r['with_loc_pct']}%   e.g. {r['prediction_sample'][:100]}")
    t = r.get("vindr_tables")
    if t:
        v, d, l = t["vqa"], t["detection"], t["localization"]
        print(f"Table 4  acc {v['accuracy']:.3f}  P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}")
        print(f"Table 5  P {d['precision']:.3f}  R {d['recall']:.3f}  F1 {d['f1']:.3f}  mIoU(TP) {d['miou_tp']:.3f}")
        print(f"Table 6  IoU>=.5 {l['iou_ge_0.5_pct']}%  IoU>=.3 {l['iou_ge_0.3_pct']}%  mean IoU {l['mean_iou_all_preds']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
