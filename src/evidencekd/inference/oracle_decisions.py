"""Decision-branch outputs synthesised from the annotation — the pipeline's
self-test (design §7.7: the oracle through the bridge must be perfect).

Writes ``outputs/decision_{split}_oracle.jsonl`` in the exact shape
``models.decision_branch.predict`` writes, using ``regions_{split}_annotation.json``:
finding probabilities 1/0 from the fused boxes, relevance/pointer mass on
the region(s) overlapping ``gt_location_fixed``.

    python -m evidencekd.inference.oracle_decisions --config configs/decision.yaml --split val
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from evidencekd.data.anatomy import anatomy_for, load_anatomy
from evidencekd.data.qa_labels import COUNT_CLASSES, QUESTION_TYPES, SIDES, geometric_side_count
from evidencekd.data.regions import load_region_sets, target_region_ids
from evidencekd.data.vindr_processed import load_images, load_samples
from evidencekd.labels import VQA_CLASSES
from evidencekd.utils.config import load_config


def write_oracle(cfg, *, split: str, out_path: str) -> str:
    images = load_images(cfg.paths.processed_dir)
    samples = load_samples(cfg.paths.processed_dir, split)
    rs_all = load_region_sets(os.path.join(cfg.paths.derived_dir, f"regions_{split}_annotation.json"))
    anat = load_anatomy(os.path.join(cfg.paths.derived_dir, "anatomy.json"))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            rec = images[s.image_id]; rs = rs_all[s.image_id]
            present = set(rec.vqa_classes_present) | {s.gt_finding}
            ids = [r.id for r in rs.finding_regions]
            tgt = set(target_region_ids(rs, s.gt_box, iou_threshold=float(cfg.data.region_iou_threshold)))
            rel = [1.0 if rid in tgt else 0.0 for rid in ids]
            if tgt:
                ptr = [0.95 / len(tgt) if rid in tgt else 0.0 for rid in ids] + [0.05]
            else:
                ptr = [0.0] * len(ids) + [1.0]
            side_i, count_i = geometric_side_count(rec, s.gt_finding, anatomy_for(anat, s.image_id, rec.width, rec.height))
            side_probs = {x: (1.0 if side_i >= 0 and SIDES[side_i] == x else 0.0) for x in SIDES}
            count_probs = [1.0 if (count_i >= 0 and i == count_i) else 0.0 for i in range(COUNT_CLASSES)]
            row = {"sample_id": s.sample_id, "image_id": s.image_id, "region_ids": ids, "regions_source": "annotation",
                   "question_type": s.type if s.type in QUESTION_TYPES else "Is_there",
                   "qtype_probs": [1.0 if t == s.type else 0.0 for t in QUESTION_TYPES],
                   "finding_probs": {c: (1.0 if c in present else 0.0) for c in VQA_CLASSES},
                   "yes_prob": 1.0, "side_probs": side_probs, "count_probs": count_probs,
                   "relevance": rel, "pointer": ptr}
            f.write(json.dumps(row) + "\n"); n += 1
    print(f"[oracle] {n} questions -> {out_path}")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/decision.yaml")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    write_oracle(cfg, split=args.split, out_path=args.out or os.path.join(cfg.paths.output_dir, f"decision_{args.split}_oracle.jsonl"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
