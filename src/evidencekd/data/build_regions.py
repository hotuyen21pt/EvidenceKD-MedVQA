"""Write the derived data every later stage reads (design §7.1, §10):

    derived/anatomy.json              per image: lungs/heart/thorax boxes (txrv or fallback)
    derived/regions_train.json        annotation regions (QA+Full) for the training images
    derived/regions_{val,test}.json   detector regions, from outputs/detections_{split}.json
    derived/regions_{split}_annotation.json   annotation regions for val/test (oracle analyses)
    derived/background.json           approved background sentences per finding
    derived/negatives.jsonl           synthetic "No" questions (train only)
    derived/labels_train.jsonl        QALabels per training sample

    python -m evidencekd.data.build_regions --config configs/decision.yaml [--no-anatomy]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from evidencekd.data.anatomy import anatomy_for, load_anatomy, segment_images
from evidencekd.data.negatives import NegativeSpec, negative_samples, normal_image_records, write_negatives
from evidencekd.data.qa_labels import labels_for, negative_labels
from evidencekd.data.regions import regions_from_annotation, regions_from_detections, save_region_sets
from evidencekd.data.vindr_processed import load_images, load_samples, splits_from_images
from evidencekd.eval.vindr_bridge import load_detections
from evidencekd.facts.build import build_background, save_background
from evidencekd.utils.config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/decision.yaml")
    ap.add_argument("--no-anatomy", action="store_true", help="skip torchxrayvision; fallback anatomy")
    ap.add_argument("--detections-dir", default=None, help="where detections_{split}.json live (default output_dir)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    proc, ds, derived = cfg.paths.processed_dir, cfg.paths.dataset_dir, cfg.paths.derived_dir
    os.makedirs(derived, exist_ok=True)
    images = load_images(proc)
    splits = splits_from_images(images)

    # --- anatomy ---------------------------------------------------------
    apath = os.path.join(derived, "anatomy.json")
    if cfg.data.get("use_anatomy", True) and not args.no_anatomy:
        items = [(rec.image_id, os.path.join(ds, rec.image_path), rec.width, rec.height)
                 for rec in images.values()]
        missing = [it for it in items if it[0] not in load_anatomy(apath)]
        if missing:
            segment_images(items, out_path=apath)
    anat = load_anatomy(apath)
    print(f"[regions] anatomy cache: {len(anat)} images ({'fallback for the rest' if len(anat) < len(images) else 'complete'})")

    # --- regions -----------------------------------------------------------
    max_regions = int(cfg.data.max_regions)
    for split in ("train", "val", "test"):
        ids = splits.of(split)
        ann = [regions_from_annotation(images[i], anatomy=anatomy_for(anat, i, images[i].width, images[i].height),
                                       max_regions=max_regions) for i in ids]
        suffix = "" if split == "train" else "_annotation"
        save_region_sets(ann, os.path.join(derived, f"regions_{split}{suffix}.json"))
        if split != "train":
            dpath = os.path.join(args.detections_dir or cfg.paths.output_dir, f"detections_{split}.json")
            if os.path.isfile(dpath):
                dets = load_detections(dpath)
                det_sets = [regions_from_detections(i, images[i].width, images[i].height, dets.get(i, ()),
                                                    anatomy=anatomy_for(anat, i, images[i].width, images[i].height),
                                                    max_regions=max_regions,
                                                    score_threshold=float(cfg.data.detector_score_threshold))
                            for i in ids]
                save_region_sets(det_sets, os.path.join(derived, f"regions_{split}.json"))
                n_empty = sum(1 for s in det_sets if not s.finding_regions)
                print(f"[regions] {split}: detector regions for {len(det_sets)} images ({n_empty} without any box)")
            else:
                print(f"[regions] {split}: {dpath} missing — run the stage-0 predict first; "
                      f"only the annotation regions were written")
        print(f"[regions] {split}: annotation regions for {len(ann)} images")

    # --- background, negatives, labels ---------------------------------------
    train = load_samples(proc, "train")
    save_background(build_background(train), os.path.join(derived, "background.json"))

    negs = []
    if cfg.data.get("negatives", True) and cfg.data.mode == "qa_full":
        normals = normal_image_records(ds, exclude=set(images))
        spec = NegativeSpec(per_positive_ratio=float(cfg.data.get("negatives_ratio", 0.5)), seed=int(cfg.train.seed))
        negs = negative_samples(train, images, normals, spec)
        info = write_negatives(negs, os.path.join(derived, "negatives.jsonl"))
        print(f"[regions] negatives: {info['n_negatives']} ({len(normals)} normal images available)")

    from evidencekd.data.regions import load_region_sets
    rs_train = load_region_sets(os.path.join(derived, "regions_train.json"))
    with open(os.path.join(derived, "labels_train.jsonl"), "w", encoding="utf-8") as f:
        n_with_target = 0
        for s in train:
            rec = images[s.image_id]
            lab = labels_for(s, rec, rs_train[s.image_id], anatomy=anatomy_for(anat, s.image_id, rec.width, rec.height),
                             mode=cfg.data.mode, iou_threshold=float(cfg.data.region_iou_threshold))
            n_with_target += bool(lab.target_regions)
            f.write(json.dumps(lab.as_dict()) + "\n")
        for s in negs:
            lab = negative_labels(s.sample_id, s.type, s.gt_finding, images.get(s.image_id))
            f.write(json.dumps(lab.as_dict()) + "\n")
    print(f"[regions] labels: {len(train)} positives ({n_with_target} with a target region) + {len(negs)} negatives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
