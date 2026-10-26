"""Table F — faithfulness by intervention (spec §6 "Faithfulness"; design §9.2).

Hide the region the explanation cites and ask whether the decision branch's
answer changes; compare with four controls on the same question:

    cited      — the first finding region the pointer selected
    random     — a random finding region that was NOT cited
    uncited_top— the highest-relevance region that was NOT cited
    other_image— the cited region's box filled with the same box from another image
    anatomy    — an anatomy region (thorax) instead of a finding region

"Answer changes" = the asked finding's presence flips across the tuned
threshold, or the Yes/No head flips. Pixel masking on the 224 px image, so
the ROI features really lose the region.

    python -m evidencekd.eval.faithfulness --config configs/decision.yaml \
        --checkpoint outputs/decision/decision_stage2.pt --pipeline outputs/pipeline_test_template.jsonl --split test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

from evidencekd.data.boxes import Box
from evidencekd.data.regions import load_region_sets
from evidencekd.data.vindr_processed import load_samples
from evidencekd.inference.pipeline import load_thresholds
from evidencekd.labels import VQA_CLASSES, finding_in_text
from evidencekd.utils.config import load_config

CONTROLS = ("cited", "random", "uncited_top", "other_image", "anatomy")


def run(cfg, *, checkpoint: str, pipeline_path: str, split: str, limit: int | None = None, seed: int = 0) -> dict:
    import torch
    from PIL import Image

    from evidencekd.models.decision_branch import DecisionDataset, collate, load_for_inference
    from evidencekd.teacher.medgemma import mask_region

    rng = random.Random(seed)
    model, tok, dev = load_for_inference(cfg, checkpoint)
    base = "val" if split == "tune" else split
    samples = {s.sample_id: s for s in load_samples(cfg.paths.processed_dir, base)}
    region_sets = load_region_sets(os.path.join(cfg.paths.derived_dir, f"regions_{base}.json"))
    thr = load_thresholds(os.path.join(cfg.paths.get("reports_dir", "reports"), "thresholds.json"))
    rows = [json.loads(l) for l in open(pipeline_path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    flips = {c: 0 for c in CONTROLS}; counts = {c: 0 for c in CONTROLS}
    ds = DecisionDataset(list(samples.values()), None, region_sets, cfg.paths.dataset_dir, tok,
                         max_q_tokens=int(cfg.data.max_question_tokens))
    index = {s.sample_id: i for i, s in enumerate(ds.samples)}
    all_ids = list(region_sets)

    def presence(batch_item, image_tensor):
        batch_item = dict(batch_item); batch_item["image"] = image_tensor
        b = collate([batch_item])
        with torch.no_grad():
            out = model(b)
        asked = finding_in_text(batch_item["question"])
        p = torch.sigmoid(out["finding_logits"][0].float())
        yes = float(torch.softmax(out["yn_logits"][0].float(), -1)[1])
        pres = float(p[VQA_CLASSES.index(asked)]) >= thr["finding"] if asked in VQA_CLASSES else yes >= thr["yes"]
        return pres, yes >= thr["yes"]

    for r in rows:
        s = samples.get(r["sample_id"]); rs = region_sets.get(r["image_id"])
        if s is None or rs is None or r["sample_id"] not in index or not rs.finding_regions:
            continue
        cited = [rid for rid in r["pointer"]["region_ids"] if rid in rs.ids and rs.by_id(rid).kind == "finding"]
        if not cited:
            continue
        item = ds[index[r["sample_id"]]]
        img = Image.open(os.path.join(cfg.paths.dataset_dir, s.image_path)).convert("RGB")
        fx, fy = img.width / rs.width, img.height / rs.height
        base_pres, base_yes = presence(item, item["image"])

        def tensor_of(im):
            return torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(im.height, im.width, 3).permute(2, 0, 1).float().div(255)

        variants = {"cited": rs.by_id(cited[0]).box}
        uncited = [x for x in rs.finding_regions if x.id not in cited]
        if uncited:
            variants["random"] = rng.choice(uncited).box
            rel = dict(zip([x.id for x in rs.finding_regions], r.get("relevance", []) or [0.0] * len(rs.finding_regions)))
            variants["uncited_top"] = max(uncited, key=lambda x: rel.get(x.id, 0.0)).box
        if rs.anatomy_regions:
            variants["anatomy"] = rs.anatomy_regions[0].box
        other_id = rng.choice([i for i in all_ids if i != rs.image_id])
        other_s = next((x for x in samples.values() if x.image_id == other_id), None)
        for name, box in variants.items():
            pres, yes = presence(item, tensor_of(mask_region(img, box, fx, fy)))
            counts[name] += 1; flips[name] += (pres != base_pres) or (yes != base_yes)
        if other_s is not None:
            o = Image.open(os.path.join(cfg.paths.dataset_dir, other_s.image_path)).convert("RGB").resize(img.size)
            b = variants["cited"]
            x0, y0, x1, y1 = int(b.x_min * fx), int(b.y_min * fy), int(b.x_max * fx), int(b.y_max * fy)
            im2 = img.copy()
            if x1 > x0 and y1 > y0:
                im2.paste(o.crop((x0, y0, x1, y1)), (x0, y0))
            pres, yes = presence(item, tensor_of(im2))
            counts["other_image"] += 1; flips["other_image"] += (pres != base_pres) or (yes != base_yes)
    rep = {c: {"n": counts[c], "flip_rate": round(flips[c] / counts[c], 4) if counts[c] else None} for c in CONTROLS}
    strongest = max((rep[c]["flip_rate"] or 0.0) for c in CONTROLS if c != "cited")
    rep["faithful"] = (rep["cited"]["flip_rate"] or 0.0) > strongest
    print(json.dumps(rep, indent=1))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/decision.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rep = run(cfg, checkpoint=args.checkpoint, pipeline_path=args.pipeline, split=args.split, limit=args.limit)
    out = args.out or os.path.join(cfg.paths.get("reports_dir", "reports"), f"faithfulness_{args.split}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
