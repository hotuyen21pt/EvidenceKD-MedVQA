"""Spec §6-§7 tables A, B, C (+ ECE for F) from a pipeline output file.

    python -m evidencekd.eval.tables_spec --config configs/decision.yaml \
        --pipeline outputs/pipeline_test_template.jsonl --split test

Pure Python. Every value is computed from ``pipeline_{split}_{gen}.jsonl``
plus the release's ground truth; metrics that cannot be measured on this
data are written as ``null`` (N/A), never 0 (spec §5).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from evidencekd.data.anatomy import anatomy_for, load_anatomy
from evidencekd.data.boxes import Box, iou
from evidencekd.data.qa_labels import geometric_side_count, SIDES
from evidencekd.data.vindr_processed import load_images, load_samples
from evidencekd.labels import VQA_CLASSES, finding_in_text
from evidencekd.utils.config import load_config


def _f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def ece(probs, labels, bins: int = 10) -> float | None:
    if not probs:
        return None
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        tot += len(idx) / len(probs) * abs(conf - acc)
    return round(tot, 4)


def compute(cfg, *, pipeline_path: str, split: str, iou_threshold: float = 0.5) -> dict:
    proc, derived = cfg.paths.processed_dir, cfg.paths.derived_dir
    images = load_images(proc)
    base = "val" if split == "tune" else split
    samples = {s.sample_id: s for s in load_samples(proc, base)}
    anat = load_anatomy(os.path.join(derived, "anatomy.json"))
    rows = [json.loads(l) for l in open(pipeline_path, encoding="utf-8") if l.strip()]

    # ---- Table A ------------------------------------------------------------
    per_img_pred: dict[str, set] = {}
    side_ok = side_n = 0; count_err = []; count_n = 0
    yn_tp = yn_fp = yn_fn = 0; has_negative = False
    conf, lab = [], []
    for r in rows:
        s = samples.get(r["sample_id"])
        if s is None:
            continue
        per_img_pred.setdefault(s.image_id, set()).update(r["findings_present"])
        rec = images[s.image_id]
        gs, gc = geometric_side_count(rec, s.gt_finding, anatomy_for(anat, s.image_id, rec.width, rec.height))
        pred_side = next((x for x in SIDES if f" {x} " in f" {r['answer'].lower()} " or f"{x} side" in r["answer"].lower()), None)
        if s.type == "Where" and gs >= 0 and pred_side:
            side_n += 1; side_ok += pred_side == SIDES[gs]
        if s.type == "How_many" and gc >= 0:
            m = [int(t) for t in r["answer"].split() if t.isdigit()]
            if m:
                count_n += 1; count_err.append(abs(m[0] - gc))
        if s.type in ("Is_there", "Yes_No"):
            pred_yes = r["answer"].lower().startswith("yes")
            gt_yes = s.gt_box is not None
            has_negative |= not gt_yes
            yn_tp += pred_yes and gt_yes; yn_fp += pred_yes and not gt_yes; yn_fn += (not pred_yes) and gt_yes
        asked = finding_in_text(s.question) or s.gt_finding
        conf.append(float(r["scores"].get("finding", 0.0))); lab.append(1.0 if asked == s.gt_finding and s.gt_box is not None else 0.0)
    cls_tp = {c: 0 for c in VQA_CLASSES}; cls_fp = dict(cls_tp); cls_fn = dict(cls_tp)
    for iid, pred in per_img_pred.items():
        gt = set(images[iid].vqa_classes_present)
        for c in VQA_CLASSES:
            if c in pred and c in gt: cls_tp[c] += 1
            elif c in pred: cls_fp[c] += 1
            elif c in gt: cls_fn[c] += 1
    macro = [_f1(cls_tp[c], cls_fp[c], cls_fn[c]) for c in VQA_CLASSES if cls_tp[c] + cls_fn[c] > 0]
    table_a = {"yes_no_f1": round(_f1(yn_tp, yn_fp, yn_fn), 4) if has_negative else None,
               "yes_no_note": None if has_negative else "N/A: the release has no negative question",
               "finding_macro_f1": round(sum(macro) / len(macro), 4) if macro else None,
               "side_accuracy": round(side_ok / side_n, 4) if side_n else None, "side_n": side_n,
               "count_mae": round(sum(count_err) / count_n, 4) if count_n else None, "count_n": count_n,
               "n_images": len(per_img_pred)}

    # ---- Table B ------------------------------------------------------------
    ground = []; ag = []; ious = []
    for r in rows:
        s = samples.get(r["sample_id"])
        if s is None or s.gt_box is None:
            continue
        v = 0.0
        if r.get("answer_box"):
            v = iou(Box(*r["answer_box"], class_name=""), s.gt_box)
        ious.append(v); ground.append(v >= iou_threshold)
        correct = s.gt_finding in r.get("finding", []) or s.gt_finding.lower() in r["answer"].lower()
        ag.append(correct and v >= iou_threshold)
    table_b = {"grounding_at_iou": round(sum(ground) / len(ground), 4) if ground else None, "miou": round(sum(ious) / len(ious), 4) if ious else None,
               "answer_and_grounding": round(sum(ag) / len(ag), 4) if ag else None, "n": len(ground), "iou_threshold": iou_threshold}

    # ---- Table C (machine-scored) --------------------------------------------
    n_obs = n_sup = n_con = 0; covered = 0; cite_ok = cite_tot = 0; recall_hit = recall_n = 0; relevant = 0; n_samples = 0
    for r in rows:
        s = samples.get(r["sample_id"]); n_samples += 1
        obs = [c for c in r["claims"] if c["type"] == "observation"]
        covered += bool(obs)
        regions = {g["id"]: g for g in r["regions"]}
        asked = finding_in_text(r["question"]) or (s.gt_finding if s else None)
        raw_boxes = images[s.image_id].boxes_raw if s else ()
        for c in obs:
            n_obs += 1; n_sup += c.get("status") == "supported"; n_con += c.get("status") == "contradicted"
            if asked and asked.lower() in c["text"].lower():
                relevant += 1
            # citation precision: a cited finding region is correct when it
            # overlaps a radiologist box of its own class (the region is real),
            # so claims about co-occurring findings are judged on their own boxes
            for rid in c.get("evidence_ids", []):
                g = regions.get(rid)
                if g and g["kind"] == "finding":
                    cite_tot += 1
                    gb = Box(*g["bbox_orig"], class_name="")
                    cite_ok += any(b.class_name == g["label"] and iou(gb, b) >= iou_threshold for b in raw_boxes)
        if s and s.gt_box is not None:
            recall_n += 1
            cited = {rid for c in obs for rid in c.get("evidence_ids", [])}
            recall_hit += any(g["kind"] == "finding" and iou(Box(*g["bbox_orig"], class_name=""), s.gt_box) >= iou_threshold
                              for rid, g in regions.items() if rid in cited)
    table_c = {"supported_claim_precision": round(n_sup / n_obs, 4) if n_obs else None,
               "citation_precision": round(cite_ok / cite_tot, 4) if cite_tot else None,
               "citation_recall": round(recall_hit / recall_n, 4) if recall_n else None,
               "coverage": round(covered / n_samples, 4) if n_samples else None,
               "contradiction_rate": round(n_con / n_obs, 4) if n_obs else None,
               "question_relevance": round(relevant / n_obs, 4) if n_obs else None,
               "conciseness_obs_per_sample": round(n_obs / n_samples, 4) if n_samples else None,
               "scorer": "machine (rule or learned verifier) — not clinician-scored"}
    table_f = {"ece_finding": ece(conf, lab), "latency_s_per_question": round(sum(r.get("latency_s", 0) for r in rows) / max(len(rows), 1), 4)}
    return {"split": split, "pipeline": pipeline_path, "n_rows": len(rows), "table_a": table_a, "table_b": table_b,
            "table_c": table_c, "table_f": table_f}


def to_markdown(rep: dict) -> str:
    a, b, c, f = rep["table_a"], rep["table_b"], rep["table_c"], rep["table_f"]
    def v(x):
        return "N/A" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    lines = [f"### Spec tables · {rep['split']} · {os.path.basename(rep['pipeline'])}", "",
             "| Table A | Yes/No F1 | Finding macro-F1 | Side acc | Count MAE |", "|---|---|---|---|---|",
             f"| | {v(a['yes_no_f1'])} | {v(a['finding_macro_f1'])} | {v(a['side_accuracy'])} (n={a['side_n']}) | {v(a['count_mae'])} (n={a['count_n']}) |", "",
             "| Table B | Grounding@.5 | mIoU | Answer & grounding |", "|---|---|---|---|",
             f"| | {v(b['grounding_at_iou'])} | {v(b['miou'])} | {v(b['answer_and_grounding'])} |", "",
             "| Table C (machine) | Supported-claim P | Citation P | Citation R | Coverage | Contradiction | Q-relevance | Obs/sample |",
             "|---|---|---|---|---|---|---|---|",
             f"| | {v(c['supported_claim_precision'])} | {v(c['citation_precision'])} | {v(c['citation_recall'])} | {v(c['coverage'])} | {v(c['contradiction_rate'])} | {v(c['question_relevance'])} | {v(c['conciseness_obs_per_sample'])} |", "",
             f"Table F: ECE(finding) {v(f['ece_finding'])} · latency {v(f['latency_s_per_question'])} s/question"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/decision.yaml")
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rep = compute(cfg, pipeline_path=args.pipeline, split=args.split)
    out = args.out or os.path.join(cfg.paths.get("reports_dir", "reports"),
                                   f"spec_{os.path.basename(args.pipeline).replace('.jsonl', '')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    print(to_markdown(rep)); print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
