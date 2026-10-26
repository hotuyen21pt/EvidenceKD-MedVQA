"""The bridge that keeps VinDr-CXR-VQA's purpose (design §7.7).

Everything EvidenceKD produces is turned into the prediction string VinDr's
scorer grades — ``"{answer} <loc_xxx_yyy_xxx_yyy>. Reasoning: ..."`` — and
written in the JSONL schema of ``vindr_cxr_vqa.infer`` so that
``python -m vindr_cxr_vqa.evaluate --pred ... --pred-space norm1000`` scores
Tables 4, 5 and 6 without a single change.

Two producers live here today:

* :func:`oracle_rows` — replays the answer key; the pipeline self-test.
* :func:`detector_rows` — baseline **B7** of the spec: the question names a
  finding, the detector's best box of that class answers it. This is the
  first number the Kaggle stage-0 notebook can put next to the paper.

The full Proposed-S/L producer (``to_prediction``) joins once the decision
branch and generator exist; its contract is fixed by :func:`compose`.

Pure Python. ``<loc>`` formatting is three-digit ``norm1000`` so the string is
identical to what VinDr's fine-tuned model is trained to emit.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from evidencekd.data.boxes import Box, iou
from evidencekd.data.vindr_processed import ImageRecord, VQASample
from evidencekd.eval.detection import Detection
from evidencekd.labels import VQA_CLASSES

NORM_HI = 999


def to_norm1000(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Original pixel space -> integer 0..999, clamped (mirrors vindr boxes.to_normalized)."""
    sx, sy = NORM_HI / max(width, 1), NORM_HI / max(height, 1)
    vals = (box.x_min * sx, box.y_min * sy, box.x_max * sx, box.y_max * sy)
    return tuple(int(min(NORM_HI, max(0, round(v)))) for v in vals)  # type: ignore[return-value]


def loc_token(box: Box, width: int, height: int) -> str:
    x0, y0, x1, y1 = to_norm1000(box, width, height)
    return f"<loc_{x0:03d}_{y0:03d}_{x1:03d}_{y1:03d}>"


def compose(answer: str, boxes: Sequence[tuple[Box, int, int]], reasoning: str = "") -> str:
    """``answer`` + one ``<loc>`` per box + optional reasoning tail.

    ``boxes`` is a sequence of ``(box, image_width, image_height)``. No box
    means no ``<loc>`` is emitted — the scorer's ``with_loc_pct`` must be able
    to see that the system declined to localise, never a fabricated box.
    """
    text = answer.strip().rstrip(".")
    locs = " ".join(loc_token(b, w, h) for b, w, h in boxes)
    if locs:
        text = f"{text} {locs}" if text else locs
    text = text + "."
    if reasoning.strip():
        text += " Reasoning: " + reasoning.strip()
    return text


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    image_id: str
    question: str
    prediction: str
    gt_finding: str
    gt_location: str

    def as_dict(self) -> dict:
        return {"sample_id": self.sample_id, "image_id": self.image_id,
                "question": self.question, "prediction": self.prediction,
                "gt_finding": self.gt_finding, "gt_location": self.gt_location}


def _row(sample: VQASample, prediction: str) -> PredictionRow:
    gt_loc = loc_token(sample.gt_box, sample.image_width, sample.image_height) if sample.gt_box else ""
    return PredictionRow(sample.sample_id, sample.image_id, sample.question,
                         prediction, sample.gt_finding, gt_loc)


def oracle_rows(samples: Sequence[VQASample]) -> list[PredictionRow]:
    """Replay the answer key: ``"{gt_finding} <loc_fixed>."``."""
    out = []
    for s in samples:
        boxes = [(s.gt_box, s.image_width, s.image_height)] if s.gt_box else []
        out.append(_row(s, compose(s.gt_finding, boxes)))
    return out


def detector_rows(
    samples: Sequence[VQASample],
    detections: Mapping[str, Sequence[Detection]],
    *,
    score_threshold: float = 0.3,
    max_boxes: int = 1,
    list_all_for: Sequence[str] = ("What", "Which", "How_many"),
    fallback_top_class: bool = True,
) -> list[PredictionRow]:
    """Baseline B7: answer with the detector's boxes of the class the question names.

    * ``Where`` / ``Is_there`` / ``Yes_No``: the top ``max_boxes`` boxes of the
      named class above ``score_threshold``; answer text = the class name.
    * ``What`` / ``Which`` / ``How_many`` (``list_all_for``): every class with a
      box above threshold, each with its best box — these questions ask for
      findings, so naming all of them is the detector's honest answer and is
      what lifts Table 4 recall past the ~4-per-image ceiling (DATA_NOTES §4).
    * A question naming no class, with ``fallback_top_class``: the highest
      scoring class on the image.
    * Nothing above threshold: the class name alone, no ``<loc>``.
    """
    out = []
    for s in samples:
        dets = sorted(detections.get(s.image_id, ()), key=lambda d: -d.score)
        dets = [d for d in dets if d.score >= score_threshold]
        w, h = s.image_width, s.image_height
        if s.type in list_all_for:
            best_per_class: dict[str, Detection] = {}
            for d in dets:
                best_per_class.setdefault(d.class_name, d)
            names = [c for c in VQA_CLASSES if c in best_per_class]
            if not names:
                out.append(_row(s, compose(s.question_finding or "No finding", [])))
                continue
            text = ", ".join(names)
            boxes = [(best_per_class[c].box, w, h) for c in names]
            out.append(_row(s, compose(text, boxes)))
            continue
        cls = s.question_finding
        if cls is None and fallback_top_class and dets:
            cls = dets[0].class_name
        if cls is None:
            out.append(_row(s, compose("", [])))
            continue
        chosen = [d for d in dets if d.class_name == cls][:max_boxes]
        out.append(_row(s, compose(cls, [(d.box, w, h) for d in chosen])))
    return out


def grounding_success(
    samples: Sequence[VQASample],
    detections: Mapping[str, Sequence[Detection]],
    *,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> dict:
    """Spec Table B, Grounding@IoU and mIoU, per question against ``gt_location_fixed``.

    The answering box is the detector's best box of the class the question
    names (B7 rule). Questions that name no class, or whose image has no box
    of that class, count as misses — they are the detector's failures to
    answer, not excluded cases.
    """
    n = hit = 0
    ious = []
    per_type: dict[str, list[float]] = {}
    for s in samples:
        if s.gt_box is None:
            continue
        n += 1
        cls = s.question_finding or s.gt_finding
        cands = [d for d in detections.get(s.image_id, ())
                 if d.class_name == cls and d.score >= score_threshold]
        v = 0.0
        if cands:
            top = max(cands, key=lambda d: d.score)
            v = iou(top.box, s.gt_box)
        ious.append(v)
        per_type.setdefault(s.type, []).append(v)
        hit += v >= iou_threshold
    return {
        "iou_threshold": iou_threshold,
        "n_questions": n,
        "grounding_at_iou": hit / n if n else None,
        "miou": sum(ious) / n if n else None,
        "per_type": {t: {"n": len(v), "grounding_at_iou": sum(x >= iou_threshold for x in v) / len(v),
                         "miou": sum(v) / len(v)} for t, v in sorted(per_type.items())},
    }


def write_rows(rows: Sequence[PredictionRow], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")


def load_detections(path: str) -> dict[str, list[Detection]]:
    """``detections_{split}.json`` written by ``models.detector`` -> Detection lists."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[Detection]] = {}
    for image_id, dets in data["detections"].items():
        out[image_id] = [
            Detection(Box(*map(float, d["bbox_orig"]), class_name=d["class_name"]),
                      float(d["score"]))
            for d in dets]
    return out


def ground_truth_boxes(images: Mapping[str, ImageRecord], image_ids: Sequence[str],
                       *, source: str = "fused") -> dict[str, list[Box]]:
    """Per-image GT for mAP: ``fused`` consensus boxes (default) or ``raw``."""
    out = {}
    for iid in image_ids:
        rec = images[iid]
        if source == "raw":
            out[iid] = [b for b in rec.boxes_raw if b.class_name in VQA_CLASSES]
        else:
            out[iid] = [f.box for f in rec.boxes_fused if f.box.class_name in VQA_CLASSES]
    return out
