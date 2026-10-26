"""Candidate evidence regions — the one data contract every stage shares.

A :class:`RegionSet` is the list of boxes a question can be answered with:
finding regions (``R01..``, from the radiologists' fused boxes at training
time in QA+Full mode, from the stage-0 detector at inference) plus the four
anatomy regions (``A_thorax, A_heart, A_lung_left, A_lung_right``) when a
segmentation is available. Region ids are what claims cite, what the pointer
selects, what KD-R masks — so the ORDER and the ids are fixed here once and
written to ``derived/regions_{split}.json``; nothing re-derives them.

All boxes are in original DICOM pixel space (design §5.1).
Pure Python.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from evidencekd.data.boxes import Box, iou
from evidencekd.data.vindr_processed import ImageRecord
from evidencekd.eval.detection import Detection
from evidencekd.facts.geometry import A_HEART, A_LUNG_LEFT, A_LUNG_RIGHT, A_THORAX, Anatomy
from evidencekd.labels import VQA_CLASSES

ANATOMY_IDS = (A_THORAX, A_HEART, A_LUNG_LEFT, A_LUNG_RIGHT)
KIND_FINDING = "finding"
KIND_ANATOMY = "anatomy"


@dataclass(frozen=True)
class Region:
    id: str
    kind: str                 # "finding" | "anatomy"
    label: str                # class name, or the anatomy id
    box: Box                  # original pixels; box.class_name == label
    score: float = 1.0        # detector score, or 1.0 for annotation
    n_raters: int = 0         # radiologists agreeing (annotation source only)

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "score": round(self.score, 4),
                "n_raters": self.n_raters,
                "bbox_orig": [self.box.x_min, self.box.y_min, self.box.x_max, self.box.y_max]}

    @staticmethod
    def from_dict(d: dict) -> "Region":
        x0, y0, x1, y1 = d["bbox_orig"]
        return Region(d["id"], d["kind"], d["label"],
                      Box(float(x0), float(y0), float(x1), float(y1), class_name=d["label"]),
                      float(d.get("score", 1.0)), int(d.get("n_raters", 0)))


@dataclass(frozen=True)
class RegionSet:
    image_id: str
    width: int
    height: int
    source: str                      # "annotation" | "detector"
    regions: tuple[Region, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.regions)

    @property
    def finding_regions(self) -> tuple[Region, ...]:
        return tuple(r for r in self.regions if r.kind == KIND_FINDING)

    @property
    def anatomy_regions(self) -> tuple[Region, ...]:
        return tuple(r for r in self.regions if r.kind == KIND_ANATOMY)

    def by_id(self, rid: str) -> Region:
        for r in self.regions:
            if r.id == rid:
                return r
        raise KeyError(rid)

    def mapping(self, *, kind: str | None = None) -> dict[str, Box]:
        return {r.id: r.box for r in self.regions if kind is None or r.kind == kind}

    def of_label(self, label: str) -> tuple[Region, ...]:
        return tuple(r for r in self.finding_regions if r.label == label)

    def as_dict(self) -> dict:
        return {"image_id": self.image_id, "width": self.width, "height": self.height,
                "source": self.source, "regions": [r.as_dict() for r in self.regions]}

    @staticmethod
    def from_dict(d: dict) -> "RegionSet":
        return RegionSet(d["image_id"], int(d["width"]), int(d["height"]), d["source"],
                         tuple(Region.from_dict(r) for r in d["regions"]))


# ---------------------------------------------------------------------------
def anatomy_regions(anatomy: Anatomy | None) -> list[Region]:
    """The anatomy boxes as regions, in the fixed order of ``ANATOMY_IDS``."""
    if anatomy is None:
        return []
    out = []
    for rid, box in ((A_THORAX, anatomy.thorax), (A_HEART, anatomy.heart),
                     (A_LUNG_LEFT, anatomy.lung_left), (A_LUNG_RIGHT, anatomy.lung_right)):
        if box is not None:
            out.append(Region(rid, KIND_ANATOMY, rid,
                              Box(box.x_min, box.y_min, box.x_max, box.y_max, class_name=rid), 1.0))
    return out


def regions_from_annotation(rec: ImageRecord, *, anatomy: Anatomy | None,
                            max_regions: int = 8, min_raters: int = 1) -> RegionSet:
    """QA+Full training regions: the fused radiologist boxes of the 14 classes.

    Ordered by (raters desc, area desc) and capped, so ``R01`` is the most
    agreed-on lesion — the same cap KD-R masks (design §7.1, §7.4).
    """
    cands = [f for f in rec.boxes_fused
             if f.box.class_name in VQA_CLASSES and f.n_raters >= min_raters and not f.box.is_degenerate()]
    cands.sort(key=lambda f: (-f.n_raters, -f.box.area, f.box.class_name))
    regs = [Region(f"R{i + 1:02d}", KIND_FINDING, f.box.class_name, f.box, 1.0, f.n_raters)
            for i, f in enumerate(cands[:max_regions])]
    return RegionSet(rec.image_id, rec.width, rec.height, "annotation",
                     tuple(regs + anatomy_regions(anatomy)))


def regions_from_detections(image_id: str, width: int, height: int,
                            detections: Sequence[Detection], *, anatomy: Anatomy | None,
                            max_regions: int = 8, score_threshold: float = 0.3,
                            nms_iou: float = 0.5) -> RegionSet:
    """Inference regions: detector boxes above threshold, class-aware NMS, capped by score."""
    dets = sorted((d for d in detections if d.score >= score_threshold), key=lambda d: -d.score)
    kept: list[Detection] = []
    for d in dets:
        if any(k.class_name == d.class_name and iou(k.box, d.box) >= nms_iou for k in kept):
            continue
        if d.box.is_degenerate():
            continue
        kept.append(d)
        if len(kept) >= max_regions:
            break
    regs = [Region(f"R{i + 1:02d}", KIND_FINDING, d.class_name, d.box, d.score) for i, d in enumerate(kept)]
    return RegionSet(image_id, width, height, "detector", tuple(regs + anatomy_regions(anatomy)))


def target_region_ids(regions: RegionSet, gt_box: Box | None, *, iou_threshold: float = 0.5,
                      same_class: bool = True) -> tuple[str, ...]:
    """Regions that answer the question: IoU >= thr with the supervision box.

    ``G_c`` of the pointer loss and the positive set of the relevance head
    and the alignment loss (spec §4.1, §4.3).
    """
    if gt_box is None:
        return ()
    out = []
    for r in regions.finding_regions:
        if same_class and r.label != gt_box.class_name:
            continue
        if iou(r.box, gt_box) >= iou_threshold:
            out.append(r.id)
    return tuple(out)


# ---------------------------------------------------------------------------
def save_region_sets(sets: Iterable[RegionSet], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({s.image_id: s.as_dict() for s in sets}, f)


def load_region_sets(path: str) -> dict[str, RegionSet]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: RegionSet.from_dict(v) for k, v in data.items()}
