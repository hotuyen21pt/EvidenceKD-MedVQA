"""Class-aware mean average precision for the detector (spec §6, Table B).

VOC-style all-point interpolated AP per class, one-to-one greedy matching by
descending score, averaged over classes that have at least one ground-truth
box. mAP@[.5:.95] averages ten IoU thresholds. Pure Python so it is tested on
the development machine; the detector itself lives in ``models.detector``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from evidencekd.data.boxes import Box, iou

IOU_RANGE = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float

    @property
    def class_name(self) -> str:
        return self.box.class_name


def _ap_from_pr(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """All-point interpolation (VOC 2010+)."""
    if not recalls:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def average_precision(
    detections: Mapping[str, Sequence[Detection]],
    ground_truth: Mapping[str, Sequence[Box]],
    class_name: str,
    iou_threshold: float,
) -> float | None:
    """AP of one class; ``None`` when the class has no ground-truth box."""
    gts = {img: [b for b in boxes if b.class_name == class_name]
           for img, boxes in ground_truth.items()}
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return None
    dets = [(img, d) for img, ds in detections.items()
            for d in ds if d.class_name == class_name]
    dets.sort(key=lambda t: -t[1].score)
    used: dict[str, list[bool]] = {img: [False] * len(v) for img, v in gts.items()}
    tp, fp = [], []
    for img, d in dets:
        cands = gts.get(img, [])
        best, best_j = 0.0, -1
        for j, g in enumerate(cands):
            v = iou(d.box, g)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= iou_threshold and not used[img][best_j]:
            used[img][best_j] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    recalls, precisions, ctp, cfp = [], [], 0, 0
    for t, f in zip(tp, fp):
        ctp += t; cfp += f
        recalls.append(ctp / n_gt)
        precisions.append(ctp / max(ctp + cfp, 1))
    return _ap_from_pr(recalls, precisions)


def mean_average_precision(
    detections: Mapping[str, Sequence[Detection]],
    ground_truth: Mapping[str, Sequence[Box]],
    classes: Sequence[str],
    *,
    iou_thresholds: Sequence[float] = IOU_RANGE,
) -> dict:
    """mAP@.5, mAP@[.5:.95] and per-class AP@.5.

    Classes absent from the ground truth are excluded from the mean rather
    than counted as zero — reporting N/A, not 0, is the house rule (spec §5).
    """
    per_class_50: dict[str, float | None] = {}
    maps = []
    for thr in iou_thresholds:
        aps = []
        for c in classes:
            ap = average_precision(detections, ground_truth, c, thr)
            if thr == 0.5:
                per_class_50[c] = ap
            if ap is not None:
                aps.append(ap)
        maps.append(sum(aps) / len(aps) if aps else 0.0)
    ap50 = maps[iou_thresholds.index(0.5)] if 0.5 in iou_thresholds else None
    return {
        "map_50": ap50,
        "map_50_95": sum(maps) / len(maps) if maps else 0.0,
        "per_class_ap50": per_class_50,
        "n_images": len(ground_truth),
        "n_gt_boxes": sum(len(v) for v in ground_truth.values()),
        "n_detections": sum(len(v) for v in detections.values()),
        "classes_scored": sum(1 for v in per_class_50.values() if v is not None),
    }
