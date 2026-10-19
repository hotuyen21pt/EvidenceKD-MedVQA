"""Bounding-box geometry and multi-radiologist box fusion.

Pure Python by design: no numpy, no torch. This module encodes decision **D4**
of ``docs/implementation_audit.md`` and must stay testable without a GPU
toolchain.

Why fusion is needed at all
---------------------------
Every VinDr-CXR *training* image was read by **three radiologists
independently, without consensus**. There is therefore no single ground-truth
box. Any grounding metric, and the KD-R region set, is undefined until a
fusion rule is fixed.

We fuse with a cluster-then-average rule (the Weighted Box Fusion family):
boxes of the same class that overlap above ``iou_threshold`` form a cluster,
and the cluster collapses to one box whose coordinates are the mean of its
members. Crucially we also keep ``n_raters``, the number of *distinct*
radiologists in the cluster, because agreement is the signal that separates a
``SUPPORTED`` finding from an ``INSUFFICIENT`` one (audit §3.5). Averaging it
away would discard the most interesting property of this dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in absolute pixel coordinates.

    Coordinates follow the VinDr CSV convention: ``x_min, y_min`` inclusive
    top-left, ``x_max, y_max`` bottom-right.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_name: str
    rad_id: str | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    def is_degenerate(self) -> bool:
        """True when the box has no area, so it cannot be evidence."""
        return self.area <= 0.0

    def clip(self, width: float, height: float) -> "Box":
        """Clamp to an image of ``width`` x ``height``.

        VinDr boxes can fall marginally outside the image; the audit requires
        these be counted, not silently accepted. Clipping is the repair, and
        :func:`evidencekd.data.validate` counts how often it fires.
        """
        return Box(
            x_min=min(max(self.x_min, 0.0), width),
            y_min=min(max(self.y_min, 0.0), height),
            x_max=min(max(self.x_max, 0.0), width),
            y_max=min(max(self.y_max, 0.0), height),
            class_name=self.class_name,
            rad_id=self.rad_id,
        )

    def scale(self, sx: float, sy: float) -> "Box":
        """Rescale, e.g. when the image is resized for the model."""
        return Box(
            x_min=self.x_min * sx,
            y_min=self.y_min * sy,
            x_max=self.x_max * sx,
            y_max=self.y_max * sy,
            class_name=self.class_name,
            rad_id=self.rad_id,
        )


@dataclass(frozen=True)
class FusedBox:
    """One consensus region, plus the agreement that produced it."""

    box: Box
    n_raters: int
    """Number of *distinct* radiologists who drew a box in this cluster."""
    member_rad_ids: tuple[str, ...] = field(default=())

    @property
    def region_id(self) -> str:
        """Stable identifier, as required by the proposal (§6.2).

        Derived from the class and the rounded geometry so the same region
        gets the same id across runs, caches and KD-R shards.
        """
        b = self.box
        return (
            f"{b.class_name}|{int(round(b.x_min))},{int(round(b.y_min))},"
            f"{int(round(b.x_max))},{int(round(b.y_max))}"
        )


def iou(a: Box, b: Box) -> float:
    """Intersection over union. Returns 0.0 if either box is degenerate."""
    if a.is_degenerate() or b.is_degenerate():
        return 0.0
    ix0 = max(a.x_min, b.x_min)
    iy0 = max(a.y_min, b.y_min)
    ix1 = min(a.x_max, b.x_max)
    iy1 = min(a.y_max, b.y_max)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def fuse_boxes(
    boxes: Sequence[Box],
    *,
    iou_threshold: float = 0.4,
    min_raters: int = 1,
) -> list[FusedBox]:
    """Fuse same-class overlapping boxes into consensus regions.

    Args:
        boxes: all boxes for **one image**, from any number of radiologists.
        iou_threshold: two boxes of the same class join the same cluster when
            their IoU meets this. 0.4 is the default; it is a *config value*,
            not a constant of nature, and the ablation in the experiment plan
            varies it.
        min_raters: drop fused regions seen by fewer than this many distinct
            radiologists. Default 1 keeps everything; the agreement count
            stays on the result so filtering can also happen downstream.

    Returns:
        Fused regions, ordered by descending agreement then descending area,
        so that ``region_id`` ordering is deterministic.

    Degenerate boxes are discarded before clustering — a zero-area box cannot
    be evidence for anything.

    Greedy single-linkage clustering: each box joins the first existing
    cluster whose *representative* (running mean) it overlaps enough, else it
    starts a new cluster. This is order-dependent in principle; we sort by
    area descending first so the largest box seeds each cluster, which makes
    the result deterministic for a given input set.
    """
    if iou_threshold <= 0.0 or iou_threshold > 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    usable = [b for b in boxes if not b.is_degenerate()]
    by_class: dict[str, list[Box]] = {}
    for b in usable:
        by_class.setdefault(b.class_name, []).append(b)

    fused: list[FusedBox] = []
    for class_name, group in by_class.items():
        group = sorted(group, key=lambda b: (-b.area, b.x_min, b.y_min))
        clusters: list[list[Box]] = []
        for box in group:
            placed = False
            for cluster in clusters:
                if iou(_mean_box(cluster, class_name), box) >= iou_threshold:
                    cluster.append(box)
                    placed = True
                    break
            if not placed:
                clusters.append([box])

        for cluster in clusters:
            rad_ids = tuple(
                sorted({b.rad_id for b in cluster if b.rad_id is not None})
            )
            n_raters = len(rad_ids) if rad_ids else len(cluster)
            if n_raters < min_raters:
                continue
            fused.append(
                FusedBox(
                    box=_mean_box(cluster, class_name),
                    n_raters=n_raters,
                    member_rad_ids=rad_ids,
                )
            )

    fused.sort(key=lambda f: (-f.n_raters, -f.box.area, f.box.class_name))
    return fused


def _mean_box(cluster: Sequence[Box], class_name: str) -> Box:
    """Coordinate-wise mean of a cluster. Unweighted, by choice.

    Weighting by confidence is the usual WBF refinement, but radiologist
    annotations carry no confidence score, so there is nothing honest to
    weight by.
    """
    n = len(cluster)
    return Box(
        x_min=sum(b.x_min for b in cluster) / n,
        y_min=sum(b.y_min for b in cluster) / n,
        x_max=sum(b.x_max for b in cluster) / n,
        y_max=sum(b.y_max for b in cluster) / n,
        class_name=class_name,
        rad_id=None,
    )


def agreement_level(fused: FusedBox, n_annotators: int = 3) -> str:
    """Map rater agreement onto the proposal's three evidence states.

    This is the bridge from VinDr's inter-rater structure to the
    ``SUPPORTED / INSUFFICIENT`` distinction the proposal is built on
    (audit §3.5). A finding one radiologist of three saw is genuinely
    *insufficient* evidence, not a labelling error.

    ``CONTRADICTED`` is deliberately **not** produced here: radiologists who
    did not draw a box did not assert absence, they simply did not annotate.
    Treating silence as contradiction is exactly the inference the proposal
    forbids (§1.1).
    """
    if n_annotators <= 0:
        raise ValueError("n_annotators must be positive")
    if fused.n_raters >= max(2, (n_annotators // 2) + 1):
        return "SUPPORTED"
    return "INSUFFICIENT"


def recall_at_k(
    predicted: Sequence[Box],
    ground_truth: Sequence[Box],
    *,
    k: int,
    iou_threshold: float = 0.5,
) -> float:
    """Fraction of ground-truth boxes hit by the top-``k`` predictions.

    Returns ``float('nan')`` when there is no ground truth, so that empty
    cases are excluded from an average rather than counted as a perfect or a
    zero score.
    """
    if not ground_truth:
        return float("nan")
    top = list(predicted)[:k]
    hits = 0
    for gt in ground_truth:
        if any(
            p.class_name == gt.class_name and iou(p, gt) >= iou_threshold
            for p in top
        ):
            hits += 1
    return hits / len(ground_truth)
