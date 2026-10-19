"""The geometric fact table (method spec §4.2).

Facts are **computed from boxes and anatomy masks by rule**, never produced by
a language model. That is the point of the component: a fact carries an
``evidence_ids`` list naming the regions it was derived from, so a claim that
cites a fact is traceable to geometry rather than to generation.

Spec §4.2 table, implemented here:

===================  ====================================================
Attribute            Rule
===================  ====================================================
presence             a box of class *c* above threshold
side                 box centre against the thorax midline
lung zone            which vertical third of the lung contains the centre
uni/bilateral        distribution of same-class boxes across sides
count                number of same-class boxes after NMS
extent               box∩lung area / lung area, in three bands
CTR                  heart width / thorax width — **PA films only**
co-occurrence        a pair of findings both present
absence              no box above threshold *and* the whole-image head low
===================  ====================================================

Laterality warning
------------------
On a frontal chest radiograph the patient's **right** lung is displayed on the
**viewer's left**. :func:`side_of` performs that inversion. Reading a side out
of *text* instead is a different operation and lives in
:mod:`evidencekd.data.answer_norm`. Do not merge them.

Pure Python: no numpy, no torch. Anatomy is supplied as boxes plus optional
pre-computed mask areas, so a segmentation model is never imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

from ..data.boxes import Box, iou

Side = Literal["left", "right", "bilateral"]
Zone = Literal["upper", "middle", "lower"]
Extent = Literal["focal", "moderate", "extensive"]

#: Anatomy region ids used by the fact table, matching the spec's output JSON.
A_THORAX: str = "A_thorax"
A_HEART: str = "A_heart"
A_LUNG_LEFT: str = "A_lung_left"
A_LUNG_RIGHT: str = "A_lung_right"

#: Extent band edges as a fraction of the ipsilateral lung area.
EXTENT_BANDS: tuple[float, float] = (0.10, 0.33)

#: CTR above this is conventionally called cardiomegaly on a PA film.
CTR_THRESHOLD: float = 0.50


@dataclass(frozen=True)
class Anatomy:
    """Frozen anatomy regions for one image.

    ``*_area`` fields carry true mask areas when a segmentation model provided
    them. When absent we fall back to box area, which overestimates a lung;
    :attr:`areas_are_masks` records which happened so a result can say so.
    """

    thorax: Box | None = None
    heart: Box | None = None
    lung_left: Box | None = None
    lung_right: Box | None = None
    lung_left_area: float | None = None
    lung_right_area: float | None = None

    @property
    def areas_are_masks(self) -> bool:
        return self.lung_left_area is not None or self.lung_right_area is not None

    def lung(self, side: str) -> Box | None:
        return self.lung_left if side == "left" else self.lung_right

    def lung_area(self, side: str) -> float | None:
        explicit = self.lung_left_area if side == "left" else self.lung_right_area
        if explicit is not None:
            return explicit
        box = self.lung(side)
        return box.area if box is not None else None


@dataclass(frozen=True)
class Fact:
    """One derived fact, traceable to the regions that produced it."""

    id: str
    attr: str
    value: object
    evidence_ids: tuple[str, ...]
    note: str | None = None
    """Set when the fact is qualified — e.g. CTR computed without a mask."""


def _centre(box: Box) -> tuple[float, float]:
    return ((box.x_min + box.x_max) / 2.0, (box.y_min + box.y_max) / 2.0)


def side_of(box: Box, anatomy: Anatomy, *, image_width: float | None = None) -> Side | None:
    """Patient-side laterality of a box.

    Uses the thorax midline when anatomy is available, else the image midline.
    Returns ``None`` when neither is known — guessing a side is worse than
    declining to assert one.

    The inversion lives here: image x below the midline is the patient's
    **right**.

    >>> a = Anatomy(thorax=Box(0, 0, 100, 100, "thorax"))
    >>> side_of(Box(10, 10, 20, 20, "Nodule/Mass"), a)
    'right'
    >>> side_of(Box(80, 10, 90, 20, "Nodule/Mass"), a)
    'left'
    """
    if anatomy.thorax is not None:
        midline = (anatomy.thorax.x_min + anatomy.thorax.x_max) / 2.0
    elif image_width is not None and image_width > 0:
        midline = image_width / 2.0
    else:
        return None
    cx, _ = _centre(box)
    return "right" if cx < midline else "left"


def zone_of(box: Box, anatomy: Anatomy, *, image_height: float | None = None) -> Zone | None:
    """Which vertical third of the ipsilateral lung holds the box centre.

    Measured against the **lung**, not the image: a film with a lot of
    abdomen below the diaphragm would otherwise push everything into
    "upper".
    """
    side = side_of(box, anatomy, image_width=None)
    lung = anatomy.lung(side) if side else None
    if lung is not None and lung.height > 0:
        top, height = lung.y_min, lung.height
    elif image_height is not None and image_height > 0:
        top, height = 0.0, image_height
    else:
        return None
    _, cy = _centre(box)
    idx = int(((cy - top) / height) * 3)
    return ("upper", "middle", "lower")[min(max(idx, 0), 2)]


def nms(boxes: Sequence[Box], *, iou_threshold: float = 0.5) -> list[Box]:
    """Greedy non-maximum suppression within a class, by descending area.

    Boxes carry no score here (they are detections already filtered, or
    ground-truth), so area breaks ties deterministically.
    """
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: (-b.area, b.x_min, b.y_min)):
        if all(iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def count_of(
    boxes: Sequence[Box], class_name: str, *, iou_threshold: float = 0.5
) -> int:
    """Number of distinct instances of ``class_name`` after NMS."""
    same = [b for b in boxes if b.class_name == class_name and not b.is_degenerate()]
    return len(nms(same, iou_threshold=iou_threshold))


def laterality_of(
    boxes: Sequence[Box], class_name: str, anatomy: Anatomy
) -> Side | None:
    """Whether a finding is one-sided or bilateral."""
    sides = {
        s
        for b in boxes
        if b.class_name == class_name
        for s in [side_of(b, anatomy)]
        if s is not None
    }
    if not sides:
        return None
    if len(sides) > 1:
        return "bilateral"
    return next(iter(sides))  # type: ignore[return-value]


def extent_of(box: Box, anatomy: Anatomy) -> tuple[Extent, float] | None:
    """Fraction of the ipsilateral lung the box covers, banded into three.

    Returns ``None`` when the lung area is unknown — an extent claim with no
    denominator would be fabricated.
    """
    side = side_of(box, anatomy)
    if side is None:
        return None
    lung = anatomy.lung(side)
    lung_area = anatomy.lung_area(side)
    if lung is None or not lung_area:
        return None

    ix0, iy0 = max(box.x_min, lung.x_min), max(box.y_min, lung.y_min)
    ix1, iy1 = min(box.x_max, lung.x_max), min(box.y_max, lung.y_max)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ratio = inter / lung_area if lung_area > 0 else 0.0

    lo, hi = EXTENT_BANDS
    band: Extent = "focal" if ratio < lo else ("moderate" if ratio < hi else "extensive")
    return band, ratio


def cardiothoracic_ratio(
    anatomy: Anatomy, *, view_position: str | None
) -> tuple[float, str | None] | None:
    """CTR = heart width / thorax width.

    Returns ``None`` unless the film is **PA**. Spec §4.2 restricts CTR to PA
    views, and §9 lists view dependence as a limitation: an AP film magnifies
    the heart, so an AP CTR would systematically over-call cardiomegaly.
    Returning ``None`` for an unknown view is deliberate — an unlabelled film
    is not assumed to be PA.

    >>> a = Anatomy(thorax=Box(0, 0, 100, 100, "t"), heart=Box(25, 40, 75, 80, "h"))
    >>> ctr, _ = cardiothoracic_ratio(a, view_position="PA")
    >>> round(ctr, 3)
    0.5
    >>> cardiothoracic_ratio(a, view_position="AP") is None
    True
    """
    if view_position is None or view_position.strip().upper() != "PA":
        return None
    if anatomy.heart is None or anatomy.thorax is None:
        return None
    if anatomy.thorax.width <= 0:
        return None
    ctr = anatomy.heart.width / anatomy.thorax.width
    note = None if anatomy.areas_are_masks else "widths from boxes, not masks"
    return ctr, note


@dataclass
class FactTable:
    """The facts for one image, with stable ids and region references."""

    facts: list[Fact] = field(default_factory=list)
    region_ids: dict[str, Box] = field(default_factory=dict)

    def add(
        self, attr: str, value: object, evidence_ids: Iterable[str], note: str | None = None
    ) -> Fact:
        fact = Fact(
            id=f"F{len(self.facts) + 1}",
            attr=attr,
            value=value,
            evidence_ids=tuple(evidence_ids),
            note=note,
        )
        self.facts.append(fact)
        return fact

    def by_id(self, fact_id: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fact_id), None)

    def as_dicts(self) -> list[dict[str, object]]:
        out = []
        for f in self.facts:
            d: dict[str, object] = {
                "id": f.id,
                "attr": f.attr,
                "value": f.value,
                "evidence_ids": list(f.evidence_ids),
            }
            if f.note:
                d["note"] = f.note
            out.append(d)
        return out


def build_fact_table(
    *,
    regions: Mapping[str, Box],
    anatomy: Anatomy,
    present_classes: Sequence[str],
    whole_image_scores: Mapping[str, float] | None = None,
    absence_threshold: float = 0.5,
    view_position: str | None = None,
) -> FactTable:
    """Derive every applicable fact for one image.

    Args:
        regions: ``region_id -> Box`` for detected/annotated finding regions.
        anatomy: frozen anatomy regions.
        present_classes: classes judged present by the decision branch.
        whole_image_scores: per-class whole-image head scores. Required for
            **absence** facts: spec §4.2 defines absence as *no box above
            threshold* **and** *whole-image head below threshold*. Without the
            head score we cannot assert absence, and we do not.
        view_position: ``"PA"``, ``"AP"`` or ``None``. Gates CTR.

    Absence is the conservative case throughout: a class with no box but no
    whole-image score produces **no fact at all**, not an absence fact.
    """
    table = FactTable(region_ids=dict(regions))
    by_class: dict[str, list[tuple[str, Box]]] = {}
    for rid, box in regions.items():
        by_class.setdefault(box.class_name, []).append((rid, box))

    for class_name in sorted(set(present_classes)):
        entries = by_class.get(class_name, [])
        if not entries:
            continue
        ids = [rid for rid, _ in entries]
        boxes = [b for _, b in entries]

        table.add("presence", class_name, ids)

        side = laterality_of(boxes, class_name, anatomy)
        if side is not None:
            anat = [A_THORAX] if anatomy.thorax is not None else []
            table.add("side", side, ids + anat)

        primary_id, primary = max(entries, key=lambda e: e[1].area)
        zone = zone_of(primary, anatomy)
        if zone is not None:
            lung_side = side_of(primary, anatomy)
            anat = []
            if lung_side and anatomy.lung(lung_side) is not None:
                anat = [A_LUNG_LEFT if lung_side == "left" else A_LUNG_RIGHT]
            table.add("zone", zone, [primary_id] + anat)

        n = count_of(boxes, class_name)
        if n > 0:
            table.add("count", n, ids)

        ext = extent_of(primary, anatomy)
        if ext is not None:
            band, ratio = ext
            lung_side = side_of(primary, anatomy)
            anat = [A_LUNG_LEFT if lung_side == "left" else A_LUNG_RIGHT]
            table.add(
                "extent",
                band,
                [primary_id] + anat,
                note=None if anatomy.areas_are_masks else f"ratio={ratio:.3f} from boxes",
            )

    # Co-occurrence: unordered pairs of present classes that both have boxes.
    grounded = sorted(c for c in set(present_classes) if by_class.get(c))
    for i, a in enumerate(grounded):
        for b in grounded[i + 1 :]:
            ids = [r for r, _ in by_class[a]] + [r for r, _ in by_class[b]]
            table.add("co_occurrence", (a, b), ids)

    # CTR, PA films only.
    ctr = cardiothoracic_ratio(anatomy, view_position=view_position)
    if ctr is not None:
        value, note = ctr
        table.add("ctr", round(value, 4), [A_HEART, A_THORAX], note=note)

    # Absence: requires BOTH no box and a low whole-image score.
    if whole_image_scores:
        for class_name, score in sorted(whole_image_scores.items()):
            if by_class.get(class_name):
                continue
            if score >= absence_threshold:
                continue
            anat = [A_THORAX] if anatomy.thorax is not None else []
            table.add("absence", class_name, anat)

    return table
