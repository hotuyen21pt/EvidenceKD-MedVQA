"""VinDr-CXR-VQA record parsing.

Source dataset: https://huggingface.co/datasets/Dangindev/VinDR-CXR-VQA
Paper: https://arxiv.org/abs/2511.00504

Verified structure (2026-10-19): 4,394 images / 17,597 QA pairs, CC BY 4.0.
One row per **image**::

    {"image_id": str, "num_questions": int, "vqa": [ {...}, ... ]}

and each ``vqa`` entry carries ``question, answer, reason, type, difficulty,
gt_finding, gt_location``.

Two properties of this dataset drive the design here:

1. ``gt_location`` and the ``answer`` text both embed boxes as
   ``<loc_xmin_ymin_xmax_ymax>`` tokens, so box extraction is string parsing,
   not a numeric column read.
2. The HF release exposes **one split only** (``train``). Validation and test
   splits are ours to make, and the method spec §2 requires splitting *by
   image* so that no image's questions straddle two splits.

Pure Python: no torch, no numpy, so this stays testable without a GPU
toolchain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Iterator, Sequence

#: ``<loc_691_1375_1653_1831>`` -> four integers.
LOC_RE: Final[re.Pattern[str]] = re.compile(
    r"<loc_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)>"
)

#: The six question types in the released taxonomy.
QUESTION_TYPES: Final[tuple[str, ...]] = (
    "Where",
    "What",
    "Is_there",
    "How_many",
    "Which",
    "Yes_No",
)


class SchemaError(ValueError):
    """Raised when a record does not match the documented schema.

    Deliberately loud. Master prompt §15.1 forbids swallowing an exception and
    continuing as though it succeeded; a malformed record is counted and
    reported by the validator, never silently dropped in the loader.
    """


@dataclass(frozen=True)
class LocBox:
    """A box parsed from a ``<loc_...>`` token, in original pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def area(self) -> int:
        return max(0, self.x_max - self.x_min) * max(0, self.y_max - self.y_min)

    def is_valid(self) -> bool:
        """A box is usable only if it has positive area and no negatives."""
        return (
            self.x_min >= 0
            and self.y_min >= 0
            and self.x_max > self.x_min
            and self.y_max > self.y_min
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def to_token(self) -> str:
        return f"<loc_{self.x_min}_{self.y_min}_{self.x_max}_{self.y_max}>"


def parse_loc_tokens(text: str) -> list[LocBox]:
    """Extract every ``<loc_...>`` box from a string, in order of appearance.

    Returns an empty list when there are none — many answers (``Is_there`` /
    ``Yes_No`` negatives) legitimately carry no location.

    >>> parse_loc_tokens("centrally located <loc_691_1375_1653_1831>.")
    [LocBox(x_min=691, y_min=1375, x_max=1653, y_max=1831)]
    >>> parse_loc_tokens("No evidence of pneumothorax.")
    []
    """
    return [
        LocBox(int(a), int(b), int(c), int(d))
        for a, b, c, d in LOC_RE.findall(text)
    ]


def strip_loc_tokens(text: str) -> str:
    """Remove ``<loc_...>`` tokens and tidy the whitespace they leave behind.

    Used when the answer text must be compared or shown as prose rather than
    parsed for geometry.

    >>> strip_loc_tokens("enlargement of the silhouette <loc_1_2_3_4>.")
    'enlargement of the silhouette.'
    """
    cleaned = LOC_RE.sub("", text)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class VQAItem:
    """One question/answer pair about one image."""

    image_id: str
    question: str
    answer: str
    reason: str
    type: str
    difficulty: str
    gt_finding: str
    gt_location: str
    index: int
    """Position within the image's ``vqa`` list; part of the sample id."""

    @property
    def sample_id(self) -> str:
        """Stable id. Keyed on image so splits-by-image stay coherent."""
        return f"{self.image_id}#{self.index}"

    @property
    def gt_boxes(self) -> list[LocBox]:
        """Boxes from ``gt_location``. May be empty for negative cases."""
        return parse_loc_tokens(self.gt_location)

    @property
    def answer_boxes(self) -> list[LocBox]:
        """Boxes embedded in the answer prose."""
        return parse_loc_tokens(self.answer)

    @property
    def answer_text(self) -> str:
        """The answer with location tokens removed."""
        return strip_loc_tokens(self.answer)

    @property
    def is_grounded(self) -> bool:
        """True when at least one valid ground-truth box exists.

        A negative case ("No, there is no pneumothorax") is **not** grounded,
        and must not be: nothing in the image shows an absence. The method
        spec §4.1 makes the same point — negative answers come from the
        Yes/No and whole-image finding heads, never from a missing box.
        """
        return any(b.is_valid() for b in self.gt_boxes)


def parse_record(record: dict[str, Any], *, strict: bool = True) -> list[VQAItem]:
    """Flatten one dataset row (an image) into its QA items.

    Args:
        record: a raw row with ``image_id``, ``num_questions``, ``vqa``.
        strict: when True, raise :class:`SchemaError` on anything unexpected.
            When False, skip malformed items — only for the validator, which
            counts what it skipped. Training must use ``strict=True``.

    Raises:
        SchemaError: on a missing field or a ``num_questions`` that disagrees
            with ``len(vqa)``. The mismatch check is cheap and catches a
            truncated download, which otherwise shows up much later as a
            quietly undersized training set.
    """
    image_id = record.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise SchemaError(f"missing or invalid image_id: {image_id!r}")

    vqa = record.get("vqa")
    if not isinstance(vqa, Sequence) or isinstance(vqa, (str, bytes)):
        raise SchemaError(f"{image_id}: 'vqa' must be a list, got {type(vqa).__name__}")

    declared = record.get("num_questions")
    if strict and declared is not None and declared != len(vqa):
        raise SchemaError(
            f"{image_id}: num_questions={declared} but len(vqa)={len(vqa)}"
        )

    required = (
        "question",
        "answer",
        "reason",
        "type",
        "difficulty",
        "gt_finding",
        "gt_location",
    )
    items: list[VQAItem] = []
    for i, raw in enumerate(vqa):
        if not isinstance(raw, dict):
            if strict:
                raise SchemaError(f"{image_id}#{i}: vqa entry is not a dict")
            continue
        missing = [k for k in required if k not in raw]
        if missing:
            if strict:
                raise SchemaError(f"{image_id}#{i}: missing fields {missing}")
            continue
        items.append(
            VQAItem(
                image_id=image_id,
                question=str(raw["question"]),
                answer=str(raw["answer"]),
                reason=str(raw["reason"]),
                type=str(raw["type"]),
                difficulty=str(raw["difficulty"]),
                gt_finding=str(raw["gt_finding"]),
                gt_location=str(raw["gt_location"]),
                index=i,
            )
        )
    return items


def iter_items(
    records: Sequence[dict[str, Any]], *, strict: bool = True
) -> Iterator[VQAItem]:
    """Flatten many rows into a stream of QA items."""
    for record in records:
        yield from parse_record(record, strict=strict)
