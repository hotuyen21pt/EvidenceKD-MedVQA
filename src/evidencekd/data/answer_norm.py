"""Rule-based normalisation of answer text into the four head labels.

Method spec §2 requires the free-text ``answer`` field to be reduced by rule
into **Yes/No, finding, side and count** labels, which are what the four
answer heads (§4.1) predict and what Table A scores.

**This module is the single source of truth for answer normalisation.**
Spec §5 requires every baseline's text output to be parsed by *the same*
rules, and master prompt §10.1 forbids comparing two models normalised
differently. Nothing else in the codebase may re-implement any of this.

A note on laterality
--------------------
Here we read the side out of **text** ("the right lung"), which already means
the *patient's* right. Deriving side from a **box** is a different operation
with an inversion in it — on a frontal film the patient's right appears on the
viewer's left — and lives in :mod:`evidencekd.facts.geometry`. Keeping the two
apart is deliberate; conflating them silently mirrors every side label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from .vqa_schema import strip_loc_tokens

#: Side labels. ``None`` means the text asserts no side.
SIDE_LEFT: Final[str] = "left"
SIDE_RIGHT: Final[str] = "right"
SIDE_BILATERAL: Final[str] = "bilateral"
SIDES: Final[tuple[str, ...]] = (SIDE_LEFT, SIDE_RIGHT, SIDE_BILATERAL)

_NUMBER_WORDS: Final[dict[str, int]] = {
    "no": 0,
    "none": 0,
    "zero": 0,
    "one": 1,
    "a": 1,
    "an": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

# Ordered longest-first so "no evidence of" wins over a bare "no".
_NEGATIVE_CUES: Final[tuple[str, ...]] = (
    "there is no evidence",
    "there is no",
    "no evidence of",
    "no significant",
    "not identified",
    "not present",
    "not seen",
    "is absent",
    "are absent",
    "without evidence",
    "negative for",
    "unremarkable",
    "normal",
)

_POSITIVE_CUES: Final[tuple[str, ...]] = (
    "there is",
    "there are",
    "is present",
    "are present",
    "is seen",
    "are seen",
    "demonstrates",
    "shows",
    "consistent with",
    "compatible with",
    "evidence of",
)

_BILATERAL_CUES: Final[tuple[str, ...]] = (
    "bilateral",
    "bilaterally",
    "both lungs",
    "both sides",
    "both hemithorac",
    "left and right",
    "right and left",
)


@dataclass(frozen=True)
class NormalizedAnswer:
    """The rule-derived label set for one answer string.

    Every field is optional because not every question asserts every
    attribute. ``None`` means *the text does not say*, which is distinct from
    a negative assertion — Table A scores each head only on the subset of
    questions where its label is defined (spec §6: "đếm trên subset nhãn hợp
    lệ").
    """

    yes_no: bool | None = None
    finding: str | None = None
    side: str | None = None
    count: int | None = None
    raw: str = ""

    def defined_heads(self) -> tuple[str, ...]:
        """Which heads this answer actually supervises."""
        names = []
        if self.yes_no is not None:
            names.append("yes_no")
        if self.finding is not None:
            names.append("finding")
        if self.side is not None:
            names.append("side")
        if self.count is not None:
            names.append("count")
        return tuple(names)


def _clean(text: str) -> str:
    """Lower-case, drop location tokens, collapse whitespace."""
    return " ".join(strip_loc_tokens(text).lower().split())


def parse_yes_no(text: str) -> bool | None:
    """Decide polarity, or ``None`` when the text does not assert one.

    Leading "yes"/"no" wins outright; otherwise the earliest-matching cue
    decides, so "There is no pneumothorax" reads negative even though it
    contains the positive cue "there is".

    >>> parse_yes_no("Yes, there is cardiomegaly.")
    True
    >>> parse_yes_no("No evidence of pneumothorax.")
    False
    >>> parse_yes_no("The lesion is in the right upper zone.")

    """
    t = _clean(text)
    if not t:
        return None
    if re.match(r"^yes\b", t):
        return True
    if re.match(r"^no\b", t):
        return False

    first_neg = min(
        (t.find(c) for c in _NEGATIVE_CUES if c in t), default=-1
    )
    first_pos = min(
        (t.find(c) for c in _POSITIVE_CUES if c in t), default=-1
    )
    if first_neg == -1 and first_pos == -1:
        return None
    if first_neg == -1:
        return True
    if first_pos == -1:
        return False
    return first_neg > first_pos


def parse_side(text: str) -> str | None:
    """Extract laterality from answer text.

    Bilateral is checked first: "bilateral pleural effusion" contains neither
    "left" nor "right" as a word, but "left and right" contains both and must
    not be read as one side.

    >>> parse_side("Bilateral pleural effusions are present.")
    'bilateral'
    >>> parse_side("Opacity in the right lower zone.")
    'right'
    >>> parse_side("Cardiomegaly is present.")

    """
    t = _clean(text)
    if any(cue in t for cue in _BILATERAL_CUES):
        return SIDE_BILATERAL
    has_left = re.search(r"\bleft\b", t) is not None
    has_right = re.search(r"\bright\b", t) is not None
    if has_left and has_right:
        return SIDE_BILATERAL
    if has_left:
        return SIDE_LEFT
    if has_right:
        return SIDE_RIGHT
    return None


def parse_count(text: str, *, question_type: str | None = None) -> int | None:
    """Extract a cardinal count.

    Only trusted for ``How_many`` questions unless a digit is present:
    prose like "three times larger" would otherwise be read as a count. A
    leading "no"/"none" on a How_many question means zero.

    >>> parse_count("There are two nodules.", question_type="How_many")
    2
    >>> parse_count("3 opacities are seen.")
    3
    >>> parse_count("Yes, there is cardiomegaly.", question_type="Is_there")

    """
    t = _clean(text)
    if not t:
        return None

    digits = re.search(r"\b(\d+)\b", t)
    if digits is not None:
        return int(digits.group(1))

    if question_type != "How_many":
        return None

    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t):
            return value
    return None


def normalize_answer(
    text: str,
    *,
    question_type: str | None = None,
    gt_finding: str | None = None,
) -> NormalizedAnswer:
    """Reduce one answer string to its head labels.

    Args:
        text: the raw ``answer`` field, or a baseline model's generated text.
        question_type: one of :data:`evidencekd.data.vqa_schema.QUESTION_TYPES`.
            Gates the count rule and selects which heads are defined.
        gt_finding: the dataset's finding label. Supplied during *training
            data construction* only. It is **never** available at inference —
            spec §2 lists ``gt_finding`` as "Không dùng" at inference — so a
            caller normalising a model's prediction must leave it ``None``.

    The ``finding`` field is taken from ``gt_finding`` rather than mined out
    of the prose: the questions already name the finding (spec §9 lists this
    as a known limitation of the dataset), so parsing it back out of the
    answer would measure string matching, not recognition.
    """
    yes_no = parse_yes_no(text)
    side = parse_side(text)
    count = parse_count(text, question_type=question_type)

    # A Yes/No-style question always defines polarity; default an
    # unparseable one to False rather than None only when the question type
    # guarantees a binary answer.
    if yes_no is None and question_type in ("Is_there", "Yes_No"):
        yes_no = False

    # Zero count and a negative answer must agree.
    if count == 0 and yes_no is None:
        yes_no = False

    return NormalizedAnswer(
        yes_no=yes_no,
        finding=gt_finding or None,
        side=side,
        count=count,
        raw=text,
    )
