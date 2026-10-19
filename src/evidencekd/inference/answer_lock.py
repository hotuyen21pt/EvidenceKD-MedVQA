"""Answer Lock and the Evidence Pointer (method spec §4.3, §2).

Spec §2: *"Đáp án, finding và điểm do nhánh quyết định sinh; generator không
được sửa."* The answer, findings and scores come from the decision branch;
the generator may not change them.

This module makes that structural rather than advisory. The generator never
receives a writable answer — it receives a :class:`LockedAnswer`, which is
frozen, and the final JSON is assembled from the locked value regardless of
what the generator produced. If generated text disagrees with the locked
answer, the **answer wins and the disagreement is recorded as a warning**;
the text is never allowed to overwrite it.

Abstention, if wanted, is a separate decision branch with its own threshold
(master prompt §9.2) — never the generator quietly concluding something else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..data.answer_norm import NormalizedAnswer, normalize_answer

#: Returned by the pointer when no region supports the answer.
NO_SUPPORT: str = "NO_SUPPORT"


class AnswerLockViolation(RuntimeError):
    """Raised when something tries to mutate a locked answer."""


@dataclass
class LockedAnswer:
    """The decision branch's output, frozen before generation.

    Not ``@dataclass(frozen=True)``: that raises ``FrozenInstanceError``, a
    bare ``AttributeError`` subclass that says nothing about *why* the write
    was refused. A custom ``__setattr__`` lets the error name the rule it is
    enforcing, which is the difference between a confusing traceback and an
    obvious one when a generator tries to write back.

    >>> locked = LockedAnswer(answer="Yes", question_type="Is_there")
    >>> locked.answer = "No"
    Traceback (most recent call last):
        ...
    evidencekd.inference.answer_lock.AnswerLockViolation: ...
    """

    answer: str
    question_type: str
    findings: tuple[str, ...] = ()
    scores: Mapping[str, float] = field(default_factory=dict)
    _initialised: bool = field(default=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_initialised", False):
            raise AnswerLockViolation(
                f"attempt to modify locked answer field {name!r}; the generator "
                "may not change the decision branch's output (spec §2)"
            )
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_initialised", True)


def lock_answer(
    *,
    answer: str,
    question_type: str,
    findings: Sequence[str] = (),
    scores: Mapping[str, float] | None = None,
) -> LockedAnswer:
    """Freeze the decision branch's answer before the generator runs."""
    return LockedAnswer(
        answer=answer,
        question_type=question_type,
        findings=tuple(findings),
        scores=dict(scores or {}),
    )


def enforce_answer_lock(
    locked: LockedAnswer, generated_text: str
) -> tuple[str, list[str]]:
    """Return the final answer and any warnings, never the generator's.

    The generated text is parsed with the *same* normalisation rules used
    everywhere else, and compared. A disagreement is a warning on the output,
    not a correction to the answer.

    >>> lk = lock_answer(answer="Yes", question_type="Is_there")
    >>> enforce_answer_lock(lk, "No, there is no cardiomegaly. [R01]")[0]
    'Yes'
    """
    warnings: list[str] = []
    locked_norm = normalize_answer(locked.answer, question_type=locked.question_type)
    generated_norm = normalize_answer(
        generated_text, question_type=locked.question_type
    )

    if (
        locked_norm.yes_no is not None
        and generated_norm.yes_no is not None
        and locked_norm.yes_no != generated_norm.yes_no
    ):
        warnings.append(
            "generator_contradicted_answer: generated text asserts "
            f"{generated_norm.yes_no}, locked answer is {locked_norm.yes_no}; "
            "locked answer kept"
        )

    if (
        locked_norm.side is not None
        and generated_norm.side is not None
        and locked_norm.side != generated_norm.side
    ):
        warnings.append(
            f"generator_contradicted_side: generated {generated_norm.side}, "
            f"locked {locked_norm.side}; locked answer kept"
        )

    return locked.answer, warnings


@dataclass(frozen=True)
class PointerResult:
    """Evidence Pointer output: real region ids, or ``NO_SUPPORT``."""

    region_ids: tuple[str, ...]
    null_probability: float
    supported: bool

    @property
    def status(self) -> str:
        return "SUPPORTED" if self.supported else NO_SUPPORT


def select_evidence(
    *,
    region_ids: Sequence[str],
    probabilities: Sequence[float],
    null_probability: float,
    top_k: int = 3,
    min_probability: float = 0.0,
) -> PointerResult:
    """Pick supporting regions, or decline to.

    ``probabilities`` and ``null_probability`` together are the pointer's
    distribution :math:`\\pi^S_c` from spec §4.3, the null entry last.

    Returns ``NO_SUPPORT`` when the null slot outranks every region. That is
    the honest outcome for a question nothing in the image answers, and master
    prompt §9.1 forbids forcing a region instead.

    Never invents a region id: every id returned came from ``region_ids``.

    >>> r = select_evidence(region_ids=["R01"], probabilities=[0.2], null_probability=0.8)
    >>> r.status
    'NO_SUPPORT'
    >>> r = select_evidence(region_ids=["R01"], probabilities=[0.9], null_probability=0.1)
    >>> r.region_ids
    ('R01',)
    """
    if len(region_ids) != len(probabilities):
        raise ValueError(
            f"{len(region_ids)} region ids but {len(probabilities)} probabilities"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    if not region_ids:
        return PointerResult((), null_probability, supported=False)

    ranked = sorted(zip(region_ids, probabilities), key=lambda p: -p[1])
    best_prob = ranked[0][1]

    if null_probability >= best_prob:
        return PointerResult((), null_probability, supported=False)

    chosen = tuple(
        rid
        for rid, p in ranked[:top_k]
        if p >= min_probability and p > null_probability
    )
    if not chosen:
        return PointerResult((), null_probability, supported=False)
    return PointerResult(chosen, null_probability, supported=True)
