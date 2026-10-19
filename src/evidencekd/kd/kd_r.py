"""KD-R — counterfactual region distillation (method spec §4.4).

The teacher sees only the image and the question. Each candidate region is
masked in turn and we measure how far the log-probability of the *correct*
answer falls:

.. math::
    r^T_i = \\max\\big(0,\\ \\log p_T(a^*\\mid I,Q) - \\log p_T(a^*\\mid I_{\\setminus i},Q)\\big)

.. math::
    \\pi^T = \\mathrm{softmax}\\big([r^T_1 \\ldots r^T_M,\\ r_\\varnothing]/\\tau_T\\big)

.. math::
    L_{KD\\text{-}R} = \\frac{\\sum_n w_n\\,\\mathrm{KL}(\\pi^T_n \\Vert \\pi^S_{n,c_0})}
                            {\\sum_n w_n + \\epsilon},
    \\qquad w_n = \\mathbb 1[\\hat a^T_n = a^*_n]\\cdot\\mathbb 1[m_n \\ge \\delta]\\cdot c_n

Two things in that formulation matter and are easy to lose:

**The null slot.** :math:`r_\\varnothing` is an extra softmax entry meaning
"no region accounts for this answer". It is what makes the distribution
well-defined when every region is irrelevant — a plain
:math:`r_i/\\sum_j r_j` normalisation divides by zero there. It also gives the
student's pointer somewhere to put mass for ``NO_SUPPORT`` (§4.3). The spec
does not pin its numeric value, so it is a config parameter here and must be
reported with any result.

**Annotation is used only at the gate.** :math:`m_n` (the teacher's mass on
the annotated target region) and teacher correctness filter *which* samples
contribute. The distillation target itself is the teacher's full distribution,
including mass on regions the annotation never mentions — spec §4.4: *"Phần
khối lượng trên vùng phụ là tri thức được truyền."* Clamping the target to the
annotated box would discard exactly the signal KD-R exists to transfer.

Pure Python throughout, so the arithmetic is testable without torch. The
torch-side loss is :func:`kd_r_loss`, which imports torch lazily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: Guard for the weighted mean's denominator.
EPS: float = 1e-8


class KDRError(ValueError):
    """Raised when KD-R inputs are unusable (non-finite, empty, bad config)."""


def counterfactual_relevance(
    logp_original: float, logp_masked: float
) -> float:
    """Relevance of one region: the clipped drop in answer log-probability.

    A *negative* drop — masking the region made the teacher *more* confident —
    is clipped to zero rather than kept. Such a region is not evidence for the
    answer, and letting it contribute negative mass would make the softmax
    reward regions that argue against the answer.

    >>> counterfactual_relevance(-0.5, -2.0)
    1.5
    >>> counterfactual_relevance(-2.0, -0.5)
    0.0
    """
    if not (math.isfinite(logp_original) and math.isfinite(logp_masked)):
        raise KDRError(
            f"non-finite log-probabilities: orig={logp_original}, masked={logp_masked}"
        )
    return max(0.0, logp_original - logp_masked)


def teacher_distribution(
    relevances: Sequence[float],
    *,
    r_null: float = 0.0,
    tau: float = 1.0,
) -> list[float]:
    """Softmax over ``[r_1 .. r_M, r_null]`` at temperature ``tau``.

    Returns ``M + 1`` probabilities; the **last** entry is the null slot.

    Unlike a sum-normalisation this is total: all-zero relevances give a
    well-defined (uniform) distribution rather than a division by zero, which
    is why the spec puts a null slot here at all.

    >>> p = teacher_distribution([0.0, 0.0], r_null=0.0)
    >>> [round(x, 6) for x in p]
    [0.333333, 0.333333, 0.333333]
    >>> p = teacher_distribution([5.0, 0.0], r_null=0.0)
    >>> p[0] > p[1] and abs(sum(p) - 1.0) < 1e-9
    True
    """
    if tau <= 0:
        raise KDRError(f"tau must be positive, got {tau}")
    if not relevances:
        raise KDRError("no regions: KD-R is undefined for an empty region set")

    logits = [r / tau for r in relevances] + [r_null / tau]
    if not all(math.isfinite(v) for v in logits):
        raise KDRError(f"non-finite logits from relevances={list(relevances)}")

    hi = max(logits)
    exps = [math.exp(v - hi) for v in logits]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        raise KDRError("softmax underflowed")
    return [e / total for e in exps]


def target_mass(
    distribution: Sequence[float], target_indices: Sequence[int]
) -> float:
    """:math:`m_n` — teacher probability mass on the annotated target regions.

    ``distribution`` is the ``M + 1`` output of :func:`teacher_distribution`;
    ``target_indices`` index into regions only, so the null slot can never be
    counted as target mass.

    >>> round(target_mass([0.6, 0.3, 0.1], [0]), 6)
    0.6
    >>> target_mass([0.6, 0.3, 0.1], [])
    0.0
    """
    n_regions = len(distribution) - 1
    for i in target_indices:
        if not 0 <= i < n_regions:
            raise KDRError(
                f"target index {i} out of range for {n_regions} regions "
                "(the null slot is not a target)"
            )
    return sum(distribution[i] for i in target_indices)


@dataclass(frozen=True)
class SampleGate:
    """The result of applying the KD-R gate to one sample."""

    weight: float
    teacher_correct: bool
    mass: float
    confidence: float
    passed: bool
    reason: str
    """Why the sample was gated out, or ``"pass"``. Logged for the Table E
    ablations, which compare gate components against each other."""


def gate_weight(
    *,
    teacher_correct: bool,
    mass: float,
    confidence: float,
    delta: float,
    use_correctness: bool = True,
    use_mass: bool = True,
    use_confidence: bool = True,
) -> SampleGate:
    """:math:`w_n = \\mathbb 1[\\hat a^T = a^*]\\cdot\\mathbb 1[m_n \\ge \\delta]\\cdot c_n`.

    The three ``use_*`` switches exist for Table E, which ablates the gate by
    component ("KD-R chỉ theo :math:`c_n`", "KD-R chỉ theo :math:`m_n`",
    "KD-R không chọn lọc" with :math:`w_n = 1`). They are not tuning knobs.

    ``confidence`` must be a **calibrated** teacher confidence. Master prompt
    §10.5 forbids calling a raw logit a calibrated probability; whatever
    produces this value must record its calibration method.

    >>> gate_weight(teacher_correct=True, mass=0.7, confidence=0.9, delta=0.5).weight
    0.9
    >>> gate_weight(teacher_correct=False, mass=0.7, confidence=0.9, delta=0.5).passed
    False
    """
    if not 0.0 <= confidence <= 1.0:
        raise KDRError(f"confidence must be in [0,1], got {confidence}")
    if not math.isfinite(mass):
        raise KDRError(f"non-finite mass: {mass}")

    if use_correctness and not teacher_correct:
        return SampleGate(0.0, teacher_correct, mass, confidence, False, "teacher_wrong")
    if use_mass and mass < delta:
        return SampleGate(
            0.0, teacher_correct, mass, confidence, False, "mass_below_delta"
        )
    weight = confidence if use_confidence else 1.0
    if weight <= 0.0:
        return SampleGate(0.0, teacher_correct, mass, confidence, False, "zero_confidence")
    return SampleGate(weight, teacher_correct, mass, confidence, True, "pass")


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """:math:`D_{KL}(p \\Vert q)` in nats, with ``p`` the teacher target.

    Terms where ``p_i == 0`` contribute nothing (the :math:`0\\log 0` limit).
    A ``q_i == 0`` against a non-zero ``p_i`` is infinite and is reported as
    such rather than silently floored — in practice the student's softmax
    cannot produce an exact zero, so this firing means something upstream is
    wrong.

    >>> round(kl_divergence([0.5, 0.5], [0.5, 0.5]), 9)
    0.0
    >>> kl_divergence([1.0, 0.0], [0.5, 0.5]) > 0
    True
    """
    if len(p) != len(q):
        raise KDRError(f"length mismatch: {len(p)} vs {len(q)}")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        total += pi * math.log(pi / qi)
    return total


def weighted_kd_r(
    targets: Sequence[Sequence[float]],
    students: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    eps: float = EPS,
) -> float:
    """The batch KD-R loss, pure-Python reference implementation.

    This is the specification that :func:`kd_r_loss` must agree with; the test
    suite checks them against each other when torch is available.

    Returns ``0.0`` when every sample is gated out — an all-gated batch
    contributes no gradient, which is correct, and is not an error.
    """
    if not (len(targets) == len(students) == len(weights)):
        raise KDRError("targets, students and weights must have equal length")
    num = 0.0
    den = 0.0
    for t, s, w in zip(targets, students, weights):
        if w <= 0.0:
            continue
        num += w * kl_divergence(t, s)
        den += w
    if den <= 0.0:
        return 0.0
    return num / (den + eps)


def kd_r_loss(teacher_probs, student_logits, weights, *, eps: float = EPS):
    """Torch KD-R loss. ``teacher_probs`` must already be detached.

    Args:
        teacher_probs: ``(B, M+1)`` teacher distribution. **Detached** — a
            gradient reaching the teacher target would let the student move
            its own target, which master prompt §8.2 explicitly forbids. This
            function asserts it rather than calling ``.detach()`` silently, so
            a mistake upstream surfaces instead of being papered over.
        student_logits: ``(B, M+1)`` student region scores, null slot last.
        weights: ``(B,)`` gate weights from :func:`gate_weight`.

    Shapes must match exactly; a student that forgot its null slot would
    otherwise broadcast into a quietly wrong loss.
    """
    import torch  # lazy: keeps this module importable without a torch install
    import torch.nn.functional as F

    if teacher_probs.shape != student_logits.shape:
        raise KDRError(
            f"shape mismatch: teacher {tuple(teacher_probs.shape)} "
            f"vs student {tuple(student_logits.shape)}"
        )
    if teacher_probs.requires_grad:
        raise KDRError(
            "teacher_probs requires grad; KD-R targets must be detached "
            "(master prompt §8.2)"
        )
    if not torch.isfinite(student_logits).all():
        raise KDRError("non-finite student logits")

    log_student = F.log_softmax(student_logits, dim=-1)
    log_teacher = torch.log(teacher_probs.clamp_min(1e-12))
    per_sample = (teacher_probs * (log_teacher - log_student)).sum(dim=-1)

    w = weights.to(per_sample.dtype)
    return (per_sample * w).sum() / (w.sum() + eps)
