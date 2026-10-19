"""KD-C — claim-level rationale filtering distillation (method spec §4.4).

The teacher is given the fact table computed from annotation and generates
rationales. Every *observation* claim must clear three independent gates
before it may become a training target for generator S:

1. it matches a declared fact,
2. it cites the right region,
3. the verifier labels it ``supported``.

.. math::
    L_{KD\\text{-}C} = -\\sum_{c\\in\\mathcal C_{pass}}\\sum_t
        \\log p_S\\big(y_{c,t}\\mid y_{c,<t}, Q, a, \\mathcal F_{sel}, \\{h_i\\}\\big)

The filtering granularity is the contribution being tested
----------------------------------------------------------
Table E contrasts three regimes, so granularity is a first-class parameter,
not a constant:

``none``
    Every teacher claim is a target. The "KD-C không lọc" row.
``sample``
    One bad claim discards the entire rationale. The "mức mẫu" row — the
    conventional approach, and the thing claim-level filtering is claimed to
    beat.
``claim``
    Each claim is judged on its own; passing claims survive their failing
    siblings. The proposed "mức mệnh đề" row.

Because ``claim`` is the proposed method, comparing it against ``sample`` and
``none`` **under an identical student and schedule** is what makes or breaks
contribution C1. The comparison must not be confounded by also changing the
retention rate silently — :attr:`FilterReport.retention` is recorded so the
"KD-C chọn ngẫu nhiên" control can be matched to it.

Pure Python. The token-level loss is :func:`kd_c_loss`, which imports torch
lazily.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal, Mapping, Sequence

from ..claims.parser import (
    CONTRADICTED,
    INSUFFICIENT,
    SUPPORTED,
    Claim,
    Status,
    validate_claim,
)

Granularity = Literal["none", "sample", "claim"]

#: A verifier maps a claim to one of the three states.
Verifier = Callable[[Claim], Status]


@dataclass
class FilterReport:
    """What the filter did, for the KD-C statistics the proposal requires."""

    total_observations: int = 0
    kept: int = 0
    dropped_by_status: Counter[str] = field(default_factory=Counter)
    dropped_by_format: Counter[str] = field(default_factory=Counter)
    rationales_seen: int = 0
    rationales_fully_dropped: int = 0

    @property
    def retention(self) -> float:
        """Fraction of observation claims kept. ``nan`` when none were seen."""
        if self.total_observations == 0:
            return float("nan")
        return self.kept / self.total_observations

    def as_dict(self) -> dict[str, object]:
        return {
            "total_observations": self.total_observations,
            "kept": self.kept,
            "retention": self.retention,
            "dropped_by_status": dict(self.dropped_by_status),
            "dropped_by_format": dict(self.dropped_by_format),
            "rationales_seen": self.rationales_seen,
            "rationales_fully_dropped": self.rationales_fully_dropped,
        }


@dataclass(frozen=True)
class FilteredRationale:
    """One rationale after filtering."""

    kept_claims: tuple[Claim, ...]
    dropped_claims: tuple[Claim, ...]
    usable: bool
    """False when nothing survived — the sample contributes no KD-C loss."""


def filter_rationale(
    claims: Sequence[Claim],
    *,
    fact_table: Mapping[str, Sequence[str]],
    known_region_ids: Sequence[str],
    verifier: Verifier,
    granularity: Granularity = "claim",
    report: FilterReport | None = None,
) -> FilteredRationale:
    """Apply the KD-C gate to one teacher rationale.

    Background claims pass through untouched: they carry no citation by
    design and are excluded from evidence metrics, so verifying them would be
    a category error.

    ``insufficient`` is treated as a **failure to keep**, never as a pass.
    Master prompt §7.5 is explicit that it must not become ``supported``.
    """
    if report is not None:
        report.rationales_seen += 1

    kept: list[Claim] = []
    dropped: list[Claim] = []

    for claim in claims:
        if not claim.is_observation:
            kept.append(claim)
            continue

        if report is not None:
            report.total_observations += 1

        checked = validate_claim(
            claim, fact_table=fact_table, known_region_ids=known_region_ids
        )
        if checked.reject_reason is not None:
            if report is not None:
                report.dropped_by_format[checked.reject_reason.split(":")[0]] += 1
            dropped.append(checked)
            continue

        if granularity == "none":
            # No verification at all — the unfiltered control.
            kept.append(Claim(**{**checked.__dict__, "status": None}))
            if report is not None:
                report.kept += 1
            continue

        status = verifier(checked)
        decided = Claim(
            type=checked.type,
            text=checked.text,
            fact_ids=checked.fact_ids,
            evidence_ids=checked.evidence_ids,
            status=status,
        )
        if status == SUPPORTED:
            kept.append(decided)
            if report is not None:
                report.kept += 1
        else:
            dropped.append(decided)
            if report is not None:
                report.dropped_by_status[status] += 1

    observations_kept = [c for c in kept if c.is_observation]

    if granularity == "sample" and dropped:
        # One bad claim discards the whole rationale.
        if report is not None:
            report.kept -= len(observations_kept)
            report.rationales_fully_dropped += 1
        return FilteredRationale(
            kept_claims=(), dropped_claims=tuple(kept + dropped), usable=False
        )

    usable = bool(observations_kept)
    if not usable and report is not None:
        report.rationales_fully_dropped += 1

    return FilteredRationale(
        kept_claims=tuple(kept), dropped_claims=tuple(dropped), usable=usable
    )


def build_targets(
    rationales: Sequence[Sequence[Claim]],
    *,
    fact_table: Mapping[str, Sequence[str]],
    known_region_ids: Sequence[str],
    verifier: Verifier,
    granularity: Granularity = "claim",
    report: FilterReport | None = None,
) -> list[Claim]:
    """Collapse the teacher's several rationales into one target claim set.

    The teacher generates three rationales per sample (spec §4.4). We keep the
    union of *passing* claims, de-duplicated on claim text, rather than
    concatenating all three — concatenation would repeat the same observation
    three times and teach the student to be redundant, which Table C's
    "độ gọn" (concision) metric penalises.

    De-duplication is on normalised text only. Two claims with the same words
    but different citations are a genuine conflict; the first is kept and the
    second dropped, because emitting both would cite two different regions for
    one statement.
    """
    seen: set[str] = set()
    targets: list[Claim] = []
    for claims in rationales:
        filtered = filter_rationale(
            claims,
            fact_table=fact_table,
            known_region_ids=known_region_ids,
            verifier=verifier,
            granularity=granularity,
            report=report,
        )
        if not filtered.usable:
            continue
        for claim in filtered.kept_claims:
            key = " ".join(claim.text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            targets.append(claim)
    return targets


def kd_c_loss(logits, target_ids, pad_id: int, *, label_smoothing: float = 0.0):
    """Token-level cross-entropy over the surviving claims.

    Args:
        logits: ``(B, T, V)`` generator logits.
        target_ids: ``(B, T)`` target token ids, padded with ``pad_id``.
        pad_id: ignored in the loss. Master prompt §7.4 requires padding be
            masked properly; letting pad tokens contribute would make the loss
            depend on batch composition.

    Returns a scalar. A batch in which every sample was filtered out should
    not reach here — callers skip the KD-C term entirely for those, rather
    than passing an empty batch.
    """
    import torch
    import torch.nn.functional as F

    if logits.dim() != 3:
        raise ValueError(f"expected (B, T, V) logits, got {tuple(logits.shape)}")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"target shape {tuple(target_ids.shape)} does not match "
            f"logits {tuple(logits.shape[:2])}"
        )
    if (target_ids != pad_id).sum() == 0:
        raise ValueError("every target token is padding; skip KD-C for this batch")

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )
