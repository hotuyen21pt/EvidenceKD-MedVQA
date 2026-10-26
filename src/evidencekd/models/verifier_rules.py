"""Verifier v1 — rules over the fact table (design §7.6).

An observation claim is

* ``supported``   when its grammar is clean (``claims.parser.validate_claim``)
  and every attribute its text asserts — finding, side, zone, count,
  absence — agrees with the facts it cites;
* ``contradicted`` when an asserted attribute disagrees with a cited fact
  (left vs right, present vs absent, 2 vs 3, wrong finding);
* ``insufficient`` when the text asserts something no cited fact covers,
  or when the grammar check already rejected it.

Used as the KD-C filter; the learned verifier v2 (``verifier.py``) is what
scores results, so the filter never grades itself. Pure Python, and the
``Verifier`` callable shape ``kd.kd_c`` expects.
"""
from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

from evidencekd.claims.parser import CONTRADICTED, INSUFFICIENT, SUPPORTED, Claim, Status, validate_claim
from evidencekd.data.answer_norm import parse_count, parse_side, parse_yes_no
from evidencekd.facts.geometry import Fact, FactTable
from evidencekd.labels import VQA_CLASSES, VQA_SYNONYMS

_ZONES = ("upper", "middle", "lower")
_NEG_RE = re.compile(r"\b(no|not|without|absent|absence of|free of)\b")


def findings_in_text(text: str) -> set[str]:
    low = text.lower()
    out = set()
    for cls, syns in VQA_SYNONYMS.items():
        if any(s in low for s in syns):
            out.add(cls)
    # 'lung opacity' contains 'opacity'; longest-match is handled by callers
    # needing one class — a set is right for a sentence that names several
    return out


def _zone_in_text(text: str) -> str | None:
    low = text.lower()
    for z in _ZONES:
        if re.search(rf"\b{z}\b", low):
            return z
    return None


def _asserts_absence(text: str) -> bool:
    return parse_yes_no(text) is False or bool(_NEG_RE.search(text.lower()))


def judge(claim: Claim, facts: Sequence[Fact]) -> Status:
    """Status of one observation against the facts it cites (already validated)."""
    text = claim.text
    by_attr: dict[str, list[Fact]] = {}
    for f in facts:
        by_attr.setdefault(f.attr, []).append(f)

    checked = 0
    # --- finding / presence / absence -----------------------------------
    named = findings_in_text(text)
    present_vals = {f.value for f in by_attr.get("presence", [])}
    for f in by_attr.get("co_occurrence", []):      # both members are present
        present_vals.update(f.value)                 # type: ignore[arg-type]
    absent_vals = {f.value for f in by_attr.get("absence", [])}
    if named:
        checked += 1
        absent_claim = _asserts_absence(text)
        for cls in named:
            # "lung opacity" names both Lung Opacity and (via 'opacity') nothing else; fine
            if cls in present_vals:
                if absent_claim and not any(c in absent_vals for c in named):
                    return CONTRADICTED
            elif cls in absent_vals:
                if not absent_claim:
                    return CONTRADICTED
            else:
                # a finding not covered by any cited fact
                if not (present_vals or absent_vals):
                    return INSUFFICIENT
                # tolerate synonyms overlap ('mass' inside 'Nodule/Mass' sentence): only
                # contradict if NO named class is covered
                if not (named & (present_vals | absent_vals)):
                    return INSUFFICIENT
    # --- side -------------------------------------------------------------
    side = parse_side(text)
    if side is not None:
        checked += 1
        fs = by_attr.get("side", [])
        if not fs:
            return INSUFFICIENT
        if all(f.value != side for f in fs):
            return CONTRADICTED
    # --- zone -------------------------------------------------------------
    zone = _zone_in_text(text)
    if zone is not None:
        checked += 1
        fs = by_attr.get("zone", [])
        if not fs:
            return INSUFFICIENT
        if all(f.value != zone for f in fs):
            return CONTRADICTED
    # --- count ------------------------------------------------------------
    count = parse_count(text, question_type="How_many")
    if count is not None and count > 0:
        checked += 1
        fs = by_attr.get("count", [])
        if not fs:
            return INSUFFICIENT
        if all(int(f.value) != count for f in fs):  # type: ignore[arg-type]
            return CONTRADICTED
    # --- extent -----------------------------------------------------------
    for ext in ("focal", "moderate", "extensive"):
        if re.search(rf"\b{ext}\b", text.lower()):
            checked += 1
            fs = by_attr.get("extent", [])
            if not fs:
                return INSUFFICIENT
            if all(f.value != ext for f in fs):
                return CONTRADICTED
            break
    if checked == 0:
        # the sentence asserts nothing checkable (e.g. "This is visible [R01]")
        return INSUFFICIENT
    return SUPPORTED


def make_rule_verifier(ft: FactTable, known_region_ids: Sequence[str]) -> Callable[[Claim], Status]:
    """A ``kd.kd_c.Verifier`` closed over one sample's fact table."""
    table: Mapping[str, Sequence[str]] = {f.id: f.evidence_ids for f in ft.facts}
    facts_by_id = {f.id: f for f in ft.facts}

    def verify(claim: Claim) -> Status:
        if not claim.is_observation:
            return SUPPORTED
        v = validate_claim(claim, fact_table=table, known_region_ids=known_region_ids)
        if v.reject_reason:
            return INSUFFICIENT
        cited = [facts_by_id[fid] for fid in claim.fact_ids if fid in facts_by_id]
        return judge(claim, cited)

    return verify


def verify_all(claims: Sequence[Claim], ft: FactTable, known_region_ids: Sequence[str]) -> list[Claim]:
    """Return copies of the claims with ``status`` filled by the rules."""
    v = make_rule_verifier(ft, known_region_ids)
    out = []
    for c in claims:
        if c.is_observation:
            out.append(Claim(c.type, c.text, c.fact_ids, c.evidence_ids, v(c), c.reject_reason))
        else:
            out.append(c)
    return out
