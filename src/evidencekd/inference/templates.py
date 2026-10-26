"""Templated text — the locked answer string and baselines B8 / B9.

* :func:`locked_answer_text` renders the decision branch's heads into the
  short answer that gets locked (design §7.3) and that the bridge prefixes
  to the ``<loc>`` tokens.
* :func:`template_claims_simple` (B8) — one presence(+side) observation.
* :func:`template_claims_full` (B9) — every fact as a fixed sentence.

Both baselines produce the same claim grammar the generators do, so the
verifier and Table C score them identically. Pure Python.
"""
from __future__ import annotations

from typing import Sequence

from evidencekd.claims.parser import Claim
from evidencekd.facts.geometry import Fact, FactTable


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def locked_answer_text(question_type: str, *, finding: str | None, present: bool,
                       findings_present: Sequence[str] = (), side: str | None = None,
                       zone: str | None = None, count: int | None = None) -> str:
    """Short, type-appropriate answer. The finding name is always spelled out
    so VinDr's text-based Table 4 can credit it."""
    f = finding or "the finding"
    if question_type in ("Is_there", "Yes_No"):
        return f"Yes, {f} is present." if present else f"No, {f} is not seen."
    if question_type == "Where":
        if not present:
            return f"{f} is not seen."
        where = " ".join(x for x in (side, zone) if x)
        return f"{f} is in the {where} lung." if where and side != "bilateral" else (
            f"{f} is seen bilaterally." if side == "bilateral" else f"{f} is present.")
    if question_type == "How_many":
        if count is None:
            return f"{f} is present." if present else f"No {f} is seen."
        return f"There {'is' if count == 1 else 'are'} {_plural(count, f)}."
    # What / Which: list what the image contains
    names = list(findings_present) or ([f] if present else [])
    if not names:
        return "No finding is seen."
    return "The findings are " + ", ".join(names) + "." if len(names) > 1 else f"The finding is {names[0]}."


def _obs(text: str, fact: Fact) -> Claim:
    return Claim("observation", text, fact_ids=(fact.id,), evidence_ids=tuple(fact.evidence_ids))


def fact_sentence(f: Fact) -> str:
    v = f.value
    if f.attr == "presence":
        return f"{v} is present."
    if f.attr == "absence":
        return f"No {v} is seen."
    if f.attr == "side":
        return "The finding is bilateral." if v == "bilateral" else f"The finding is on the {v} side."
    if f.attr == "zone":
        return f"The finding lies in the {v} zone of the lung."
    if f.attr == "count":
        return f"There {'is' if v == 1 else 'are'} {_plural(int(v), 'lesion')}."
    if f.attr == "extent":
        return f"The involvement is {v}."
    if f.attr == "co_occurrence":
        a, b = v  # type: ignore[misc]
        return f"{a} co-occurs with {b}."
    if f.attr == "ctr":
        return f"The cardiothoracic ratio is {float(v):.2f}."
    return f"{f.attr} is {v}."


def template_claims_full(ft: FactTable, *, background: Sequence[str] = ()) -> list[Claim]:
    """B9: every fact that cites at least one region, verbatim."""
    out = [_obs(fact_sentence(f), f) for f in ft.facts if f.evidence_ids]
    out += [Claim("background", b) for b in background]
    return out


def template_claims_simple(ft: FactTable, finding: str | None, *, background: Sequence[str] = ()) -> list[Claim]:
    """B8: the presence fact of the asked finding, plus its side if known."""
    out: list[Claim] = []
    pres = [f for f in ft.facts if f.attr == "presence" and (finding is None or f.value == finding)]
    if pres:
        p = pres[0]
        out.append(_obs(fact_sentence(p), p))
        for f in ft.facts:
            if f.attr == "side" and set(f.evidence_ids) & set(p.evidence_ids):
                out.append(_obs(fact_sentence(f), f))
                break
    else:
        absent = [f for f in ft.facts if f.attr == "absence" and f.value == finding]
        if absent and absent[0].evidence_ids:
            out.append(_obs(fact_sentence(absent[0]), absent[0]))
    out += [Claim("background", b) for b in background[:1]]
    return out
