"""Prompts for the teacher and the rule that turns free text into claims.

* :func:`rationale_prompt` — KD-C (spec §4.4): the teacher sees the question,
  the locked answer and the annotation-derived fact table, and is asked for
  observations in the ``<obs facts=F1> ... [R01] </obs>`` grammar.
* :func:`claims_from_text` — if the teacher followed the grammar, parse it;
  otherwise split sentences and attach each to the facts it matches
  (finding + side + zone + count), citing those facts' regions. Sentences
  that match nothing become ``background``. Design §7.5 risk mitigation.

Pure Python.
"""
from __future__ import annotations

import re
from typing import Sequence

from evidencekd.claims.parser import Claim, parse_claims
from evidencekd.data.answer_norm import parse_count, parse_side
from evidencekd.facts.build import serialize_facts
from evidencekd.facts.geometry import Fact, FactTable
from evidencekd.models.verifier_rules import _zone_in_text, findings_in_text

GRAMMAR_HELP = (
    "Write 2-4 short observations. Each observation must be wrapped as "
    "<obs facts=F#> sentence [R##] </obs>, where F# is the id of the fact it states "
    "and [R##] cites one of that fact's regions. Only cite regions listed with the "
    "fact. After the observations you may add one background sentence without tags."
)

FEW_SHOT = (
    "Facts: F1: presence Cardiomegaly [R01]; F2: side bilateral [R01, A_thorax]; "
    "F3: presence Pleural effusion [R02]; F4: side right [R02, A_thorax]\n"
    "Question: Where is the Pleural effusion?\nAnswer: Pleural effusion is in the right lung.\n"
    "Observations: <obs facts=F3> There is a pleural effusion [R02] </obs> "
    "<obs facts=F4> It lies on the right side [R02] </obs> "
    "Pleural effusion is fluid accumulating in the pleural space."
)


def rationale_prompt(question: str, answer: str, ft: FactTable) -> str:
    return (f"{GRAMMAR_HELP}\n\nExample:\n{FEW_SHOT}\n\n"
            f"Facts: {serialize_facts(ft)}\nQuestion: {question}\nAnswer: {answer}\nObservations:")


_SENT_RE = re.compile(r"(?<=[.!?])\s+")
# Sentences carrying general knowledge rather than an observation of THIS image.
_GENERIC_RE = re.compile(
    r"\b(often|typically|usually|commonly|generally|may|might|can|could|tend|tends|"
    r"indicative|indicates|associated|suggests|suggestive|refers to|is defined|caused by|"
    r"due to|result of|secondary to|known as)\b")


def _facts_for_sentence(sent: str, ft: FactTable) -> list[Fact]:
    """Facts a sentence asserts: presence of the findings it names, plus the
    side/zone/count facts attached to those presence facts that it also states."""
    named = findings_in_text(sent)
    side, zone = parse_side(sent), _zone_in_text(sent)
    count = parse_count(sent, question_type="How_many")
    out: list[Fact] = []
    pres = [f for f in ft.facts if f.attr == "presence" and f.value in named]
    out += pres
    regions = {rid for f in pres for rid in f.evidence_ids}
    for f in ft.facts:
        if not (set(f.evidence_ids) & regions):
            continue
        if f.attr == "side" and side is not None and f.value == side:
            out.append(f)
        elif f.attr == "zone" and zone is not None and f.value == zone:
            out.append(f)
        elif f.attr == "count" and count is not None and int(f.value) == count:  # type: ignore[arg-type]
            out.append(f)
    return out


def claims_from_text(text: str, ft: FactTable) -> tuple[list[Claim], bool]:
    """``(claims, followed_grammar)``."""
    if "<obs" in text and "</obs>" in text:
        return parse_claims(text), True
    claims: list[Claim] = []
    for sent in _SENT_RE.split(" ".join(text.split())):
        sent = sent.strip()
        if not sent:
            continue
        facts = [] if _GENERIC_RE.search(sent.lower()) else _facts_for_sentence(sent, ft)
        if facts:
            fids = tuple(f.id for f in facts)
            cites = tuple(dict.fromkeys(rid for f in facts for rid in f.evidence_ids))
            claims.append(Claim("observation", sent, fact_ids=fids, evidence_ids=cites[:2]))
        else:
            claims.append(Claim("background", sent))
    return claims, False


def canonical_answer(answer_text: str) -> str:
    """``a*`` for KD-R: the released answer with ``<loc>`` tokens removed
    (what the teacher was trained to emit before the box) — design §7.4."""
    out = re.sub(r"\s*<loc_-?\d+_-?\d+_-?\d+_-?\d+>", "", answer_text)
    out = re.sub(r"\s+([.,;:])", r"\1", out)
    return " ".join(out.split()).strip()


def teacher_named_finding(generated: str, gt_finding: str) -> bool:
    """Teacher correctness for the KD-R gate: the generated answer names the
    gold finding, using VinDr's negation-aware parser when importable."""
    try:
        from vindr_cxr_vqa.evaluate import mentioned  # sibling scorer's parser
        return gt_finding in mentioned(generated)
    except ImportError:
        return gt_finding in findings_in_text(generated)
