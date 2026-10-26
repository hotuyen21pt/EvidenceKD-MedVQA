"""From regions + anatomy + decision-branch outputs to the fact table and its
text serialisation (spec §4.2, design §7.2). Pure Python.
"""
from __future__ import annotations

import json
import os
from typing import Mapping, Sequence

from evidencekd.data.regions import KIND_FINDING, RegionSet
from evidencekd.facts.geometry import Anatomy, Fact, FactTable, build_fact_table
from evidencekd.labels import VQA_CLASSES


def build_facts(regions: RegionSet, anatomy: Anatomy, *, present_classes: Sequence[str],
                whole_image_scores: Mapping[str, float] | None = None,
                absence_threshold: float = 0.5) -> FactTable:
    """``geometry.build_fact_table`` over the finding regions of a RegionSet."""
    return build_fact_table(
        regions=regions.mapping(kind=KIND_FINDING), anatomy=anatomy,
        present_classes=[c for c in VQA_CLASSES if c in set(present_classes)],
        whole_image_scores=whole_image_scores, absence_threshold=absence_threshold,
        view_position=None)


def fact_value_text(f: Fact) -> str:
    if f.attr == "co_occurrence":
        a, b = f.value  # type: ignore[misc]
        return f"{a} with {b}"
    if f.attr == "ctr":
        return f"{float(f.value):.2f}"
    return str(f.value)


def serialize_facts(ft: FactTable) -> str:
    """``F1: presence Cardiomegaly [R01]; F2: side left [R01, A_thorax]; ...``

    The text the generators read and the teacher is prompted with. Region
    ids appear verbatim so a generator can only cite what it was shown.
    """
    parts = []
    for f in ft.facts:
        cites = ", ".join(f.evidence_ids) if f.evidence_ids else "-"
        parts.append(f"{f.id}: {f.attr} {fact_value_text(f)} [{cites}]")
    return "; ".join(parts)


def facts_for_classes(ft: FactTable, classes: Sequence[str]) -> list[Fact]:
    """Facts whose value or region labels concern any of ``classes``."""
    wanted = set(classes)
    out = []
    for f in ft.facts:
        if f.attr in ("presence", "absence") and f.value in wanted:
            out.append(f)
        elif f.attr == "co_occurrence" and (f.value[0] in wanted or f.value[1] in wanted):  # type: ignore[index]
            out.append(f)
        elif f.attr in ("side", "zone", "count", "extent"):
            # these facts are emitted right after their presence fact and share
            # its regions; attribute by region overlap with the presence facts
            pres = [p for p in ft.facts if p.attr == "presence" and p.value in wanted]
            if any(set(p.evidence_ids) & set(f.evidence_ids) for p in pres):
                out.append(f)
    return out


def fact_region_map(ft: FactTable) -> dict[str, list[str]]:
    """``{fact_id: [region ids]}`` — the citation whitelist (claims.parser)."""
    return {f.id: list(f.evidence_ids) for f in ft.facts}


def facts_to_dict(ft: FactTable) -> list[dict]:
    return ft.as_dicts()


def facts_from_dicts(rows: Sequence[dict], regions: RegionSet) -> FactTable:
    ft = FactTable(facts=[], region_ids=dict(regions.mapping()))
    for r in rows:
        v = r["value"]
        if r["attr"] == "co_occurrence" and isinstance(v, list):
            v = tuple(v)
        ft.facts.append(Fact(r["id"], r["attr"], v, tuple(r.get("evidence_ids", ())), r.get("note")))
    return ft


# ---------------------------------------------------------------------------
# Approved background sentences (spec §2: "background lấy từ danh sách đã duyệt")
# ---------------------------------------------------------------------------
def build_background(samples, *, max_per_finding: int = 20, min_words: int = 4) -> dict[str, list[str]]:
    """Distinct ``reason`` sentences per finding from the training split."""
    seen: dict[str, dict[str, int]] = {c: {} for c in VQA_CLASSES}
    for s in samples:
        txt = " ".join((s.reason or "").split())
        if not txt or len(txt.split()) < min_words or s.gt_finding not in seen:
            continue
        if "specified location" in txt.lower():
            continue  # the degenerate 'Visible at specified location.'
        seen[s.gt_finding][txt] = seen[s.gt_finding].get(txt, 0) + 1
    return {c: [t for t, _ in sorted(d.items(), key=lambda kv: -kv[1])[:max_per_finding]]
            for c, d in seen.items()}


def save_background(bg: Mapping[str, Sequence[str]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bg, f, ensure_ascii=False, indent=1)


def load_background(path: str | None) -> dict[str, list[str]]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
