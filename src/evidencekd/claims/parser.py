"""Claim grammar, parsing and citation validation (method spec §4.3).

The generator emits observation claims in a tagged form::

    <obs facts=F3> The cardiac silhouette is enlarged. [R01] </obs>

and background claims as plain text drawn from an approved per-finding list.

Two claim types, two different standards
----------------------------------------
``observation``
    An assertion about *this* image. **Must** cite at least one region id and
    **must** pass the verifier. Counted in supported-claim precision,
    citation precision/recall and contradiction rate.

``background``
    General clinical knowledge ("cardiomegaly is often associated with chronic
    hypertension"). Drawn from a vetted list, carries no citation, and is
    **excluded from every evidence metric** (spec §2). Letting background text
    into supported-claim precision would inflate it with statements that were
    never about the image.

Constrained decoding
--------------------
Spec §4.3: *"token trích dẫn chỉ lấy từ ID vùng của dữ kiện đã khai báo; mệnh
đề quan sát không đóng được nếu chưa trích dẫn."* :func:`allowed_citations`
and :func:`validate_claim` are the rule half of that constraint — a decoder
calls them to decide which citation tokens are legal and whether ``</obs>``
may be emitted yet.

Pure Python: no torch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

ClaimType = Literal["observation", "background"]
Status = Literal["supported", "contradicted", "insufficient"]

#: The three verifier states. ``insufficient`` is never promoted to
#: ``supported`` (master prompt §7.5) — "we could not tell" is not "yes".
SUPPORTED: Status = "supported"
CONTRADICTED: Status = "contradicted"
INSUFFICIENT: Status = "insufficient"

_OBS_RE = re.compile(
    r"<obs\s+facts=(?P<facts>[^>]*)>(?P<body>.*?)</obs>", re.DOTALL
)
_CITATION_RE = re.compile(r"\[([A-Za-z0-9_]+)\]")
_FACT_ID_RE = re.compile(r"\bF\d+\b")


class ClaimFormatError(ValueError):
    """Raised when claim text violates the grammar."""


@dataclass(frozen=True)
class Claim:
    """One parsed claim."""

    type: ClaimType
    text: str
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    status: Status | None = None
    reject_reason: str | None = None

    @property
    def is_observation(self) -> bool:
        return self.type == "observation"

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {"type": self.type, "text": self.text}
        if self.is_observation:
            d["fact_ids"] = list(self.fact_ids)
            d["evidence_ids"] = list(self.evidence_ids)
            if self.status is not None:
                d["status"] = self.status
        return d


def parse_claims(text: str) -> list[Claim]:
    """Parse generator output into claims.

    Text outside ``<obs>`` blocks becomes ``background`` claims, split on
    sentence boundaries. Empty segments are dropped.

    >>> cs = parse_claims("<obs facts=F3> Heart enlarged. [R01] </obs>")
    >>> cs[0].type, cs[0].fact_ids, cs[0].evidence_ids
    ('observation', ('F3',), ('R01',))
    """
    claims: list[Claim] = []
    cursor = 0
    for match in _OBS_RE.finditer(text):
        before = text[cursor : match.start()].strip()
        claims.extend(_background_claims(before))

        facts = tuple(_FACT_ID_RE.findall(match.group("facts")))
        body = match.group("body")
        evidence = tuple(dict.fromkeys(_CITATION_RE.findall(body)))
        claim_text = _CITATION_RE.sub("", body)
        claim_text = " ".join(claim_text.split())
        claims.append(
            Claim(
                type="observation",
                text=claim_text,
                fact_ids=facts,
                evidence_ids=evidence,
            )
        )
        cursor = match.end()

    claims.extend(_background_claims(text[cursor:].strip()))
    return claims


def _background_claims(blob: str) -> list[Claim]:
    if not blob.strip():
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", blob) if s.strip()]
    return [Claim(type="background", text=s) for s in sentences]


def allowed_citations(
    fact_ids: Iterable[str], fact_table: Mapping[str, Sequence[str]]
) -> frozenset[str]:
    """Region ids a claim citing ``fact_ids`` is permitted to emit.

    This is the decoder's citation whitelist. A region that no declared fact
    references is **not** citable, which is what stops the generator from
    inventing a region id or borrowing an unrelated one.

    >>> allowed_citations(["F3"], {"F3": ["R01", "A_thorax"]}) == {"R01", "A_thorax"}
    True
    >>> allowed_citations(["F9"], {"F3": ["R01"]}) == frozenset()
    True
    """
    allowed: set[str] = set()
    for fid in fact_ids:
        allowed.update(fact_table.get(fid, ()))
    return frozenset(allowed)


def can_close_observation(body_so_far: str) -> bool:
    """Whether ``</obs>`` may be emitted yet.

    False until at least one citation has been produced — spec §4.3 forbids
    closing an observation claim that has not cited.

    >>> can_close_observation("Heart enlarged.")
    False
    >>> can_close_observation("Heart enlarged. [R01]")
    True
    """
    return bool(_CITATION_RE.search(body_so_far))


def validate_claim(
    claim: Claim,
    *,
    fact_table: Mapping[str, Sequence[str]],
    known_region_ids: Iterable[str],
) -> Claim:
    """Check a claim against the grammar and the declared facts.

    Returns the claim unchanged when valid, or with ``status=insufficient``
    and a ``reject_reason`` when not. Rejections are **never** silent: the
    reason is kept so the KD-C filtering report can break down why claims were
    dropped.

    Rejection cases, all of which are real failure modes of a constrained
    decoder that has slipped its constraint:

    * an observation with no citation;
    * a citation to a region id that does not exist (hallucinated region);
    * a citation to a real region that the claim's declared facts do not
      reference (borrowed evidence);
    * an observation declaring a fact id absent from the table.
    """
    if not claim.is_observation:
        return claim

    known = set(known_region_ids)

    unknown_facts = [f for f in claim.fact_ids if f not in fact_table]
    if unknown_facts:
        return _reject(claim, f"undeclared fact ids: {sorted(unknown_facts)}")

    if not claim.evidence_ids:
        return _reject(claim, "observation claim has no citation")

    hallucinated = [r for r in claim.evidence_ids if r not in known]
    if hallucinated:
        return _reject(claim, f"citation to unknown region: {sorted(hallucinated)}")

    permitted = allowed_citations(claim.fact_ids, fact_table)
    borrowed = [r for r in claim.evidence_ids if r not in permitted]
    if borrowed:
        return _reject(
            claim, f"citation not supported by declared facts: {sorted(borrowed)}"
        )

    if not claim.text.strip():
        return _reject(claim, "empty claim text")

    return claim


def _reject(claim: Claim, reason: str) -> Claim:
    return Claim(
        type=claim.type,
        text=claim.text,
        fact_ids=claim.fact_ids,
        evidence_ids=claim.evidence_ids,
        status=INSUFFICIENT,
        reject_reason=reason,
    )


def render_claims(claims: Sequence[Claim]) -> str:
    """Serialise claims back to the generator's wire format.

    Round-trips with :func:`parse_claims` for well-formed input, which the
    test suite checks.
    """
    parts: list[str] = []
    for c in claims:
        if c.is_observation:
            facts = ",".join(c.fact_ids)
            cites = " ".join(f"[{r}]" for r in c.evidence_ids)
            parts.append(f"<obs facts={facts}> {c.text} {cites} </obs>".replace("  ", " "))
        else:
            parts.append(c.text)
    return " ".join(parts)
