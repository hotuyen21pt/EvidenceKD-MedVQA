"""Citation constraint — the decoding half of ``claims.parser`` (spec §4.3).

Two layers, so the guarantee holds whatever the decoder does:

* :class:`CitationLogitsProcessor` — during HF ``generate``: once an
  observation is open and a ``[`` has been emitted, only tokens that continue
  one of the allowed region ids may follow; ``</obs>`` is blocked until the
  open observation holds a citation. Best effort — tokenisers split ids
  unpredictably, so it works on the decoded suffix, not on token ids.
* :func:`enforce_citations` — after decoding: citations outside the whitelist
  are dropped; an observation left without a citation gets the first region
  of its declared facts, or is demoted to ``background`` when it declared no
  usable fact. This is the rule the metrics can rely on.

The rule half is pure Python.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

from evidencekd.claims.parser import Claim, allowed_citations, can_close_observation

_OPEN_RE = re.compile(r"<obs\s+facts=([^>]*)>")
_FACT_RE = re.compile(r"\bF\d+\b")


def enforce_citations(claims: Sequence[Claim], fact_table: Mapping[str, Sequence[str]],
                      known_region_ids: Sequence[str]) -> list[Claim]:
    known = set(known_region_ids)
    out: list[Claim] = []
    for c in claims:
        if not c.is_observation:
            out.append(c); continue
        fids = tuple(f for f in c.fact_ids if f in fact_table)
        allowed = allowed_citations(fids, fact_table) & known
        cites = tuple(r for r in c.evidence_ids if r in allowed)
        if not cites and allowed:
            first = next((r for f in fids for r in fact_table[f] if r in known), None)
            cites = (first,) if first else ()
        if not cites or not fids:
            out.append(Claim("background", c.text))
            continue
        out.append(Claim("observation", c.text, fids, cites, c.status, c.reject_reason))
    return out


class CitationState:
    """Tracks the decoded suffix of one sequence to decide what may follow."""

    def __init__(self, fact_table: Mapping[str, Sequence[str]], known_region_ids: Sequence[str]):
        self.fact_table = fact_table
        self.known = set(known_region_ids)

    def allowed_now(self, text: str) -> set[str] | None:
        """Region ids allowed after the current ``[``, or None when not inside a citation."""
        last_open = text.rfind("<obs")
        last_close = text.rfind("</obs>")
        if last_open < 0 or last_close > last_open:
            return None
        m = _OPEN_RE.search(text, last_open)
        if not m:
            return None
        fids = _FACT_RE.findall(m.group(1))
        allowed = allowed_citations(fids, self.fact_table) & self.known
        body = text[m.end():]
        lb = body.rfind("[")
        if lb < 0 or "]" in body[lb:]:
            return None
        return allowed if allowed else set()

    def may_close(self, text: str) -> bool:
        last_open = text.rfind("<obs")
        m = _OPEN_RE.search(text, last_open) if last_open >= 0 else None
        if not m:
            return True
        return can_close_observation(text[m.end():])


class CitationLogitsProcessor:  # transformers.LogitsProcessor duck-type (torch lazy)
    def __init__(self, tokenizer, fact_tables: Sequence[Mapping[str, Sequence[str]]],
                 known_region_ids: Sequence[Sequence[str]], *, prompt_len: int = 0, decoder_only: bool = False):
        self.tok = tokenizer
        self.states = [CitationState(ft, ids) for ft, ids in zip(fact_tables, known_region_ids)]
        self.prompt_len = prompt_len
        self.decoder_only = decoder_only
        self._close_ids = None

    def _closing_token_ids(self):
        if self._close_ids is None:
            ids = set()
            for s in ("</obs>", "</", "</ob", "obs>"):
                for t in self.tok(s, add_special_tokens=False)["input_ids"]:
                    ids.add(t)
            self._close_ids = ids
        return self._close_ids

    def __call__(self, input_ids, scores):
        import torch
        vocab = scores.shape[-1]
        n = input_ids.shape[0]
        reps = max(1, n // max(len(self.states), 1))
        for b in range(n):
            st = self.states[min(b // reps, len(self.states) - 1)]
            seq = input_ids[b, self.prompt_len:] if self.decoder_only else input_ids[b]
            text = self.tok.decode(seq, skip_special_tokens=True)
            allowed = st.allowed_now(text)
            if allowed is not None:
                lb = text.rfind("[")
                partial = text[lb + 1:]
                cont = [r[len(partial):] for r in allowed if r.startswith(partial)]
                keep = torch.full((vocab,), float("-inf"), device=scores.device)
                for c in cont:
                    target = (c + "]") if c else "]"
                    for L in range(1, len(target) + 1):
                        for t in self.tok(target[:L], add_special_tokens=False)["input_ids"][:1]:
                            keep[t] = scores[b, t]
                if torch.isfinite(keep).any():
                    scores[b] = keep
            elif not st.may_close(text):
                for t in self._closing_token_ids():
                    if t < vocab:
                        scores[b, t] = float("-inf")
        return scores
