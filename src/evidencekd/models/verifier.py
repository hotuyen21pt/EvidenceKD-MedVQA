"""Verifier v2 — learned 3-state classifier (design §7.6; spec §4.3).

Input: ``claim text [SEP] facts of the cited regions`` (serialised). Output:
supported / contradicted / insufficient. Trained on the KD-C cache's kept
claims (positives) plus synthetic negatives built by construction, never by
the rule verifier: swap the finding name (contradicted), flip the side
(contradicted), cite a region of another class (contradicted), assert an
attribute no cited fact covers (insufficient).

    python -m evidencekd.models.verifier train --config configs/generator.yaml

The synthetic-negative construction is pure Python (tested); the model
imports torch lazily.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Sequence

from evidencekd.claims.parser import CONTRADICTED, INSUFFICIENT, SUPPORTED, Claim, parse_claims
from evidencekd.facts.build import fact_value_text
from evidencekd.facts.geometry import Fact
from evidencekd.labels import VQA_CLASSES, VQA_SYNONYMS
from evidencekd.models.text_encoder import SimpleTokenizer, pad_sequences
from evidencekd.utils.config import load_config

STATUSES = (SUPPORTED, CONTRADICTED, INSUFFICIENT)
_SIDE_FLIP = {"left": "right", "right": "left"}
_ZONES = ("upper", "middle", "lower")


def cited_facts_text(claim: Claim, facts: Sequence[Fact]) -> str:
    by_id = {f.id: f for f in facts}
    parts = []
    for fid in claim.fact_ids:
        f = by_id.get(fid)
        if f:
            parts.append(f"{f.attr} {fact_value_text(f)} [{', '.join(f.evidence_ids)}]")
    return "; ".join(parts) if parts else "none"


def verifier_input(claim: Claim, facts: Sequence[Fact]) -> str:
    return f"claim: {claim.text} [SEP] cited: {' '.join(claim.evidence_ids)} [SEP] facts: {cited_facts_text(claim, facts)}"


def synthetic_negatives(claim: Claim, facts: Sequence[Fact], rng: random.Random, n: int = 2) -> list[tuple[Claim, str]]:
    """Perturbed copies of a supported claim with their status by construction."""
    out: list[tuple[Claim, str]] = []
    text = claim.text
    low = text.lower()
    # 1. finding swap -> contradicted
    for cls, syns in VQA_SYNONYMS.items():
        hit = next((s for s in syns if s in low), None)
        if hit:
            other = rng.choice([c for c in VQA_CLASSES if c != cls])
            i = low.find(hit)
            out.append((Claim("observation", text[:i] + other + text[i + len(hit):], claim.fact_ids, claim.evidence_ids), CONTRADICTED))
            break
    # 2. side flip -> contradicted
    m = re.search(r"\b(left|right)\b", low)
    if m:
        s = m.group(1)
        out.append((Claim("observation", re.sub(rf"\b{s}\b", _SIDE_FLIP[s], text, count=1, flags=re.I),
                          claim.fact_ids, claim.evidence_ids), CONTRADICTED))
    # 3. attribute not covered -> insufficient
    has_zone = any(f.attr == "zone" for f in facts if f.id in claim.fact_ids)
    if not has_zone and not any(z in low for z in _ZONES):
        z = rng.choice(_ZONES)
        out.append((Claim("observation", text.rstrip(".") + f" in the {z} zone.", claim.fact_ids, claim.evidence_ids), INSUFFICIENT))
    # 4. count change -> contradicted
    m = re.search(r"\b(\d+)\b", text)
    if m:
        k = int(m.group(1))
        out.append((Claim("observation", text.replace(m.group(1), str(k + 1), 1), claim.fact_ids, claim.evidence_ids), CONTRADICTED))
    rng.shuffle(out)
    return out[:n]


def training_rows(cache_dir: str, *, negatives_per_claim: int = 2, seed: int = 0) -> list[dict]:
    from evidencekd.data.regions import RegionSet
    from evidencekd.facts.build import facts_from_dicts

    rng = random.Random(seed)
    rows = []
    d = os.path.join(cache_dir, "kd_c")
    if not os.path.isdir(d):
        return rows
    for name in sorted(os.listdir(d)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(d, name), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("header") or not r.get("kept"):
                    continue
                ft = facts_from_dicts(r["facts"], RegionSet(r["image_id"], 1, 1, "annotation", ()))
                for c in parse_claims(r["kept"]):
                    if not c.is_observation:
                        continue
                    rows.append({"text": verifier_input(c, ft.facts), "label": SUPPORTED})
                    for nc, st in synthetic_negatives(c, ft.facts, rng, negatives_per_claim):
                        rows.append({"text": verifier_input(nc, ft.facts), "label": st})
    return rows


# ---------------------------------------------------------------------------
def _build(vocab_size: int, d: int):
    import torch.nn as nn
    from evidencekd.models.text_encoder import SimpleTextEncoder

    class V(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = SimpleTextEncoder(vocab_size, d, 2, 8, 0.1, max_len=128)
            self.head = nn.Linear(d, 3)

        def forward(self, ids, mask):
            h, _ = self.enc(ids, mask)
            return self.head(h[:, 0])

    return V()


def train(cfg, *, output_dir: str, epochs: int | None = None, limit: int | None = None) -> str:
    import torch
    from torch.utils.data import DataLoader

    v = cfg.verifier
    rows = training_rows(cfg.paths.cache_dir, negatives_per_claim=int(v.negatives_per_claim))
    if not rows:
        raise SystemExit("[verifier] no KD-C cache — run teacher/kd_c_targets first")
    if limit:
        rows = rows[:limit]
    os.makedirs(output_dir, exist_ok=True)
    tok = SimpleTokenizer.build([r["text"] for r in rows])
    tok.save(os.path.join(output_dir, "vocab.json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build(len(tok), int(v.d)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(v.lr))
    dist = {s: sum(r["label"] == s for r in rows) for s in STATUSES}
    print(f"[verifier] {len(rows)} rows {dist}")

    def collate(batch):
        ids, mask = pad_sequences([tok.encode(b["text"], int(v.max_tokens)) for b in batch])
        return ids, mask, torch.tensor([STATUSES.index(b["label"]) for b in batch])

    loader = DataLoader(rows, batch_size=int(v.batch_size), shuffle=True, collate_fn=collate)
    for epoch in range(int(epochs if epochs is not None else v.epochs)):
        model.train(); tot = 0.0; n = 0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(dev), mask.to(dev), y.to(dev)
            loss = torch.nn.functional.cross_entropy(model(ids, mask), y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); n += 1
        print(f"[verifier] epoch {epoch} loss {tot / max(n, 1):.4f}")
    path = os.path.join(output_dir, "verifier.pt")
    torch.save({"model": model.state_dict(), "vocab": tok.vocab, "d": int(v.d)}, path)
    print(f"[verifier] wrote {path}")
    return path


class LearnedVerifier:
    def __init__(self, path: str):
        import torch
        st = torch.load(path, map_location="cpu")
        self.tok = SimpleTokenizer(st["vocab"])
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build(len(self.tok), int(st["d"])).to(self.dev).eval()
        self.model.load_state_dict(st["model"])
        self.torch = torch

    def statuses(self, claims: Sequence[Claim], facts: Sequence[Fact]) -> list[tuple[str, float]]:
        obs = [c for c in claims if c.is_observation]
        if not obs:
            return []
        ids, mask = pad_sequences([self.tok.encode(verifier_input(c, facts), 96) for c in obs])
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(ids.to(self.dev), mask.to(self.dev)).float(), -1).cpu()
        return [(STATUSES[int(p[i].argmax())], float(p[i].max())) for i in range(len(obs))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verifier v2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/generator.yaml")
    t.add_argument("--output-dir", default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    train(cfg, output_dir=args.output_dir or os.path.join(cfg.paths.output_dir, "verifier"), epochs=args.epochs, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
