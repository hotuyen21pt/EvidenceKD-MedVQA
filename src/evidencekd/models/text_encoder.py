"""Question encoders for the decision branch (design §6).

* :class:`SimpleTokenizer` / :class:`SimpleTextEncoder` — word vocabulary
  built from the training questions, 2 Transformer layers, trained from
  scratch. The questions are templated (6 types x 14 findings), so this is
  enough to carry type and finding; it needs no download, which matters on
  an offline Kaggle session.
* :class:`HFTextEncoder` — any HuggingFace encoder directory (CXR-BERT),
  frozen, projected to ``d``.

The tokenizer half is pure Python; the modules import torch lazily.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable, Sequence

PAD, CLS, UNK = 0, 1, 2
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)?")


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    @classmethod
    def build(cls, texts: Iterable[str], *, min_count: int = 1) -> "SimpleTokenizer":
        counts: dict[str, int] = {}
        for t in texts:
            for w in cls.tokenize(t):
                counts[w] = counts.get(w, 0) + 1
        vocab = {"<pad>": PAD, "<cls>": CLS, "<unk>": UNK}
        for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if c >= min_count:
                vocab[w] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, max_len: int = 32) -> list[int]:
        ids = [CLS] + [self.vocab.get(w, UNK) for w in self.tokenize(text)]
        return ids[:max_len]

    def __len__(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def pad_sequences(seqs: Sequence[Sequence[int]], pad: int = PAD):
    import torch
    T = max(len(s) for s in seqs) if seqs else 1
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = True
    return ids, mask


# ---------------------------------------------------------------------------
def _nn():
    import torch
    import torch.nn as nn
    return torch, nn


class SimpleTextEncoder:  # nn.Module built lazily
    def __new__(cls, vocab_size: int, d: int = 256, layers: int = 2, heads: int = 8, dropout: float = 0.1, max_len: int = 64):
        torch, nn = _nn()

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
                self.pos = nn.Embedding(max_len, d)
                layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True)
                self.enc = nn.TransformerEncoder(layer, layers)
                self.norm = nn.LayerNorm(d)
                self.out_dim = d

            def forward(self, ids, mask):
                pos = torch.arange(ids.shape[1], device=ids.device)[None]
                x = self.emb(ids) + self.pos(pos)
                x = self.enc(x, src_key_padding_mask=~mask)
                return self.norm(x), mask

        return _Enc()


class HFTextEncoder:
    def __new__(cls, path: str, d: int = 256, freeze: bool = True):
        torch, nn = _nn()
        from transformers import AutoModel, AutoTokenizer

        class _Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok = AutoTokenizer.from_pretrained(path)
                self.model = AutoModel.from_pretrained(path)
                if freeze:
                    for p in self.model.parameters():
                        p.requires_grad = False
                self.proj = nn.Linear(self.model.config.hidden_size, d)
                self.norm = nn.LayerNorm(d)
                self.out_dim = d
                self.hf = True

            def encode_texts(self, texts: Sequence[str], max_len: int, device):
                b = self.tok(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                return b["input_ids"].to(device), b["attention_mask"].bool().to(device)

            def forward(self, ids, mask):
                out = self.model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
                return self.norm(self.proj(out)), mask

        return _Enc()
