"""Stages 1-2 — the decision branch (design §6, §7.1, §7.3, §7.4; spec §4.1, §4.3).

Frozen stage-0 backbone (ResNet-50-FPN) -> ROIAlign region tokens (original
box + x1.5 context) with kind / class / coordinate embeddings and a
whole-image token -> question encoder -> 2 bidirectional cross-attention
layers -> 2-layer evidence encoder -> heads:

    relevance ρ̂_j (which region answers the question)   BCE
    question type (6)                                      CE
    finding multi-label (14, whole-image token)            BCE
    Yes/No (2) · side (3) · count (5)                      CE (ignore -1)
    pointer π_c over finding regions + NO_SUPPORT          -log Σ_{G} π   (+ KD-R KL, + faithfulness KL)
    contrastive alignment L_align                          InfoNCE

Stage 1 trains everything but KD-R/faithfulness; stage 2 continues from the
stage-1 weights with the teacher's cached π^T (``teacher/kd_r_targets``).

    python -m evidencekd.models.decision_branch train   --config configs/decision.yaml --stage 1
    python -m evidencekd.models.decision_branch train   --config configs/decision.yaml --stage 2 --init outputs/decision/decision_stage1.pt
    python -m evidencekd.models.decision_branch predict --config configs/decision.yaml --checkpoint outputs/decision/decision_stage2.pt --split test

Outputs of ``predict``: ``outputs/decision_{split}.jsonl`` — one JSON per
question with the head probabilities, per-region relevance and the pointer
distribution (null slot last), keyed to the region ids of
``derived/regions_{split}.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from typing import Sequence

from evidencekd.data.qa_labels import COUNT_CLASSES, IGNORE, QALabels, QUESTION_TYPES, SIDES
from evidencekd.data.regions import ANATOMY_IDS, KIND_FINDING, RegionSet, load_region_sets
from evidencekd.data.vindr_processed import VQASample, load_images, load_samples, splits_from_images
from evidencekd.labels import VQA_CLASSES
from evidencekd.models.text_encoder import PAD, SimpleTokenizer, pad_sequences
from evidencekd.utils.config import Config, load_config

N_CLASSES = len(VQA_CLASSES)
KIND_IMG, KIND_FIND, KIND_ANAT = 0, 1, 2
CLASS_PAD = N_CLASSES + len(ANATOMY_IDS) + 1   # 14 findings + 4 anatomy + IMG + pad


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_labels(path: str) -> dict[str, QALabels]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["sample_id"]] = QALabels(d["sample_id"], d["question_type"], d["yes_no"], tuple(d["findings"]),
                                           d["side"], d["count"], tuple(d["target_regions"]), d["asked_class"],
                                           bool(d.get("is_negative", False)))
    return out


def load_negatives(path: str) -> list[VQASample]:
    if not os.path.isfile(path):
        return []
    from evidencekd.data.vindr_processed import _sample_from_dict
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(_sample_from_dict(d))
    return out


def load_kd_r_cache(cache_dir: str | None) -> dict[str, dict]:
    """``{sample_id: {"region_ids": [...], "pi_t": [...], "weight": w}}`` from the shards."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return {}
    out = {}
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(cache_dir, name), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if "pi_t" in d:
                    out[d["sample_id"]] = d
    return out


class DecisionDataset:  # torch Dataset duck-type
    def __init__(self, samples: Sequence[VQASample], labels: dict[str, QALabels] | None,
                 region_sets: dict[str, RegionSet], dataset_dir: str, tokenizer, *,
                 max_q_tokens: int = 32, kd_r: dict[str, dict] | None = None, hf_tokenizer=None):
        self.samples = [s for s in samples if s.image_id in region_sets]
        self.labels = labels or {}
        self.region_sets = region_sets
        self.dataset_dir = dataset_dir
        self.tok = tokenizer
        self.hf_tok = hf_tokenizer
        self.max_q = max_q_tokens
        self.kd_r = kd_r or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        s = self.samples[i]
        rs = self.region_sets[s.image_id]
        img = Image.open(os.path.join(self.dataset_dir, s.image_path)).convert("RGB")
        w, h = img.size
        t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(h, w, 3).permute(2, 0, 1).float().div_(255.0)
        fx, fy = w / max(rs.width, 1), h / max(rs.height, 1)
        finding = list(rs.finding_regions)
        anat = list(rs.anatomy_regions)
        regs = finding + anat
        boxes = torch.tensor([[r.box.x_min * fx, r.box.y_min * fy, r.box.x_max * fx, r.box.y_max * fy] for r in regs],
                             dtype=torch.float32).reshape(-1, 4)
        kinds = [KIND_FIND] * len(finding) + [KIND_ANAT] * len(anat)
        classes = [VQA_CLASSES.index(r.label) if r.label in VQA_CLASSES else CLASS_PAD for r in finding] + \
                  [N_CLASSES + ANATOMY_IDS.index(r.label) for r in anat]
        lab = self.labels.get(s.sample_id)
        n_f = len(finding)
        rel = torch.zeros(n_f)
        ptr = torch.zeros(n_f)
        if lab is not None:
            for j, r in enumerate(finding):
                if r.id in lab.target_regions:
                    rel[j] = 1.0
                    ptr[j] = 1.0
        item = {
            "sample_id": s.sample_id, "image_id": s.image_id, "image": t, "boxes": boxes,
            "kinds": torch.tensor(kinds, dtype=torch.long), "classes": torch.tensor(classes, dtype=torch.long),
            "n_finding": n_f, "region_ids": [r.id for r in finding],
            "question": s.question, "q_ids": self.tok.encode(s.question, self.max_q) if self.tok else [],
            "rel": rel, "ptr": ptr,
            "qtype": lab.question_type if lab else IGNORE,
            "yes_no": lab.yes_no if lab else IGNORE,
            "findings": torch.tensor(lab.findings, dtype=torch.float32) if lab else torch.full((N_CLASSES,), -1.0),
            "side": lab.side if lab else IGNORE, "count": lab.count if lab else IGNORE,
            "asked": lab.asked_class if lab else (VQA_CLASSES.index(s.gt_finding) if s.gt_finding in VQA_CLASSES else IGNORE),
            "is_negative": bool(lab.is_negative) if lab else False,
        }
        kd = self.kd_r.get(s.sample_id)
        if kd and kd.get("region_ids") == item["region_ids"] and float(kd.get("weight", 0.0)) > 0:
            item["pi_t"] = torch.tensor(kd["pi_t"], dtype=torch.float32)   # n_f + 1, null last
            item["kd_w"] = float(kd["weight"])
        return item


def collate(batch):
    import torch
    B = len(batch)
    M = max(int(b["boxes"].shape[0]) for b in batch)
    Mf = max(b["n_finding"] for b in batch)
    boxes = torch.zeros(B, M, 4); kinds = torch.zeros(B, M, dtype=torch.long)
    classes = torch.full((B, M), CLASS_PAD, dtype=torch.long)
    rmask = torch.zeros(B, M, dtype=torch.bool); fmask = torch.zeros(B, Mf, dtype=torch.bool)
    rel = torch.zeros(B, Mf); ptr = torch.zeros(B, Mf)
    pi_t = torch.zeros(B, Mf + 1); kd_w = torch.zeros(B)
    for i, b in enumerate(batch):
        m = int(b["boxes"].shape[0]); nf = b["n_finding"]
        boxes[i, :m] = b["boxes"]; kinds[i, :m] = b["kinds"]; classes[i, :m] = b["classes"]
        rmask[i, :m] = True; fmask[i, :nf] = True
        rel[i, :nf] = b["rel"]; ptr[i, :nf] = b["ptr"]
        if "pi_t" in b:
            p = b["pi_t"]
            pi_t[i, :nf] = p[:nf]; pi_t[i, Mf] = p[-1]; kd_w[i] = b["kd_w"]
    q_ids, q_mask = pad_sequences([b["q_ids"] for b in batch]) if batch[0]["q_ids"] else (None, None)
    return {
        "sample_id": [b["sample_id"] for b in batch], "image_id": [b["image_id"] for b in batch],
        "images": [b["image"] for b in batch], "boxes": boxes, "kinds": kinds, "classes": classes,
        "rmask": rmask, "fmask": fmask, "n_finding": [b["n_finding"] for b in batch],
        "region_ids": [b["region_ids"] for b in batch], "questions": [b["question"] for b in batch],
        "q_ids": q_ids, "q_mask": q_mask, "rel": rel, "ptr": ptr, "pi_t": pi_t, "kd_w": kd_w,
        "qtype": torch.tensor([b["qtype"] for b in batch]), "yes_no": torch.tensor([b["yes_no"] for b in batch]),
        "findings": torch.stack([b["findings"] for b in batch]),
        "side": torch.tensor([b["side"] for b in batch]), "count": torch.tensor([b["count"] for b in batch]),
        "asked": torch.tensor([b["asked"] for b in batch]),
        "is_negative": torch.tensor([b["is_negative"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def build_model(cfg, *, vocab_size: int, detector_ckpt: str | None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.ops import MultiScaleRoIAlign

    from evidencekd.models import detector as det_mod
    from evidencekd.models.text_encoder import HFTextEncoder, SimpleTextEncoder

    d = int(cfg.model.d)
    heads = int(cfg.model.heads)
    drop = float(cfg.model.dropout)

    class CrossBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.r2q = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
            self.q2r = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
            self.n1, self.n2, self.n3, self.n4 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
            self.f1 = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
            self.f2 = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))

        def forward(self, r, rmask, q, qmask):
            r = self.n1(r + self.r2q(r, q, q, key_padding_mask=~qmask)[0])
            r = self.n2(r + self.f1(r))
            q = self.n3(q + self.q2r(q, r, r, key_padding_mask=~rmask)[0])
            q = self.n4(q + self.f2(q))
            return r, q

    class DecisionBranch(nn.Module):
        def __init__(self):
            super().__init__()
            # --- frozen stage-0 backbone --------------------------------
            if detector_ckpt and os.path.isfile(detector_ckpt):
                state = torch.load(detector_ckpt, map_location="cpu")
                dcfg = Config(state["config"])
                det, _ = det_mod.build_model(dcfg, coco_weights=None)
                det.load_state_dict(state["model"])
            else:
                dcfg = Config({"model": {"min_size": 512, "max_size": 512}})
                det, _ = det_mod.build_model(dcfg, coco_weights=None)
                print("[decision] WARNING: no detector checkpoint — backbone is random")
            self.backbone = det.backbone
            self.transform = det.transform
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            self.roi = MultiScaleRoIAlign(["0", "1", "2", "3"], int(cfg.model.roi_output), 2)
            k = int(cfg.model.roi_output)
            self.roi_fc = nn.Sequential(nn.Linear(256 * k * k, d), nn.GELU(), nn.Dropout(drop))
            self.ctx_fc = nn.Sequential(nn.Linear(256 * k * k, d), nn.GELU(), nn.Dropout(drop))
            self.merge = nn.Linear(2 * d, d)
            self.img_fc = nn.Linear(256, d)
            self.kind_emb = nn.Embedding(3, d)
            self.class_emb = nn.Embedding(CLASS_PAD + 1, d, padding_idx=CLASS_PAD)
            self.coord_fc = nn.Linear(4, d)
            self.region_norm = nn.LayerNorm(d)
            # --- text -------------------------------------------------------
            if cfg.text_encoder.kind == "hf" and cfg.text_encoder.hf_path:
                self.text = HFTextEncoder(cfg.text_encoder.hf_path, d, bool(cfg.text_encoder.get("freeze_hf", True)))
            else:
                self.text = SimpleTextEncoder(vocab_size, d, 2, heads, drop)
            # --- fusion + evidence ------------------------------------------
            self.fusion = nn.ModuleList([CrossBlock() for _ in range(int(cfg.model.fusion_layers))])
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True)
            self.evidence = nn.TransformerEncoder(layer, int(cfg.model.evidence_layers))
            # --- heads --------------------------------------------------------
            self.rel_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
            self.qtype_head = nn.Linear(d, len(QUESTION_TYPES))
            self.finding_head = nn.Linear(d, N_CLASSES)
            self.yn_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2))
            self.side_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(SIDES)))
            self.count_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, COUNT_CLASSES))
            self.ptr_q = nn.Linear(d, d)
            self.ptr_r = nn.Linear(d, d)
            self.ptr_null = nn.Parameter(torch.zeros(d))
            self.align_q = nn.Linear(d, d)
            self.align_r = nn.Linear(d, d)
            self.d = d

        def train(self, mode: bool = True):
            super().train(mode)
            self.backbone.eval()  # always frozen
            return self

        # ---- region features -------------------------------------------
        def region_tokens(self, images, boxes, kinds, classes, rmask):
            with torch.no_grad():
                il, _ = self.transform(images)
                feats = self.backbone(il.tensors)
            B, M = boxes.shape[:2]
            scales = torch.tensor([il.image_sizes[i][0] / images[i].shape[1] for i in range(B)],
                                  device=boxes.device).view(B, 1, 1)
            b_scaled = boxes * scales
            sizes = torch.tensor([[s[1], s[0]] for s in il.image_sizes], device=boxes.device, dtype=torch.float32)  # (w,h)
            ratio = float(cfg.model.expand_ratio)
            cx, cy = (b_scaled[..., 0] + b_scaled[..., 2]) / 2, (b_scaled[..., 1] + b_scaled[..., 3]) / 2
            bw, bh = (b_scaled[..., 2] - b_scaled[..., 0]) * ratio, (b_scaled[..., 3] - b_scaled[..., 1]) * ratio
            ctx = torch.stack([(cx - bw / 2).clamp(min=0), (cy - bh / 2).clamp(min=0),
                               torch.minimum(cx + bw / 2, sizes[:, None, 0]), torch.minimum(cy + bh / 2, sizes[:, None, 1])], -1)
            box_list = [b_scaled[i][rmask[i]] for i in range(B)]
            ctx_list = [ctx[i][rmask[i]] for i in range(B)]
            fd = {k: feats[k] for k in ("0", "1", "2", "3")}
            r = self.roi(fd, box_list, list(il.image_sizes)).flatten(1)
            c = self.roi(fd, ctx_list, list(il.image_sizes)).flatten(1)
            tok = self.merge(torch.cat([self.roi_fc(r), self.ctx_fc(c)], -1))
            out = torch.zeros(B, M, self.d, device=boxes.device, dtype=tok.dtype)
            out[rmask] = tok
            norm = boxes / sizes[:, None].repeat(1, 1, 2) * scales  # normalised cx-free coords
            out = out + self.kind_emb(kinds) + self.class_emb(classes) + self.coord_fc(norm.float())
            img = self.img_fc(feats["0"].mean((2, 3))) + self.kind_emb.weight[KIND_IMG]
            return self.region_norm(torch.cat([img[:, None], out], 1))   # (B, 1+M, d), IMG first

        # ---- full forward ------------------------------------------------
        def forward(self, batch, *, drop_region=None):
            images = [im.to(self.ptr_null.device) for im in batch["images"]]
            dev = self.ptr_null.device
            boxes, kinds, classes = batch["boxes"].to(dev), batch["kinds"].to(dev), batch["classes"].to(dev)
            rmask = batch["rmask"].to(dev)
            tokens = self.region_tokens(images, boxes, kinds, classes, rmask)
            tmask = torch.cat([torch.ones(rmask.shape[0], 1, dtype=torch.bool, device=dev), rmask], 1)
            if drop_region is not None:                       # (B, 1+M) bool: True = drop
                tmask = tmask & ~drop_region
            if getattr(self.text, "hf", False):
                q_ids, q_mask = self.text.encode_texts(batch["questions"], 48, dev)
            else:
                q_ids, q_mask = batch["q_ids"].to(dev), batch["q_mask"].to(dev)
            q, q_mask = self.text(q_ids, q_mask)
            r = tokens
            for blk in self.fusion:
                r, q = blk(r, tmask, q, q_mask)
            x = torch.cat([r, q], 1)
            xmask = torch.cat([tmask, q_mask], 1)
            h = self.evidence(x, src_key_padding_mask=~xmask)
            hr, hq = h[:, :r.shape[1]], h[:, r.shape[1]:]
            q_pool = (hq * q_mask[..., None]).sum(1) / q_mask.sum(1, keepdim=True).clamp(min=1)
            img_tok = hr[:, 0]
            reg = hr[:, 1:]                                   # (B, M, d)
            Mf = batch["fmask"].shape[1]
            fmask = batch["fmask"].to(dev)
            regf = reg[:, :Mf]
            qexp = q_pool[:, None].expand(-1, Mf, -1)
            rel_logits = self.rel_head(torch.cat([regf, qexp], -1)).squeeze(-1).masked_fill(~fmask, -1e4)
            w = torch.softmax(rel_logits, -1) * fmask
            pooled = (regf * w[..., None]).sum(1)
            qr = torch.cat([q_pool, pooled], -1)
            pq = self.ptr_q(q_pool)
            ptr_logits = (self.ptr_r(regf) * pq[:, None]).sum(-1) / math.sqrt(self.d)
            ptr_logits = ptr_logits.masked_fill(~fmask, -1e4)
            null = (pq * self.ptr_null).sum(-1, keepdim=True) / math.sqrt(self.d)
            return {
                "rel_logits": rel_logits, "ptr_logits": torch.cat([ptr_logits, null], -1),
                "qtype_logits": self.qtype_head(q_pool), "finding_logits": self.finding_head(img_tok),
                "yn_logits": self.yn_head(qr), "side_logits": self.side_head(qr), "count_logits": self.count_head(qr),
                "align_q": F.normalize(self.align_q(q_pool), dim=-1),
                "align_r": F.normalize(self.align_r(regf), dim=-1),
                "fmask": fmask,
            }

    return DecisionBranch()


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def compute_losses(model, out, batch, cfg, *, stage: int) -> dict:
    import torch
    import torch.nn.functional as F

    dev = out["rel_logits"].device
    fmask = out["fmask"]
    rel_t, ptr_t = batch["rel"].to(dev), batch["ptr"].to(dev)
    losses = {}
    # relevance
    valid = fmask & ~batch["is_negative"].to(dev)[:, None]
    if valid.any():
        losses["rel"] = F.binary_cross_entropy_with_logits(out["rel_logits"][valid], rel_t[valid])
    # alignment (InfoNCE over regions with at least one positive)
    tau = float(cfg.loss.align_tau)
    sim = (out["align_r"] * out["align_q"][:, None]).sum(-1) / tau
    sim = sim.masked_fill(~fmask, -1e4)
    has_pos = (ptr_t * fmask).sum(1) > 0
    if has_pos.any():
        lse_all = torch.logsumexp(sim[has_pos], -1)
        lse_pos = torch.logsumexp(sim[has_pos].masked_fill(ptr_t[has_pos] == 0, -1e4), -1)
        losses["align"] = (lse_all - lse_pos).mean()
    # answers
    qt = batch["qtype"].to(dev)
    if (qt != IGNORE).any():
        losses["qtype"] = F.cross_entropy(out["qtype_logits"], qt, ignore_index=IGNORE)
    fl = batch["findings"].to(dev)
    fvalid = fl[:, 0] >= 0
    if fvalid.any():
        losses["finding"] = F.binary_cross_entropy_with_logits(out["finding_logits"][fvalid], fl[fvalid])
    for name, key in (("yes_no", "yn_logits"), ("side", "side_logits"), ("count", "count_logits")):
        t = batch[name].to(dev)
        if (t != IGNORE).any():
            losses[name] = F.cross_entropy(out[key], t, ignore_index=IGNORE)
    # pointer: -log sum_{G} pi ; negatives -> null slot
    logp = torch.log_softmax(out["ptr_logits"], -1)
    neg = batch["is_negative"].to(dev)
    tgt = torch.cat([ptr_t, torch.zeros(ptr_t.shape[0], 1, device=dev)], 1)
    tgt[neg, -1] = 1.0
    hasg = tgt.sum(1) > 0
    if hasg.any():
        losses["pointer"] = -(torch.logsumexp(logp[hasg].masked_fill(tgt[hasg] == 0, -1e4), -1)).mean()
    if stage >= 2:
        # KD-R
        w = batch["kd_w"].to(dev)
        if (w > 0).any():
            from evidencekd.kd.kd_r import kd_r_loss
            pi_t = batch["pi_t"].to(dev)
            losses["kd_r"] = kd_r_loss(pi_t[w > 0], out["ptr_logits"][w > 0], w[w > 0])
        # faithfulness: counterfactual on region tokens of the asked class logit
        asked = batch["asked"].to(dev)
        ok = (asked != IGNORE) & (fmask.sum(1) > 0) & ~neg
        if ok.any():
            with torch.no_grad():
                base = out["finding_logits"].gather(1, asked.clamp(min=0)[:, None]).squeeze(1)
            B, Mf = fmask.shape
            drops = torch.zeros(B, Mf, device=dev)
            for j in range(Mf):
                dmask = torch.zeros(B, 1 + batch["rmask"].shape[1], dtype=torch.bool, device=dev)
                dmask[:, 1 + j] = fmask[:, j]
                with torch.no_grad():
                    o2 = model(batch, drop_region=dmask)
                    masked = o2["finding_logits"].gather(1, asked.clamp(min=0)[:, None]).squeeze(1)
                drops[:, j] = (F.logsigmoid(base) - F.logsigmoid(masked)).clamp(min=0) * fmask[:, j]
            cf = torch.cat([drops, torch.zeros(B, 1, device=dev)], 1)
            cf = cf.masked_fill(torch.cat([~fmask, torch.zeros(B, 1, dtype=torch.bool, device=dev)], 1), -1e4)
            pi_cf = torch.softmax(cf, -1)
            losses["faith"] = (pi_cf[ok] * (torch.log(pi_cf[ok].clamp_min(1e-12)) - logp[ok])).sum(-1).mean()
    weights = {"rel": cfg.loss.lambda_rel, "align": cfg.loss.lambda_align, "qtype": cfg.loss.lambda_answer,
               "finding": cfg.loss.lambda_answer, "yes_no": cfg.loss.lambda_answer, "side": cfg.loss.lambda_answer,
               "count": cfg.loss.lambda_answer, "pointer": cfg.loss.lambda_pointer,
               "kd_r": cfg.loss.lambda_kd_r, "faith": cfg.loss.lambda_faith}
    total = sum(float(weights[k]) * v for k, v in losses.items())
    losses["total"] = total
    return losses


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------
def _datasets(cfg, *, split: str, tokenizer, labels, kd_r=None, include_negatives=False):
    derived = cfg.paths.derived_dir
    samples = load_samples(cfg.paths.processed_dir, split)
    if include_negatives:
        samples = samples + load_negatives(os.path.join(derived, "negatives.jsonl"))
    rs_path = os.path.join(derived, f"regions_{split}.json")
    if split == "train" or not os.path.isfile(rs_path):
        rs_path = os.path.join(derived, "regions_train.json" if split == "train" else f"regions_{split}_annotation.json")
    region_sets = load_region_sets(rs_path)
    return DecisionDataset(samples, labels, region_sets, cfg.paths.dataset_dir, tokenizer,
                           max_q_tokens=int(cfg.data.max_question_tokens), kd_r=kd_r)


def train(cfg, *, stage: int, output_dir: str, init: str | None, resume: str | None,
          epochs: int | None, max_steps: int = -1, limit: int | None = None) -> str:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(int(cfg.train.seed)); random.seed(int(cfg.train.seed))
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    derived = cfg.paths.derived_dir
    labels = load_labels(os.path.join(derived, "labels_train.jsonl"))
    train_samples = load_samples(cfg.paths.processed_dir, "train")
    negs = load_negatives(os.path.join(derived, "negatives.jsonl")) if cfg.data.get("negatives", True) else []
    vocab_path = os.path.join(output_dir, "vocab.json")
    if os.path.isfile(vocab_path):
        tok = SimpleTokenizer.load(vocab_path)
    else:
        tok = SimpleTokenizer.build([s.question for s in train_samples + negs])
        tok.save(vocab_path)
    kd_r = load_kd_r_cache(cfg.paths.get("kd_r_cache")) if stage >= 2 else {}
    if stage >= 2:
        print(f"[decision] KD-R cache: {len(kd_r)} samples with a target")
    ds = _datasets(cfg, split="train", tokenizer=tok, labels=labels, kd_r=kd_r, include_negatives=bool(negs))
    if limit:
        ds.samples = ds.samples[:limit]
    print(f"[decision] stage {stage}: {len(ds)} training questions ({len(negs)} synthetic negatives)")
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=True,
                        num_workers=int(cfg.train.num_workers), collate_fn=collate, drop_last=True)

    model = build_model(cfg, vocab_size=len(tok), detector_ckpt=cfg.paths.get("detector_checkpoint")).to(dev)
    if init and os.path.isfile(init):
        st = torch.load(init, map_location="cpu")
        model.load_state_dict(st["model"], strict=False)
        print(f"[decision] initialised from {init}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    n_epochs = int(epochs if epochs is not None else (cfg.train.epochs_stage1 if stage == 1 else cfg.train.epochs_stage2))
    total_steps = n_epochs * len(loader)
    warm = int(cfg.train.warmup_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * max(0.0, 1 - s / max(total_steps, 1)))
    use_amp = bool(cfg.train.get("amp", True)) and dev.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, step = 0, 0
    ck = None
    if resume == "auto":
        cks = sorted((f for f in os.listdir(output_dir) if re.fullmatch(rf"checkpoint-s{stage}-\d+\.pt", f)),
                     key=lambda f: int(f.split("-")[-1][:-3]))
        ck = os.path.join(output_dir, cks[-1]) if cks else None
    elif resume and resume not in ("none", "no", "false"):
        ck = resume
    if ck:
        st = torch.load(ck, map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"])
        start_epoch, step = int(st["epoch"]) + 1, int(st["step"])
        print(f"[decision] resumed from {ck}")

    log_path = os.path.join(output_dir, f"train_log_stage{stage}.jsonl")
    t0 = time.time(); done = False
    epoch = start_epoch - 1
    for epoch in range(start_epoch, n_epochs):
        model.train()
        agg: dict[str, float] = {}; n = 0
        for batch in loader:
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                out = model(batch)
                losses = compute_losses(model, out, batch, cfg, stage=stage)
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {losses}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1; n += 1
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            if step % int(cfg.train.log_every) == 0:
                rec = {"stage": stage, "epoch": epoch, "step": step, "elapsed_s": round(time.time() - t0, 1),
                       **{k: round(v / n, 4) for k, v in agg.items()}}
                print(json.dumps(rec))
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                agg, n = {}, 0
            if 0 < max_steps <= step:
                done = True; break
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "step": step, "stage": stage, "config": cfg.to_dict()},
                   os.path.join(output_dir, f"checkpoint-s{stage}-{epoch}.pt"))
        print(f"[decision] saved checkpoint-s{stage}-{epoch}.pt")
        if done:
            break
    final = os.path.join(output_dir, f"decision_stage{stage}.pt")
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "stage": stage, "vocab": tok.vocab}, final)
    print(f"[decision] wrote {final}")
    return final


def load_for_inference(cfg, checkpoint: str):
    import torch
    st = torch.load(checkpoint, map_location="cpu")
    tok = SimpleTokenizer(st["vocab"])
    model = build_model(cfg, vocab_size=len(tok), detector_ckpt=cfg.paths.get("detector_checkpoint"))
    model.load_state_dict(st["model"], strict=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(dev).eval(), tok, dev


def predict(cfg, *, checkpoint: str, split: str, out_path: str, batch_size: int = 16,
            regions_path: str | None = None, limit: int | None = None) -> str:
    import torch
    from torch.utils.data import DataLoader

    model, tok, dev = load_for_inference(cfg, checkpoint)
    samples = load_samples(cfg.paths.processed_dir, split)
    if limit:
        samples = samples[:limit]
    rs_path = regions_path or os.path.join(cfg.paths.derived_dir, f"regions_{split}.json")
    region_sets = load_region_sets(rs_path)
    ds = DecisionDataset(samples, None, region_sets, cfg.paths.dataset_dir, tok,
                         max_q_tokens=int(cfg.data.max_question_tokens))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=int(cfg.train.get("num_workers", 2)),
                        collate_fn=collate)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n = 0; t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f, torch.no_grad():
        for batch in loader:
            out = model(batch)
            B = len(batch["sample_id"])
            rel = torch.sigmoid(out["rel_logits"]).cpu()
            ptr = torch.softmax(out["ptr_logits"].float(), -1).cpu()
            qtype = torch.softmax(out["qtype_logits"].float(), -1).cpu()
            findings = torch.sigmoid(out["finding_logits"].float()).cpu()
            yn = torch.softmax(out["yn_logits"].float(), -1).cpu()
            side = torch.softmax(out["side_logits"].float(), -1).cpu()
            count = torch.softmax(out["count_logits"].float(), -1).cpu()
            for i in range(B):
                nf = batch["n_finding"][i]
                row = {"sample_id": batch["sample_id"][i], "image_id": batch["image_id"][i],
                       "region_ids": batch["region_ids"][i], "regions_source": rs_path,
                       "question_type": QUESTION_TYPES[int(qtype[i].argmax())],
                       "qtype_probs": [round(float(x), 4) for x in qtype[i]],
                       "finding_probs": {c: round(float(findings[i][j]), 4) for j, c in enumerate(VQA_CLASSES)},
                       "yes_prob": round(float(yn[i][1]), 4),
                       "side_probs": {s: round(float(side[i][j]), 4) for j, s in enumerate(SIDES)},
                       "count_probs": [round(float(x), 4) for x in count[i]],
                       "relevance": [round(float(x), 4) for x in rel[i][:nf]],
                       "pointer": [round(float(x), 4) for x in ptr[i][:nf]] + [round(float(ptr[i][-1]), 4)]}
                f.write(json.dumps(row) + "\n"); n += 1
    print(f"[decision] {n} questions -> {out_path} ({(time.time() - t0) / max(n, 1):.3f} s/question)")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Decision branch (stages 1-2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/decision.yaml")
    t.add_argument("--stage", type=int, default=1, choices=[1, 2])
    t.add_argument("--output-dir", default=None)
    t.add_argument("--init", default=None)
    t.add_argument("--resume", default="auto")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("predict")
    p.add_argument("--config", default="configs/decision.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--regions", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "train":
        out_dir = args.output_dir or os.path.join(cfg.paths.output_dir, "decision")
        train(cfg, stage=args.stage, output_dir=out_dir, init=args.init, resume=args.resume,
              epochs=args.epochs, max_steps=args.max_steps, limit=args.limit)
    else:
        out = args.out or os.path.join(cfg.paths.output_dir, f"decision_{args.split}.jsonl")
        predict(cfg, checkpoint=args.checkpoint, split=args.split, out_path=out,
                batch_size=args.batch_size, regions_path=args.regions, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
