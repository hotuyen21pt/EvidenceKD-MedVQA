"""Generator S — flan-T5 over text (design §6, §7.5; spec §4.3 "Generator S").

Input:  ``question: ... answer: ... facts: F1: presence Cardiomegaly [R01]; ... regions: R01 R02 A_thorax``
Target: the claim grammar, e.g. ``<obs facts=F1> The heart is enlarged [R01] </obs> Background sentence.``

Targets come from the KD-C cache (filtered teacher claims, ``targets: kd_c``)
or from the B9 templates (``targets: template`` — the "S không KD" row of
Table E). Decoding is constrained (``claims.constrain``) and the pipeline
applies ``enforce_citations`` afterwards regardless.

    python -m evidencekd.models.generator_s train --config configs/generator.yaml
    python -m evidencekd.models.generator_s train --config configs/generator.yaml --targets template
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

from evidencekd.claims.parser import render_claims
from evidencekd.inference.templates import template_claims_full
from evidencekd.utils.config import load_config


def build_input(question: str, answer: str, facts_text: str, region_ids: Sequence[str]) -> str:
    return f"question: {question} answer: {answer} facts: {facts_text} regions: {' '.join(region_ids)}"


def kd_c_rows(cache_dir: str, *, targets: str = "kd_c") -> list[dict]:
    """Training rows from ``cache/kd_c/shard-*.jsonl``."""
    from evidencekd.data.regions import RegionSet
    from evidencekd.facts.build import facts_from_dicts, serialize_facts

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
                if r.get("header"):
                    continue
                fake = RegionSet(r["image_id"], 1, 1, "annotation", ())
                ft = facts_from_dicts(r["facts"], fake)
                ft.region_ids = {rid: None for rid in r["region_ids"]}  # type: ignore[assignment]
                facts_text = serialize_facts(ft)
                if targets == "template":
                    target = render_claims(template_claims_full(ft, background=r.get("background", [])))
                elif targets == "kd_c_unfiltered":
                    target = " ".join(r["raw_rationales"][:1])
                else:
                    if r.get("n_kept_observations", 0) == 0:
                        continue
                    target = r["kept"]
                    if r.get("background"):
                        target = target + " " + r["background"][0]
                rows.append({"sample_id": r["sample_id"], "input": build_input(r["question"], r["answer"], facts_text, r["region_ids"]),
                             "target": target, "facts": r["facts"], "region_ids": r["region_ids"]})
    return rows


def template_rows_from_derived(cfg) -> list[dict]:
    """B9 targets without any teacher run: facts from the annotation regions."""
    from evidencekd.data.anatomy import anatomy_for, load_anatomy
    from evidencekd.data.regions import load_region_sets
    from evidencekd.data.vindr_processed import load_images, load_samples
    from evidencekd.facts.build import build_facts, facts_to_dict, load_background, serialize_facts
    from evidencekd.inference.templates import locked_answer_text

    derived = cfg.paths.derived_dir
    images = load_images(cfg.paths.processed_dir)
    rs_all = load_region_sets(os.path.join(derived, "regions_train.json"))
    anat = load_anatomy(os.path.join(derived, "anatomy.json"))
    bg = load_background(os.path.join(derived, "background.json"))
    rows = []
    for s in load_samples(cfg.paths.processed_dir, "train"):
        rec = images[s.image_id]; rs = rs_all[s.image_id]
        ft = build_facts(rs, anatomy_for(anat, s.image_id, rec.width, rec.height),
                         present_classes=list(rec.vqa_classes_present) or [s.gt_finding])
        answer = locked_answer_text(s.type, finding=s.gt_finding, present=True, findings_present=list(rec.vqa_classes_present))
        rows.append({"sample_id": s.sample_id, "input": build_input(s.question, answer, serialize_facts(ft), rs.ids),
                     "target": render_claims(template_claims_full(ft, background=bg.get(s.gt_finding, [])[:1])),
                     "facts": facts_to_dict(ft), "region_ids": list(rs.ids)})
    return rows


# ---------------------------------------------------------------------------
def train(cfg, *, output_dir: str, targets: str | None = None, resume: bool = True,
          epochs: int | None = None, limit: int | None = None) -> str:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    g = cfg.generator_s
    targets = targets or g.get("targets", "kd_c")
    rows = kd_c_rows(cfg.paths.cache_dir, targets=targets) if targets != "template_derived" else []
    if not rows:
        print(f"[gen_s] no KD-C cache rows for targets={targets}; falling back to template targets from derived/")
        rows = template_rows_from_derived(cfg)
        targets = "template_derived"
    if limit:
        rows = rows[:limit]
    print(f"[gen_s] {len(rows)} training rows (targets={targets})")
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(g.model)
    model = T5ForConditionalGeneration.from_pretrained(g.model).to(dev)
    ck = os.path.join(output_dir, "last")
    start_epoch = 0
    if resume and os.path.isdir(ck):
        model = T5ForConditionalGeneration.from_pretrained(ck).to(dev)
        meta = json.load(open(os.path.join(ck, "meta.json")))
        start_epoch = int(meta["epoch"]) + 1
        print(f"[gen_s] resumed from {ck} (epoch {start_epoch})")

    def collate(batch):
        enc = tok([b["input"] for b in batch], max_length=int(g.max_input_tokens), truncation=True, padding=True, return_tensors="pt")
        lab = tok([b["target"] for b in batch], max_length=int(g.max_target_tokens), truncation=True, padding=True, return_tensors="pt")["input_ids"]
        lab[lab == tok.pad_token_id] = -100
        return {**enc, "labels": lab}

    loader = DataLoader(rows, batch_size=int(g.batch_size), shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=float(g.lr), weight_decay=float(g.weight_decay))
    n_epochs = int(epochs if epochs is not None else g.epochs)
    total = n_epochs * len(loader); warm = int(g.warmup_steps); step = start_epoch * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * max(0.0, 1 - s / max(total, 1)))
    for _ in range(step):
        sched.step()
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    t0 = time.time()
    for epoch in range(start_epoch, n_epochs):
        model.train(); run = 0.0; n = 0
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1; run += float(loss); n += 1
            if step % 50 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": round(run / n, 4), "elapsed_s": round(time.time() - t0)}))
                run = 0.0; n = 0
        model.save_pretrained(ck); tok.save_pretrained(ck)
        json.dump({"epoch": epoch, "targets": targets}, open(os.path.join(ck, "meta.json"), "w"))
        print(f"[gen_s] epoch {epoch} saved -> {ck}")
    final = os.path.join(output_dir, "final")
    model.save_pretrained(final); tok.save_pretrained(final)
    json.dump({"targets": targets, "rows": len(rows)}, open(os.path.join(final, "meta.json"), "w"))
    print(f"[gen_s] wrote {final}")
    return final


class GeneratorS:
    def __init__(self, cfg, model_dir: str):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.dev).eval()

    def generate(self, inputs: Sequence[str], fact_tables, known_region_ids) -> list[str]:
        import torch
        from transformers import LogitsProcessorList

        from evidencekd.claims.constrain import CitationLogitsProcessor
        g = self.cfg.generator_s
        enc = self.tok(list(inputs), max_length=int(g.max_input_tokens), truncation=True, padding=True, return_tensors="pt").to(self.dev)
        procs = LogitsProcessorList()
        if bool(g.get("constrained_decoding", True)):
            procs.append(CitationLogitsProcessor(self.tok, fact_tables, known_region_ids))
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=int(g.max_new_tokens), num_beams=int(g.get("num_beams", 1)),
                                      logits_processor=procs)
        return [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generator S (flan-T5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/generator.yaml")
    t.add_argument("--output-dir", default=None)
    t.add_argument("--targets", default=None, choices=[None, "kd_c", "template", "kd_c_unfiltered", "template_derived"])
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = args.output_dir or os.path.join(cfg.paths.output_dir, "generator_s")
    train(cfg, output_dir=out, targets=args.targets, resume=not args.no_resume, epochs=args.epochs, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
