"""Generator L — MedGemma + a NEW LoRA (design §6; spec "Proposed-L", the ceiling).

Same inputs as generator S plus the image: a user turn holding the question,
the locked answer and the fact table; the target is the claim grammar from
the KD-C cache. Reuses the sibling repo's loader and LoRA wrapping
(``vindr_cxr_vqa.train.load_model``) with its own adapter directory — never
the teacher's adapter (spec §3: no shared weights).

    python -m evidencekd.models.generator_l train --config configs/generator.yaml --teacher-config configs/teacher.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from evidencekd.models.generator_s import kd_c_rows, template_rows_from_derived
from evidencekd.utils.config import Config, load_config

SYSTEM_L = ("You are an expert radiologist. Explain the given answer to the question using only the "
            "listed facts. Write each observation as <obs facts=F#> sentence [R##] </obs>, citing a region "
            "of that fact. You may end with one background sentence.")


def user_text(row: dict) -> str:
    return row["input"]  # question/answer/facts/regions already serialised by generator_s.build_input


def _vindr_cfg(cfg, tcfg) -> Config:
    g = cfg.generator_l
    return Config({
        "model": {"name": g.model, "attn_implementation": "eager", "torch_dtype": "auto"},
        "lora": dict(g.lora),
        "train": {"gradient_checkpointing": True, "load_in_4bit": g.get("load_in_4bit"),
                  "precision": g.get("precision", "auto"), "epochs": g.epochs, "batch_size": g.batch_size,
                  "grad_accum": g.grad_accum, "lr": g.lr, "weight_decay": 0.01, "warmup_ratio": 0.03,
                  "lr_scheduler": "cosine", "max_grad_norm": 1.0, "logging_steps": 25,
                  "save_strategy": "steps", "save_steps": 100, "save_total_limit": 2, "seed": 42},
        "data": {"max_seq_length": g.max_seq_length, "loc_format": "norm1000"},
        "eval": {"max_new_tokens": g.max_new_tokens}, "generate": {"do_sample": False, "num_beams": 1},
    })


class _Collator:
    def __init__(self, processor, max_len: int):
        from vindr_cxr_vqa.data import _image_token_ids
        self.p = processor; self.max_len = max_len
        self.pad = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        self.image_ids = _image_token_ids(processor)

    def _messages(self, text):
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_L}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

    def __call__(self, rows):
        import torch
        from PIL import Image
        texts, imgs, n_prompt = [], [], []
        for r in rows:
            prompt = self.p.apply_chat_template(self._messages(r["user_text"]), add_generation_prompt=True, tokenize=False)
            img = Image.open(r["image_path"]).convert("RGB")
            p_ids = self.p(text=[prompt], images=[[img]], return_tensors="pt")["input_ids"].shape[1]
            texts.append(prompt + r["target"] + (self.p.tokenizer.eos_token or "")); imgs.append(img); n_prompt.append(p_ids)
        self.p.tokenizer.padding_side = "right"
        enc = self.p(text=texts, images=[[i] for i in imgs], return_tensors="pt", padding=True,
                     truncation=True, max_length=self.max_len)
        labels = enc["input_ids"].clone()
        for i, n in enumerate(n_prompt):
            labels[i, :n] = -100
        labels[enc["input_ids"] == self.pad] = -100
        for t in self.image_ids:
            labels[enc["input_ids"] == t] = -100
        enc["labels"] = labels
        return dict(enc)


def train(cfg, tcfg, *, output_dir: str, limit: int | None = None, max_steps: int = -1, resume: str = "auto") -> str:
    import torch
    from transformers import Trainer

    from vindr_cxr_vqa.infer import load_processor
    from vindr_cxr_vqa.train import build_training_arguments, latest_checkpoint, load_model

    vcfg = _vindr_cfg(cfg, tcfg)
    rows = kd_c_rows(cfg.paths.cache_dir) or template_rows_from_derived(cfg)
    if limit:
        rows = rows[:limit]
    from evidencekd.data.vindr_processed import load_samples
    paths = {s.sample_id: os.path.join(cfg.paths.dataset_dir, s.image_path) for s in load_samples(cfg.paths.processed_dir, "train")}
    data = [{"user_text": user_text(r), "target": r["target"], "image_path": paths[r["sample_id"]]}
            for r in rows if r["sample_id"] in paths]
    print(f"[gen_l] {len(data)} rows")
    os.makedirs(output_dir, exist_ok=True)
    model, used_4bit = load_model(vcfg)
    processor = load_processor(vcfg)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = build_training_arguments(
        output_dir=output_dir, num_train_epochs=float(vcfg.train.epochs), max_steps=max_steps,
        per_device_train_batch_size=int(vcfg.train.batch_size), gradient_accumulation_steps=int(vcfg.train.grad_accum),
        learning_rate=float(vcfg.train.lr), weight_decay=0.01, warmup_ratio=0.03, lr_scheduler_type="cosine",
        max_grad_norm=1.0, optim="paged_adamw_8bit" if used_4bit else "adamw_torch", fp16=not bf16, bf16=bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=25,
        save_strategy="steps", save_steps=100, save_total_limit=2, seed=42, remove_unused_columns=False,
        dataloader_pin_memory=False, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=data,
                      data_collator=_Collator(processor, int(vcfg.data.max_seq_length)))
    ck = latest_checkpoint(output_dir) if resume == "auto" else (resume if resume not in ("none", "no", "") else None)
    trainer.train(resume_from_checkpoint=ck)
    model.save_pretrained(output_dir); processor.save_pretrained(output_dir)
    json.dump(vcfg.to_dict(), open(os.path.join(output_dir, "train_config.json"), "w"), indent=2)
    print(f"[gen_l] adapter -> {output_dir}")
    return output_dir


class GeneratorL:
    def __init__(self, cfg, tcfg, adapter_dir: str):
        from evidencekd.teacher.medgemma import MedGemmaTeacher
        tcfg2 = Config(tcfg.to_dict())
        tcfg2["model"]["name"] = cfg.generator_l.model
        tcfg2["model"]["adapter"] = adapter_dir
        tcfg2["generation"]["max_new_tokens"] = int(cfg.generator_l.max_new_tokens)
        self.inner = MedGemmaTeacher(tcfg2, adapter=adapter_dir)
        self.cfg = cfg

    def generate(self, image, text: str) -> str:
        return self.inner.generate_text(image, text, system=SYSTEM_L, n=1,
                                        max_new_tokens=int(self.cfg.generator_l.max_new_tokens), temperature=0.0)[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generator L (MedGemma + LoRA#2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/generator.yaml")
    t.add_argument("--teacher-config", default="configs/teacher.yaml")
    t.add_argument("--output-dir", default=None)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--resume", default="auto")
    args = ap.parse_args(argv)
    cfg, tcfg = load_config(args.config), load_config(args.teacher_config)
    train(cfg, tcfg, output_dir=args.output_dir or os.path.join(cfg.paths.output_dir, "generator_l"),
          limit=args.limit, max_steps=args.max_steps, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
