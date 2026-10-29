"""MedGemma baselines on *our* split — B4 (fine-tuned) and pretrained.

``vindr_tables`` prints the paper's published numbers (``vindr_cxr_vqa.results.PAPER``)
for reference, but those are val-659 constants: they cannot answer "did this
repository improve on the model it started from?" when the proposed system is
scored on test-300. The spec's exit criterion for stage 1
(``docs/method_spec_v4.1.md`` §"Sau giai đoạn 1") is explicit about it —
*"Vượt B4 về answer & grounding, **cùng chế độ**"*.

This module runs MedGemma over a split and writes predictions in the schema
``vindr_cxr_vqa.evaluate`` grades, so the comparison table gets two more rows
measured exactly like every other row:

    python -m evidencekd.eval.medgemma_baseline --config configs/teacher.yaml \
        --split test --tag b4           # MedGemma + the VinDr LoRA adapter
    python -m evidencekd.eval.medgemma_baseline --config configs/teacher.yaml \
        --split test --tag pretrained   # the same base, no adapter

MedGemma is trained to emit ``<loc_xxx_yyy_xxx_yyy>`` itself, so its reply is
written verbatim — no re-formatting, no fabricated box.

Appends as it goes and skips what a previous run finished, because on Kaggle a
659-sample pass over a 4B model regularly outlives the session.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Iterable, Sequence

from evidencekd.data.vindr_processed import VQASample, load_samples
from evidencekd.eval.vindr_bridge import PredictionRow, row_for
from evidencekd.utils.config import load_config

TAGS = ("b4", "pretrained")


def adapter_for(tag: str, configured: str | None) -> str | None:
    """The adapter directory a tag runs with. ``pretrained`` runs without one."""
    if tag == "pretrained":
        return None
    if not configured:
        raise SystemExit(f"--tag {tag} is the fine-tuned baseline and needs the VinDr LoRA adapter; "
                         "set model.adapter in the teacher config or pass --adapter")
    return configured


def done_sample_ids(path: str) -> set[str]:
    """Sample ids already written to ``path`` (a killed session leaves a partial line)."""
    out: set[str] = set()
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def baseline_rows(samples: Sequence[VQASample], load_image: Callable[[VQASample], object],
                  generate: Callable[[Sequence, Sequence[str]], Sequence[str]], *,
                  batch_size: int = 4, skip_ids: Iterable[str] = (),
                  on_batch: Callable[[list[PredictionRow]], None] | None = None) -> list[PredictionRow]:
    """Prediction rows for ``samples``, the model's reply used verbatim.

    ``load_image`` is called one batch at a time so a split never sits in RAM.
    ``on_batch`` receives each finished batch — the CLI uses it to append, so a
    session that dies keeps everything up to the last batch.
    """
    skip = set(skip_ids)
    todo = [s for s in samples if s.sample_id not in skip]
    rows: list[PredictionRow] = []
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        images = [load_image(s) for s in chunk]
        replies = generate(images, [s.question for s in chunk])
        batch = [row_for(s, (r or "").strip()) for s, r in zip(chunk, replies)]
        rows.extend(batch)
        if on_batch is not None:
            on_batch(batch)
    return rows


def run(cfg, *, split: str, tag: str, adapter: str | None = None, limit: int | None = None,
        batch_size: int = 4, out_path: str | None = None, resume: bool = True) -> str:
    from PIL import Image

    from evidencekd.teacher.medgemma import MedGemmaTeacher

    reports = cfg.paths.get("reports_dir", "reports")
    os.makedirs(reports, exist_ok=True)
    out_path = out_path or os.path.join(reports, f"preds_{split}_{tag}.jsonl")
    samples = load_samples(cfg.paths.processed_dir, split)
    if limit:
        samples = samples[:limit]
    skip = done_sample_ids(out_path) if resume else set()
    if skip:
        print(f"[{tag}] resuming: {len(skip)} of {len(samples)} already done")
    if len(skip) >= len(samples):
        print(f"[{tag}] nothing left to do -> {out_path}")
        return out_path

    use = adapter_for(tag, adapter if adapter is not None else cfg.model.get("adapter"))
    print(f"[{tag}] MedGemma {cfg.model.name} | adapter: {use or '(none — pretrained)'}")
    teacher = MedGemmaTeacher(cfg, adapter=use)
    dataset_dir = cfg.paths.dataset_dir
    n = [0]

    def load_image(s: VQASample):
        return Image.open(os.path.join(dataset_dir, s.image_path)).convert("RGB")

    with open(out_path, "a" if skip else "w", encoding="utf-8") as f:
        def append(batch: list[PredictionRow]) -> None:
            for r in batch:
                f.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
            f.flush()
            n[0] += len(batch)
            print(f"[{tag}] {n[0] + len(skip)}/{len(samples)}", flush=True)

        baseline_rows(samples, load_image, teacher.generate, batch_size=batch_size,
                      skip_ids=skip, on_batch=append)
    print(f"[{tag}] wrote {out_path}")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/teacher.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tag", default="b4", choices=list(TAGS))
    ap.add_argument("--adapter", default=None, help="override model.adapter (ignored for --tag pretrained)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, split=args.split, tag=args.tag, adapter=args.adapter, limit=args.limit,
        batch_size=args.batch_size, out_path=args.out, resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
