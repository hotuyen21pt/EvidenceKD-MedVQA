"""Stage 0 — Faster R-CNN R50-FPN on VinDr-CXR (design §6, §7.1).

torchvision only: no custom CUDA kernels, so it builds offline on Kaggle.
The 224 px PNG is upsampled by the model's own transform to ``min_size``;
predictions come back in 224 px and are scaled to original DICOM pixels with
the per-image resize factors before anything downstream sees them.

    python -m evidencekd.models.detector train   --config configs/detector.yaml --output-dir outputs/detector --epochs 8 --resume auto
    python -m evidencekd.models.detector predict --config configs/detector.yaml --checkpoint outputs/detector/detector.pt --split val

Checkpoints: ``checkpoint-{epoch}.pt`` after every epoch (model + optimizer +
scheduler + epoch), ``detector.pt`` at the end. ``--resume auto`` picks the
newest checkpoint, so a 12 h Kaggle session can be continued.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

from evidencekd.data.detection_dataset import (DetSample, VinDrDetectionDataset,
                                               build_index, collate, summarize)
from evidencekd.data.vindr_processed import load_images, splits_from_images
from evidencekd.labels import VQA_CLASSES
from evidencekd.utils.config import load_config

NUM_CLASSES = len(VQA_CLASSES) + 1  # + background


# ---------------------------------------------------------------------------
def build_model(cfg, *, coco_weights: str | None = None):
    import torch
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    arch = cfg.model.get("arch", "fasterrcnn_resnet50_fpn")
    ctor = getattr(torchvision.models.detection, arch)
    model = ctor(
        weights=None, weights_backbone=None,
        min_size=int(cfg.model.min_size), max_size=int(cfg.model.max_size),
        box_detections_per_img=int(cfg.model.get("box_detections_per_img", 50)),
        box_score_thresh=float(cfg.model.get("box_score_thresh", 0.05)),
        box_nms_thresh=float(cfg.model.get("box_nms_thresh", 0.5)),
        trainable_backbone_layers=int(cfg.model.get("trainable_backbone_layers", 3)),
    )
    loaded = "scratch"
    if coco_weights and os.path.isfile(coco_weights):
        state = torch.load(coco_weights, map_location="cpu")
        if isinstance(state, dict) and "model" in state and "state_dict" not in state:
            state = state["model"]
        # drop the 91-class COCO heads; everything else matches
        state = {k: v for k, v in state.items() if not k.startswith("roi_heads.box_predictor")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = f"coco:{os.path.basename(coco_weights)} (missing {len(missing)}, unexpected {len(unexpected)})"
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model, loaded


def latest_checkpoint(out_dir: str) -> str | None:
    if not os.path.isdir(out_dir):
        return None
    best, best_n = None, -1
    for name in os.listdir(out_dir):
        m = re.fullmatch(r"checkpoint-(\d+)\.pt", name)
        if m and int(m.group(1)) > best_n:
            best, best_n = os.path.join(out_dir, name), int(m.group(1))
    return best


def _resolve_resume(arg: str | None, out_dir: str) -> str | None:
    if arg in (None, "", "none", "no", "false"):
        return None
    if arg == "auto":
        return latest_checkpoint(out_dir)
    return arg


# ---------------------------------------------------------------------------
def train(cfg, *, output_dir: str, epochs: int | None, resume: str | None,
          max_steps: int = -1, limit: int | None = None) -> str:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(int(cfg.train.seed))
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = load_images(cfg.paths.processed_dir)
    splits = splits_from_images(images)
    samples = build_index(
        cfg.paths.dataset_dir, splits=splits,
        image_source=cfg.data.image_source, fuse_iou=float(cfg.data.fuse_iou),
        min_votes=int(cfg.data.min_votes), normal_ratio=float(cfg.data.normal_ratio),
        seed=int(cfg.train.seed))
    if limit:
        samples = samples[:limit]
    summary = summarize(samples)
    print(f"[train] {summary['n_images']} images, {summary['n_with_boxes']} with boxes, "
          f"{summary['n_boxes']} boxes; held out {len(splits.val)} val + {len(splits.test)} test")
    with open(os.path.join(output_dir, "train_index_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    ds = VinDrDetectionDataset(samples, train=True)
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=True,
                        num_workers=int(cfg.train.num_workers), collate_fn=collate,
                        drop_last=False, pin_memory=device.type == "cuda")

    model, loaded = build_model(cfg, coco_weights=cfg.model.get("coco_weights"))
    print(f"[train] init weights: {loaded}")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=float(cfg.train.lr), momentum=float(cfg.train.momentum),
                          weight_decay=float(cfg.train.weight_decay))
    n_epochs = int(epochs if epochs is not None else cfg.train.epochs)
    steps_per_epoch = len(loader)
    milestones = [int(m) * steps_per_epoch for m in cfg.train.get("lr_milestones", [])]
    warmup = int(cfg.train.get("warmup_iters", 0))

    def lr_at(step: int) -> float:
        f = 1.0
        if warmup and step < warmup:
            f = 0.001 + (1 - 0.001) * step / warmup
        for m in milestones:
            if step >= m:
                f *= 0.1
        return f

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)   # torch >= 2.3
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, global_step = 0, 0
    ck = _resolve_resume(resume, output_dir)
    if ck:
        state = torch.load(ck, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", start_epoch * steps_per_epoch))
        print(f"[train] resumed from {ck} (next epoch {start_epoch}, step {global_step})")
    elif resume == "auto":
        print("[train] no checkpoint to resume from; starting fresh")

    log_path = os.path.join(output_dir, "train_log.jsonl")
    log_every = int(cfg.train.get("log_every", 50))
    t0 = time.time()
    done = False
    epoch = start_epoch - 1  # stays so when every epoch is already done
    for epoch in range(start_epoch, n_epochs):
        model.train()
        running, n_run = 0.0, 0
        for it, (imgs, targets) in enumerate(loader):
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at step {global_step}: "
                                   f"{ {k: float(v) for k, v in losses.items()} }")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            global_step += 1
            running += loss.item(); n_run += 1
            if global_step % log_every == 0:
                rec = {"epoch": epoch, "step": global_step, "loss": running / n_run,
                       "lr": opt.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1),
                       **{k: round(float(v), 4) for k, v in losses.items()}}
                print(json.dumps(rec)); running, n_run = 0.0, 0
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if 0 < max_steps <= global_step:
                done = True
                break
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "epoch": epoch,
                 "global_step": global_step, "config": cfg.to_dict(),
                 "classes": list(VQA_CLASSES)}
        ck_path = os.path.join(output_dir, f"checkpoint-{epoch}.pt")
        torch.save(state, ck_path)
        print(f"[train] saved {ck_path}")
        if done:
            break

    final = os.path.join(output_dir, "detector.pt")
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "classes": list(VQA_CLASSES), "epochs_completed": epoch + 1},
               final)
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"[train] wrote {final}")
    return final


# ---------------------------------------------------------------------------
def load_for_inference(cfg, checkpoint: str, device=None):
    import torch
    model, _ = build_model(cfg, coco_weights=None)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device


def predict(cfg, *, checkpoint: str, split: str, out_path: str,
            batch_size: int = 8, limit: int | None = None) -> str:
    """Detections for every image of a split, in original pixel space.

    Output JSON: ``{"split", "checkpoint", "score_threshold", "detections":
    {image_id: [{class_name, score, bbox_orig, bbox_224}]}}``.
    """
    import torch
    from torch.utils.data import DataLoader

    images = load_images(cfg.paths.processed_dir)
    splits = splits_from_images(images)
    ids = list(splits.of(split))
    if limit:
        ids = ids[:limit]
    samples = build_index(cfg.paths.dataset_dir, splits=None, image_ids=ids)
    model, device = load_for_inference(cfg, checkpoint)
    ds = VinDrDetectionDataset(samples, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=int(cfg.train.get("num_workers", 2)), collate_fn=collate)
    thr = float(cfg.eval.get("score_threshold", 0.0))
    keep_thr = min(thr, float(cfg.model.get("box_score_thresh", 0.05)))
    out: dict[str, list[dict]] = {}
    t0 = time.time()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            preds = model(imgs)
            for p, t in zip(preds, targets):
                s: DetSample = samples[int(t["image_id"])]
                rows = []
                for box, lab, sc in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
                    if sc < keep_thr:
                        continue
                    x0, y0, x1, y1 = box
                    rows.append({
                        "class_name": VQA_CLASSES[int(lab) - 1],
                        "score": round(float(sc), 4),
                        "bbox_224": [round(v, 2) for v in box],
                        "bbox_orig": [round(x0 / s.fx, 1), round(y0 / s.fy, 1),
                                      round(x1 / s.fx, 1), round(y1 / s.fy, 1)],
                    })
                out[s.image_id] = rows
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"split": split, "checkpoint": checkpoint, "score_threshold": keep_thr,
                   "n_images": len(out), "latency_s_per_image": round((time.time() - t0) / max(len(out), 1), 4),
                   "detections": out}, f)
    print(f"[predict] {len(out)} images -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0 detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/detector.yaml")
    t.add_argument("--output-dir", default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--limit", type=int, default=None, help="images (smoke test)")
    t.add_argument("--resume", default="auto")
    p = sub.add_parser("predict")
    p.add_argument("--config", default="configs/detector.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "train":
        out_dir = args.output_dir or os.path.join(cfg.paths.output_dir, "detector")
        train(cfg, output_dir=out_dir, epochs=args.epochs, resume=args.resume,
              max_steps=args.max_steps, limit=args.limit)
    else:
        out = args.out or os.path.join(cfg.paths.output_dir, f"detections_{args.split}.json")
        predict(cfg, checkpoint=args.checkpoint, split=args.split, out_path=out,
                batch_size=args.batch_size, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
