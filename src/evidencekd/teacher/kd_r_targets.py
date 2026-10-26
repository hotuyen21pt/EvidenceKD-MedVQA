"""KD-R target generation (spec §4.4, design §7.4) — sharded, resumable.

For every training question: ``log p_T(a*|I,Q)`` on the original image and on
the image with each candidate finding region hidden; relevance
``r_i = max(0, drop)``; ``π^T = softmax([r_1..r_M, r_∅]/τ)``; the teacher's free
answer (correctness for the gate); the raw confidence ``exp(mean logp)``; the
mass on the target region. The gate weight itself is computed at training
time from these fields (``--calibrate`` fits the temperature on the tune
split), so the cache never has to be regenerated to change ``δ``.

    python -m evidencekd.teacher.kd_r_targets --config configs/teacher.yaml --shard 0 --n-shards 30
    python -m evidencekd.teacher.kd_r_targets --config configs/teacher.yaml --gate   # -> weights into the shards

Cache: ``cache/kd_r/shard-XXXX.jsonl``; a line per sample; a sample already
present in its shard is skipped on resume. Key fields (configs/kd_r.yaml) are
recorded in the shard header line.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

from evidencekd.data.regions import load_region_sets
from evidencekd.data.vindr_processed import load_images, load_samples, splits_from_images
from evidencekd.kd.kd_r import counterfactual_relevance, gate_weight, target_mass, teacher_distribution
from evidencekd.models.decision_branch import load_labels
from evidencekd.teacher.prompts import canonical_answer, teacher_named_finding
from evidencekd.utils.config import load_config


def shard_path(cache_dir: str, shard: int) -> str:
    return os.path.join(cache_dir, f"shard-{shard:04d}.jsonl")


def _existing_ids(path: str) -> set[str]:
    if not os.path.isfile(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "sample_id" in d:
                out.add(d["sample_id"])
    return out


def select_samples(cfg, *, shard: int, n_shards: int):
    samples = load_samples(cfg.paths.processed_dir, "train")
    frac = float(cfg.kd_r.get("subset_fraction", 1.0))
    if frac < 1.0:
        images = sorted({s.image_id for s in samples})
        random.Random(42).shuffle(images)
        keep = set(images[: int(round(len(images) * frac))])
        samples = [s for s in samples if s.image_id in keep]
    samples.sort(key=lambda s: s.sample_id)
    return samples[shard::n_shards]


def generate_shard(cfg, *, shard: int, n_shards: int, limit: int | None = None, adapter: str | None = None) -> str:
    from PIL import Image

    from evidencekd.teacher.medgemma import MedGemmaTeacher, mask_region

    cache_dir = os.path.join(cfg.paths.cache_dir, "kd_r")
    os.makedirs(cache_dir, exist_ok=True)
    path = shard_path(cache_dir, shard)
    done = _existing_ids(path)
    samples = select_samples(cfg, shard=shard, n_shards=n_shards)
    if limit:
        samples = samples[:limit]
    todo = [s for s in samples if s.sample_id not in done]
    print(f"[kd_r] shard {shard}/{n_shards}: {len(samples)} samples, {len(done)} done, {len(todo)} to do")
    if not todo:
        return path

    images = load_images(cfg.paths.processed_dir)
    splits = splits_from_images(images)
    splits.assert_held_out((s.image_id for s in todo), stage="kd_r_targets")
    region_sets = load_region_sets(os.path.join(cfg.paths.derived_dir, "regions_train.json"))
    teacher = MedGemmaTeacher(cfg, adapter=adapter)
    header = {"header": True, "teacher_model_id": cfg.model.name, "adapter": teacher.adapter,
              "prompt_version": "vindr_prompts.build_messages/norm1000", "mask_method": cfg.kd_r.mask_method,
              "tau": float(cfg.kd_r.tau), "r_null": float(cfg.kd_r.r_null), "a_star": "answer without <loc>"}
    if not done:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(header) + "\n")

    max_regions = int(cfg.kd_r.max_regions)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for k, s in enumerate(todo):
            rs = region_sets[s.image_id]
            regs = list(rs.finding_regions)[:max_regions]
            img = Image.open(os.path.join(cfg.paths.dataset_dir, s.image_path)).convert("RGB")
            fx, fy = img.width / max(rs.width, 1), img.height / max(rs.height, 1)
            a_star = canonical_answer(s.answer)
            ims = [img] + [mask_region(img, r.box, fx, fy, method=cfg.kd_r.mask_method,
                                       fill=int(cfg.kd_r.get("fill_value", 0)),
                                       blur_sigma=float(cfg.kd_r.get("blur_sigma", 8.0))) for r in regs]
            logps: list[float] = []
            bs = int(cfg.kd_r.get("batch_images", 9))
            for i in range(0, len(ims), bs):
                logps += teacher.answer_logprob(ims[i:i + bs], s.question, a_star)
            logp0, masked = logps[0], logps[1:]
            rel = [counterfactual_relevance(logp0, lm) if math.isfinite(lm) else 0.0 for lm in masked]
            pi_t = teacher_distribution(rel, r_null=float(cfg.kd_r.r_null), tau=float(cfg.kd_r.tau)) if rel else [1.0]
            gen = teacher.generate([img], [s.question])[0]
            row = {"sample_id": s.sample_id, "image_id": s.image_id, "region_ids": [r.id for r in regs],
                   "a_star": a_star, "logp_original": logp0, "logp_masked": masked, "relevance": rel,
                   "pi_t": pi_t, "teacher_answer": gen,
                   "teacher_correct": teacher_named_finding(gen, s.gt_finding),
                   "confidence_raw": float(math.exp(logp0)) if math.isfinite(logp0) else 0.0}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if (k + 1) % 20 == 0:
                el = time.time() - t0
                print(f"[kd_r] {k + 1}/{len(todo)}  {el / (k + 1):.1f} s/sample  eta {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min")
    return path


# ---------------------------------------------------------------------------
def gate_shards(cfg, *, delta: float | None = None, temperature: float | None = None,
                use_correctness: bool = True, use_mass: bool = True, use_confidence: bool = True,
                random_gate: bool = False, seed: int = 0) -> dict:
    """Fill ``weight`` on every cached sample from its stored fields (spec §4.4 gate).

    Confidence is temperature-scaled: ``c = exp(logp0 / T)``; ``T`` is fitted
    on the tune split by ``calibrate`` or given. Table E ablation rows map to
    the ``use_*`` flags and ``random_gate`` (same retention, random subset).
    """
    cache_dir = os.path.join(cfg.paths.cache_dir, "kd_r")
    labels = load_labels(os.path.join(cfg.paths.derived_dir, "labels_train.jsonl"))
    delta = float(cfg.kd_r.delta if delta is None else delta)
    T = float(temperature or 1.0)
    rows, paths = [], []
    for name in sorted(os.listdir(cache_dir)):
        if name.endswith(".jsonl"):
            paths.append(os.path.join(cache_dir, name))
    stats = {"n": 0, "passed": 0, "teacher_wrong": 0, "mass_below_delta": 0, "zero_confidence": 0,
             "delta": delta, "temperature": T, "mean_null_mass": 0.0, "mean_offtarget_mass": 0.0}
    for p in paths:
        new_lines = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("header"):
                    new_lines.append(d); continue
                lab = labels.get(d["sample_id"])
                tgt = [i for i, rid in enumerate(d["region_ids"]) if lab and rid in lab.target_regions]
                mass = target_mass(d["pi_t"], tgt) if tgt else 0.0
                conf = min(1.0, math.exp(d["logp_original"] / T)) if math.isfinite(d["logp_original"]) else 0.0
                g = gate_weight(teacher_correct=bool(d["teacher_correct"]), mass=mass, confidence=conf, delta=delta,
                                use_correctness=use_correctness, use_mass=use_mass, use_confidence=use_confidence)
                d["mass"] = mass; d["confidence"] = conf; d["weight"] = g.weight; d["gate_reason"] = g.reason
                stats["n"] += 1
                stats["passed"] += g.passed
                if g.reason in stats:
                    stats[g.reason] += 1
                stats["mean_null_mass"] += d["pi_t"][-1]
                stats["mean_offtarget_mass"] += (1.0 - d["pi_t"][-1] - mass)
                new_lines.append(d); rows.append(d)
        if random_gate:
            rng = random.Random(seed)
            keep = stats["passed"]
            cand = [d for d in new_lines if not d.get("header")]
            rng.shuffle(cand)
            chosen = {d["sample_id"] for d in cand[:keep]}
            for d in new_lines:
                if not d.get("header"):
                    d["weight"] = 1.0 if d["sample_id"] in chosen else 0.0
                    d["gate_reason"] = "random"
        with open(p, "w", encoding="utf-8") as f:
            for d in new_lines:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    if stats["n"]:
        stats["mean_null_mass"] /= stats["n"]; stats["mean_offtarget_mass"] /= stats["n"]
        stats["retention"] = stats["passed"] / stats["n"]
    print(json.dumps(stats, indent=1))
    os.makedirs(cfg.paths.get("reports_dir", "reports"), exist_ok=True)
    with open(os.path.join(cfg.paths.get("reports_dir", "reports"), "kd_r_gate.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def teacher_sensitivity(cfg) -> dict:
    """Spec §8 entry check ("Trước KD-R"): is the teacher sensitive to masking
    the target region, and is there mass off-target? From the cache."""
    cache_dir = os.path.join(cfg.paths.cache_dir, "kd_r")
    labels = load_labels(os.path.join(cfg.paths.derived_dir, "labels_train.jsonl"))
    drops, null, off = [], [], []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(cache_dir, name), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("header"):
                    continue
                lab = labels.get(d["sample_id"])
                tgt = [i for i, rid in enumerate(d["region_ids"]) if lab and rid in lab.target_regions]
                if tgt and math.isfinite(d["logp_original"]):
                    drops.append(max(d["logp_original"] - d["logp_masked"][i] for i in tgt))
                    m = target_mass(d["pi_t"], tgt)
                    null.append(d["pi_t"][-1]); off.append(1.0 - d["pi_t"][-1] - m)
    n = len(drops)
    if n == 0:
        return {"n": 0}
    mean = sum(drops) / n
    rng = random.Random(0)
    boots = []
    for _ in range(500):
        b = [drops[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(b) / n)
    boots.sort()
    out = {"n": n, "mean_drop_target": mean, "ci95": [boots[12], boots[487]],
           "share_positive_drop": sum(x > 0 for x in drops) / n,
           "mean_null_mass": sum(null) / n, "mean_offtarget_mass": sum(off) / n,
           "sensitive": boots[12] > 0 and (sum(off) / n) > 0.05}
    print(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/teacher.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-correctness", action="store_true")
    ap.add_argument("--no-mass", action="store_true")
    ap.add_argument("--no-confidence", action="store_true")
    ap.add_argument("--random-gate", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.gate:
        gate_shards(cfg, delta=args.delta, temperature=args.temperature, use_correctness=not args.no_correctness,
                    use_mass=not args.no_mass, use_confidence=not args.no_confidence, random_gate=args.random_gate)
        return 0
    if args.sensitivity:
        out = teacher_sensitivity(cfg)
        return 0 if out.get("sensitive", False) else 1
    generate_shard(cfg, shard=args.shard, n_shards=args.n_shards, limit=args.limit, adapter=args.adapter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
