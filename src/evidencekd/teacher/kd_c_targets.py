"""KD-C target generation (spec §4.4, design §7.5) — sharded, resumable.

For every training question the teacher receives the question, the locked
(templated) answer and the fact table computed from the ANNOTATION regions,
and writes ``rationale_n`` rationales. Each is turned into claims
(``teacher/prompts.claims_from_text``), validated against the fact table and
filtered by the rule verifier (``kd.kd_c.build_targets``); only passing
observation claims become generator targets.

    python -m evidencekd.teacher.kd_c_targets --config configs/teacher.yaml --shard 0 --n-shards 30

Cache: ``cache/kd_c/shard-XXXX.jsonl`` with, per sample, the raw rationales,
the kept claims (rendered in the claim grammar) and the filter report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from evidencekd.claims.parser import render_claims
from evidencekd.data.anatomy import anatomy_for, load_anatomy
from evidencekd.data.regions import load_region_sets
from evidencekd.data.vindr_processed import load_images, load_samples, splits_from_images
from evidencekd.facts.build import build_facts, fact_region_map, facts_to_dict, load_background
from evidencekd.inference.templates import locked_answer_text
from evidencekd.kd.kd_c import FilterReport, build_targets
from evidencekd.models.decision_branch import load_labels
from evidencekd.models.verifier_rules import make_rule_verifier
from evidencekd.teacher.kd_r_targets import _existing_ids, select_samples
from evidencekd.teacher.prompts import GRAMMAR_HELP, claims_from_text, rationale_prompt
from evidencekd.utils.config import load_config

SYSTEM = ("You are an expert radiologist. You are given a chest X-ray, a question, the answer, "
          "and a list of verified facts with region ids. " + GRAMMAR_HELP)


def facts_for_training_sample(s, rec, rs, anatomy):
    present = list(rec.vqa_classes_present) or [s.gt_finding]
    return build_facts(rs, anatomy, present_classes=present)


def generate_shard(cfg, *, shard: int, n_shards: int, limit: int | None = None, adapter: str | None = None) -> str:
    from PIL import Image

    from evidencekd.teacher.medgemma import MedGemmaTeacher

    cache_dir = os.path.join(cfg.paths.cache_dir, "kd_c")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"shard-{shard:04d}.jsonl")
    done = _existing_ids(path)
    samples = select_samples(cfg, shard=shard, n_shards=n_shards)
    if limit:
        samples = samples[:limit]
    todo = [s for s in samples if s.sample_id not in done]
    print(f"[kd_c] shard {shard}/{n_shards}: {len(samples)} samples, {len(done)} done, {len(todo)} to do")
    if not todo:
        return path
    images = load_images(cfg.paths.processed_dir)
    splits = splits_from_images(images)
    splits.assert_held_out((s.image_id for s in todo), stage="kd_c_targets")
    derived = cfg.paths.derived_dir
    region_sets = load_region_sets(os.path.join(derived, "regions_train.json"))
    anat = load_anatomy(os.path.join(derived, "anatomy.json"))
    labels = load_labels(os.path.join(derived, "labels_train.jsonl"))
    background = load_background(os.path.join(derived, "background.json"))
    teacher = MedGemmaTeacher(cfg, adapter=adapter)
    gran = cfg.kd_c.get("granularity", "claim")
    n_rat = int(cfg.generation.get("rationale_n", 3))
    report = FilterReport()
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        for k, s in enumerate(todo):
            rec = images[s.image_id]
            rs = region_sets[s.image_id]
            a = anatomy_for(anat, s.image_id, rec.width, rec.height)
            ft = facts_for_training_sample(s, rec, rs, a)
            lab = labels.get(s.sample_id)
            from evidencekd.data.qa_labels import SIDES
            side = SIDES[lab.side] if lab and lab.side >= 0 else None
            count = lab.count if lab and lab.count >= 0 else None
            answer = locked_answer_text(s.type, finding=s.gt_finding, present=True,
                                        findings_present=list(rec.vqa_classes_present), side=side, count=count)
            prompt = rationale_prompt(s.question, answer, ft)
            img = Image.open(os.path.join(cfg.paths.dataset_dir, s.image_path)).convert("RGB")
            texts = teacher.generate_text(img, prompt, system=SYSTEM, n=n_rat,
                                          max_new_tokens=int(cfg.generation.get("rationale_max_new_tokens", 200)),
                                          temperature=float(cfg.generation.get("rationale_temperature", 0.7)))
            rationales, followed = [], 0
            for t in texts:
                claims, ok = claims_from_text(t, ft)
                followed += ok
                rationales.append(claims)
            kept = build_targets(rationales, fact_table=fact_region_map(ft), known_region_ids=rs.ids,
                                 verifier=make_rule_verifier(ft, rs.ids), granularity=gran, report=report)
            bg = background.get(s.gt_finding, [])[:1]
            row = {"sample_id": s.sample_id, "image_id": s.image_id, "question": s.question, "answer": answer,
                   "facts": facts_to_dict(ft), "region_ids": list(rs.ids), "raw_rationales": texts,
                   "followed_grammar": followed, "kept": render_claims(kept),
                   "n_kept_observations": sum(c.is_observation for c in kept), "background": bg}
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if (k + 1) % 20 == 0:
                el = time.time() - t0
                print(f"[kd_c] {k + 1}/{len(todo)}  {el / (k + 1):.1f} s/sample  retention {report.retention:.3f}")
    with open(os.path.join(cache_dir, f"report-{shard:04d}.json"), "w") as f:
        json.dump(report.as_dict(), f, indent=2)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/teacher.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args(argv)
    generate_shard(load_config(args.config), shard=args.shard, n_shards=args.n_shards, limit=args.limit, adapter=args.adapter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
