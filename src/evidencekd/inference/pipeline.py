"""Proposed-S / Proposed-L inference (design §7.3, §7.7, §7.8; spec §2 output).

    decision outputs -> Answer Lock -> fact table -> Evidence Pointer
    -> generator (template | S | L) -> verifier (rules | learned)
    -> enforce_answer_lock -> JSON -> VinDr bridge rows

Batch mode scores a split from the cached files:

    python -m evidencekd.inference.pipeline --split test --generator template
    python -m evidencekd.inference.pipeline --split test --generator s --generator-dir outputs/generator_s/final
    python -m evidencekd.inference.pipeline --split tune --tune       # choose thresholds -> reports/thresholds.json

It writes ``outputs/pipeline_{split}_{generator}.jsonl`` (one JSON per
question, spec §2 shape + the bridge string) and
``reports/preds_{split}_{generator}.jsonl`` for ``vindr_cxr_vqa.evaluate``.

:class:`Runner` is the same logic for ONE image + question in process (the
HTTP server), running the detector and the decision branch itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Mapping, Sequence

from evidencekd.claims.constrain import enforce_citations
from evidencekd.claims.parser import SUPPORTED, Claim, parse_claims, render_claims
from evidencekd.data.anatomy import anatomy_for, load_anatomy
from evidencekd.data.boxes import Box
from evidencekd.data.qa_labels import QUESTION_TYPES, SIDES
from evidencekd.data.regions import RegionSet, load_region_sets
from evidencekd.data.vindr_processed import VQASample, load_images, load_samples, splits_from_images
from evidencekd.eval.vindr_bridge import PredictionRow, compose, loc_token, write_rows
from evidencekd.facts.build import build_facts, fact_region_map, facts_for_classes, facts_to_dict, load_background, serialize_facts
from evidencekd.facts.geometry import FactTable
from evidencekd.inference.answer_lock import enforce_answer_lock, lock_answer, select_evidence
from evidencekd.inference.templates import locked_answer_text, template_claims_full, template_claims_simple
from evidencekd.labels import VQA_CLASSES, finding_in_text
from evidencekd.models.verifier_rules import verify_all
from evidencekd.utils.config import load_config

DEFAULT_THRESHOLDS = {"finding": 0.5, "yes": 0.5, "absence": 0.3}


def load_thresholds(path: str | None) -> dict:
    if path and os.path.isfile(path):
        with open(path) as f:
            return {**DEFAULT_THRESHOLDS, **json.load(f)}
    return dict(DEFAULT_THRESHOLDS)


# ---------------------------------------------------------------------------
def decide(sample_type: str, question: str, dec: dict, thr: Mapping[str, float], *,
           gt_finding_hint: str | None = None, regions: RegionSet | None = None) -> dict:
    """Heads -> the locked decision. ``dec`` is one line of decision_{split}.jsonl.

    When the question names no class ("What abnormality is in the upper
    lobe?"), the class asked about is the class of the region the relevance
    head ranks first — that head exists precisely to tie a question to a
    region (spec §4.1); the whole-image finding head only says what is there.
    """
    qtype = sample_type if sample_type in QUESTION_TYPES else dec.get("question_type", "Is_there")
    fprob: dict[str, float] = dec["finding_probs"]
    present_all = [c for c in VQA_CLASSES if fprob.get(c, 0.0) >= thr["finding"]]
    asked = finding_in_text(question) or gt_finding_hint
    if asked is None and regions is not None and regions.finding_regions and dec.get("relevance"):
        j = max(range(len(regions.finding_regions)), key=lambda i: dec["relevance"][i] if i < len(dec["relevance"]) else -1)
        asked = regions.finding_regions[j].label
    if asked is None:
        asked = max(fprob, key=fprob.get) if fprob else None
    present = (asked in present_all) if asked else bool(present_all)
    if qtype in ("Is_there", "Yes_No"):
        present = present and dec.get("yes_prob", 1.0) >= thr["yes"] if dec.get("yes_prob") is not None else present
    side = max(dec["side_probs"], key=dec["side_probs"].get) if dec.get("side_probs") else None
    count_probs = dec.get("count_probs") or []
    count = int(max(range(len(count_probs)), key=lambda i: count_probs[i])) if count_probs else None
    return {"question_type": qtype, "finding": asked, "present": present, "findings_present": present_all,
            "side": side, "count": count, "scores": {"answer": round(float(dec.get("yes_prob", fprob.get(asked, 0.0) if asked else 0.0)), 4),
                                                     "finding": round(float(fprob.get(asked, 0.0)) if asked else 0.0, 4)}}


def answer_box(regions: RegionSet, pointer_ids: Sequence[str], relevance: Sequence[float], finding: str | None) -> Box | None:
    """The box the bridge emits: first pointed finding region of the asked class,
    else the most relevant region of that class, else None (no fabricated box)."""
    for rid in pointer_ids:
        r = regions.by_id(rid) if rid in regions.ids else None
        if r and r.kind == "finding" and (finding is None or r.label == finding):
            return r.box
    best, best_v = None, -1.0
    for r, v in zip(regions.finding_regions, relevance):
        if (finding is None or r.label == finding) and v > best_v:
            best, best_v = r.box, v
    return best


def build_output(sample: VQASample | None, question: str, qtype: str, regions: RegionSet, anatomy, dec: dict,
                 thr: Mapping[str, float], *, generator: str, gen_fn=None, verifier=None,
                 background: Mapping[str, Sequence[str]], image=None, pointer_top_k: int = 3,
                 pointer_min_p: float = 0.05) -> dict:
    t0 = time.time()
    d = decide(qtype, question, dec, thr, gt_finding_hint=None, regions=regions)
    finding = d["finding"]
    # fact table from the decision branch's own outputs
    ft = build_facts(regions, anatomy, present_classes=d["findings_present"],
                     whole_image_scores=dec["finding_probs"], absence_threshold=thr["absence"])
    # answer lock
    zone = next((f.value for f in ft.facts if f.attr == "zone" and any(
        regions.by_id(r).label == finding for r in f.evidence_ids if r in regions.ids and regions.by_id(r).kind == "finding")), None)
    text = locked_answer_text(d["question_type"], finding=finding, present=d["present"],
                              findings_present=d["findings_present"], side=d["side"] if d["present"] else None,
                              zone=zone, count=d["count"] if d["question_type"] == "How_many" else None)
    locked = lock_answer(answer=text, question_type=d["question_type"],
                         findings=tuple(d["findings_present"]), scores=d["scores"])
    # pointer
    finding_ids = [r.id for r in regions.finding_regions]
    ptr = dec.get("pointer") or []
    probs = ptr[:len(finding_ids)]
    null_p = ptr[-1] if ptr else 1.0
    pr = select_evidence(region_ids=finding_ids, probabilities=probs, null_probability=null_p,
                         top_k=pointer_top_k, min_probability=pointer_min_p) if finding_ids else \
        select_evidence(region_ids=[], probabilities=[], null_probability=1.0)
    # selected facts: those about the asked finding + those citing pointed regions
    sel = {f.id: f for f in facts_for_classes(ft, [finding] if finding else [])}
    for f in ft.facts:
        if set(f.evidence_ids) & set(pr.region_ids):
            sel[f.id] = f
    if not sel:
        sel = {f.id: f for f in ft.facts[:4]}
    sel_ft = FactTable(facts=list(sel.values()), region_ids=ft.region_ids)
    bg = list(background.get(finding, [])[:1]) if finding else []
    # generator
    if generator == "template_simple":
        claims = template_claims_simple(ft, finding, background=bg)
    elif generator == "template":
        claims = template_claims_full(sel_ft, background=bg)
    else:
        gen_text = gen_fn(question, locked.answer, serialize_facts(sel_ft), regions.ids, image)
        claims = parse_claims(gen_text)
    claims = enforce_citations(claims, fact_region_map(ft), regions.ids)
    # verifier
    if verifier is None:
        claims = verify_all(claims, ft, regions.ids)
    else:
        rules = verify_all(claims, ft, regions.ids)          # grammar/citation sanity
        learned = verifier.statuses(claims, ft.facts)
        out_claims, j = [], 0
        for c, rc in zip(claims, rules):
            if c.is_observation:
                st, conf = learned[j]; j += 1
                st = st if rc.status != "insufficient" or rc.reject_reason is None else rc.status
                out_claims.append(Claim(c.type, c.text, c.fact_ids, c.evidence_ids, st, rc.reject_reason))
            else:
                out_claims.append(c)
        claims = out_claims
    # answer lock enforced against what the generator said
    _, warnings = enforce_answer_lock(locked, " ".join(c.text for c in claims if c.is_observation))
    supported = [c for c in claims if c.is_observation and c.status == SUPPORTED]
    box = answer_box(regions, pr.region_ids, dec.get("relevance", []), finding) if d["present"] else None
    out = {
        "sample_id": sample.sample_id if sample else None, "image_id": regions.image_id, "question": question,
        "question_type": locked.question_type, "answer": locked.answer, "finding": [finding] if (finding and d["present"]) else [],
        "findings_present": d["findings_present"],
        "scores": {**d["scores"], "verifier": round(sum(1 for c in claims if c.is_observation and c.status == SUPPORTED) /
                                                  max(1, sum(1 for c in claims if c.is_observation)), 4)},
        "regions": [r.as_dict() for r in regions.regions],
        "facts": facts_to_dict(ft), "pointer": {"region_ids": list(pr.region_ids), "null_probability": round(pr.null_probability, 4),
                                                "status": pr.status},
        "claims": [c.as_dict() for c in claims], "warnings": warnings,
        "answer_box": None if box is None else [box.x_min, box.y_min, box.x_max, box.y_max],
        "bridge": compose(locked.answer, [(box, regions.width, regions.height)] if box is not None else [],
                          " ".join(c.text for c in supported)),
        "latency_s": round(time.time() - t0, 4),
    }
    return out


# ---------------------------------------------------------------------------
def run_split(cfg, *, split: str, generator: str, decision_path: str | None, generator_dir: str | None,
              verifier_path: str | None, thresholds_path: str | None, out_path: str | None, limit: int | None,
              teacher_cfg=None, regions_path: str | None = None, tag: str | None = None) -> dict:
    proc, derived = cfg.paths.processed_dir, cfg.paths.derived_dir
    images = load_images(proc)
    splits = splits_from_images(images)
    base_split = "val" if split == "tune" else split
    samples = load_samples(proc, base_split)
    if split == "tune":
        tune = set(splits.tune)
        samples = [s for s in samples if s.image_id in tune]
    if limit:
        samples = samples[:limit]
    region_sets = load_region_sets(regions_path or os.path.join(derived, f"regions_{base_split}.json"))
    tag = tag or generator
    anat = load_anatomy(os.path.join(derived, "anatomy.json"))
    background = load_background(os.path.join(derived, "background.json"))
    dpath = decision_path or os.path.join(cfg.paths.output_dir, f"decision_{base_split}.jsonl")
    decisions = {}
    with open(dpath, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); decisions[d["sample_id"]] = d
    thr = load_thresholds(thresholds_path or os.path.join(cfg.paths.get("reports_dir", "reports"), "thresholds.json"))
    gen_fn, verifier = None, None
    if generator == "s":
        from evidencekd.models.generator_s import GeneratorS, build_input
        gs = GeneratorS(cfg, generator_dir or os.path.join(cfg.paths.output_dir, "generator_s", "final"))
        gen_fn = lambda q, a, facts, ids, img: gs.generate([build_input(q, a, facts, ids)], [None], [ids])[0]  # noqa: E731
    elif generator == "l":
        from evidencekd.models.generator_l import GeneratorL
        from evidencekd.models.generator_s import build_input
        gl = GeneratorL(cfg, teacher_cfg, generator_dir or os.path.join(cfg.paths.output_dir, "generator_l"))
        gen_fn = lambda q, a, facts, ids, img: gl.generate(img, build_input(q, a, facts, ids))  # noqa: E731
    if verifier_path and os.path.isfile(verifier_path):
        from evidencekd.models.verifier import LearnedVerifier
        verifier = LearnedVerifier(verifier_path)
    if generator == "s":
        # S uses the fact table as the citation whitelist at decode time
        from evidencekd.models.generator_s import build_input  # noqa: F401
    out_path = out_path or os.path.join(cfg.paths.output_dir, f"pipeline_{split}_{tag}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rows: list[PredictionRow] = []
    n = 0; t0 = time.time(); missing = 0
    need_image = generator == "l"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            dec = decisions.get(s.sample_id)
            rs = region_sets.get(s.image_id)
            if dec is None or rs is None:
                missing += 1; continue
            a = anatomy_for(anat, s.image_id, s.image_width, s.image_height)
            img = None
            if need_image:
                from PIL import Image
                img = Image.open(os.path.join(cfg.paths.dataset_dir, s.image_path)).convert("RGB")
            if generator == "s":
                ft_tmp = build_facts(rs, a, present_classes=[c for c in VQA_CLASSES if dec["finding_probs"].get(c, 0) >= thr["finding"]])
                fmap = fact_region_map(ft_tmp)
                gen_fn_s = lambda q, a_, facts, ids, img_, fmap=fmap, ids_=rs.ids: gs.generate([build_input(q, a_, facts, ids)], [fmap], [ids_])[0]  # noqa: E731
                out = build_output(s, s.question, s.type, rs, a, dec, thr, generator=generator, gen_fn=gen_fn_s,
                                   verifier=verifier, background=background, image=img)
            else:
                out = build_output(s, s.question, s.type, rs, a, dec, thr, generator=generator, gen_fn=gen_fn,
                                   verifier=verifier, background=background, image=img)
            f.write(json.dumps(out, ensure_ascii=False) + "\n"); n += 1
            gt_loc = loc_token(s.gt_box, s.image_width, s.image_height) if s.gt_box else ""
            rows.append(PredictionRow(s.sample_id, s.image_id, s.question, out["bridge"], s.gt_finding, gt_loc))
    rep_dir = cfg.paths.get("reports_dir", "reports")
    pred_path = os.path.join(rep_dir, f"preds_{split}_{tag}.jsonl")
    write_rows(rows, pred_path)
    info = {"split": split, "generator": generator, "tag": tag, "n": n, "missing": missing, "out": out_path, "preds": pred_path,
            "latency_s_per_question": round((time.time() - t0) / max(n, 1), 4), "thresholds": thr}
    print(json.dumps(info, indent=1))
    return info


def tune_thresholds(cfg, *, decision_path: str | None, out_path: str) -> dict:
    """Pick the finding threshold on the tune split by image-level F1 of the
    predicted finding set vs the fused annotation (the Table 4 criterion)."""
    proc = cfg.paths.processed_dir
    images = load_images(proc); splits = splits_from_images(images)
    tune = set(splits.tune)
    dpath = decision_path or os.path.join(cfg.paths.output_dir, "decision_val.jsonl")
    per_image: dict[str, dict[str, float]] = {}
    with open(dpath, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["image_id"] in tune:
                per_image.setdefault(d["image_id"], d["finding_probs"])
    best = (0.0, 0.5)
    for t in [i / 20 for i in range(2, 19)]:
        tp = fp = fn = 0
        for iid, probs in per_image.items():
            pred = {c for c in VQA_CLASSES if probs.get(c, 0) >= t}
            gt = set(images[iid].vqa_classes_present)
            tp += len(pred & gt); fp += len(pred - gt); fn += len(gt - pred)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, t)
    thr = {**DEFAULT_THRESHOLDS, "finding": best[1], "tuned_on": "tune (val minus test)", "finding_f1_tune": round(best[0], 4)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thr, f, indent=2)
    print(f"[tune] finding threshold {best[1]} (F1 {best[0]:.4f}) -> {out_path}")
    return thr


# ---------------------------------------------------------------------------
class Runner:
    """One image + one question, in process (the HTTP server)."""

    def __init__(self, dcfg, *, detector_ckpt: str, decision_ckpt: str, thresholds: dict | None = None,
                 anatomy_cache: str | None = None, background_path: str | None = None, generator="template",
                 generator_dir: str | None = None, verifier_path: str | None = None):
        import torch
        from evidencekd.models.decision_branch import DecisionDataset, collate, load_for_inference
        from evidencekd.models.detector import load_for_inference as load_det
        from evidencekd.utils.config import Config

        self.dcfg = dcfg
        det_cfg = Config(torch.load(detector_ckpt, map_location="cpu")["config"])
        self.detector, self.dev = load_det(det_cfg, detector_ckpt)
        self.model, self.tok, _ = load_for_inference(dcfg, decision_ckpt)
        self.thr = thresholds or dict(DEFAULT_THRESHOLDS)
        self.anat = load_anatomy(anatomy_cache)
        self.background = load_background(background_path)
        self.generator = generator
        self.gen = None
        if generator == "s":
            from evidencekd.models.generator_s import GeneratorS
            self.gen = GeneratorS(dcfg, generator_dir)
        self.verifier = None
        if verifier_path and os.path.isfile(verifier_path):
            from evidencekd.models.verifier import LearnedVerifier
            self.verifier = LearnedVerifier(verifier_path)
        self._DecisionDataset, self._collate = DecisionDataset, collate

    def run(self, image, question: str, *, image_id: str = "upload", orig_size: tuple[int, int] | None = None,
            qtype: str | None = None) -> dict:
        import torch
        from evidencekd.data.regions import regions_from_detections
        from evidencekd.eval.detection import Detection
        from evidencekd.labels import VQA_CLASSES as CLS

        W, H = orig_size or image.size
        x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(image.height, image.width, 3).permute(2, 0, 1).float().div(255).to(self.dev)
        with torch.no_grad():
            p = self.detector([x])[0]
        fx, fy = image.width / W, image.height / H
        dets = [Detection(Box(b[0] / fx, b[1] / fy, b[2] / fx, b[3] / fy, class_name=CLS[l - 1]), s)
                for b, l, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist())]
        a = anatomy_for(self.anat, image_id, W, H)
        rs = regions_from_detections(image_id, W, H, dets, anatomy=a, max_regions=int(self.dcfg.data.max_regions),
                                     score_threshold=float(self.dcfg.data.detector_score_threshold))
        fake = VQASample(f"{image_id}_q", image_id, "", "test", question, "", "", qtype or "Is_there", "", "",
                         None, None, "published", True, W, H, image.width, image.height)
        ds = self._DecisionDataset([fake], None, {image_id: rs}, "", self.tok, max_q_tokens=int(self.dcfg.data.max_question_tokens))
        item = ds[0]; item["image"] = x.cpu()
        batch = self._collate([item])
        with torch.no_grad():
            out = self.model(batch)
        nf = item["n_finding"]
        dec = {"question_type": QUESTION_TYPES[int(out["qtype_logits"].argmax())],
               "finding_probs": {c: float(v) for c, v in zip(CLS, torch.sigmoid(out["finding_logits"][0].float()))},
               "yes_prob": float(torch.softmax(out["yn_logits"][0].float(), -1)[1]),
               "side_probs": {s: float(v) for s, v in zip(SIDES, torch.softmax(out["side_logits"][0].float(), -1))},
               "count_probs": torch.softmax(out["count_logits"][0].float(), -1).tolist(),
               "relevance": torch.sigmoid(out["rel_logits"][0][:nf].float()).tolist(),
               "pointer": torch.softmax(out["ptr_logits"][0].float(), -1)[:nf].tolist() + [float(torch.softmax(out["ptr_logits"][0].float(), -1)[-1])]}
        gen_fn = None
        if self.gen is not None:
            from evidencekd.models.generator_s import build_input
            gen_fn = lambda q, a_, facts, ids, img: self.gen.generate([build_input(q, a_, facts, ids)], [{}], [ids])[0]  # noqa: E731
        qt = qtype or dec["question_type"]
        return build_output(None, question, qt, rs, a, dec, self.thr, generator=self.generator if self.gen is None else "s",
                            gen_fn=gen_fn, verifier=self.verifier, background=self.background, image=image)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/decision.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test", "tune"])
    ap.add_argument("--generator", default="template", choices=["template", "template_simple", "s", "l"])
    ap.add_argument("--generator-dir", default=None)
    ap.add_argument("--verifier", default=None, help="path to verifier.pt (learned v2); default rules")
    ap.add_argument("--decision", default=None)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--regions", default=None, help="region-set file (default derived/regions_{split}.json)")
    ap.add_argument("--tag", default=None, help="name suffix of the output files (default = generator)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tcfg = load_config(args.teacher_config) if os.path.isfile(args.teacher_config) else None
    if args.tune:
        tune_thresholds(cfg, decision_path=args.decision,
                        out_path=args.thresholds or os.path.join(cfg.paths.get("reports_dir", "reports"), "thresholds.json"))
        return 0
    run_split(cfg, split=args.split, generator=args.generator, decision_path=args.decision, generator_dir=args.generator_dir,
              verifier_path=args.verifier, thresholds_path=args.thresholds, out_path=args.out, limit=args.limit,
              teacher_cfg=tcfg, regions_path=args.regions, tag=args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
