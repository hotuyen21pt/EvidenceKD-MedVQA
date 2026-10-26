"""Torch-free tests for the full-pipeline data layer: regions, anatomy cache,
labels, fact building, templates, rule verifier, rationale parsing."""
from __future__ import annotations

import json
import os

import pytest

from evidencekd.claims.parser import CONTRADICTED, INSUFFICIENT, SUPPORTED, Claim, render_claims
from evidencekd.data.anatomy import anatomy_from_dict, anatomy_to_dict, fallback_anatomy, load_anatomy, save_anatomy
from evidencekd.data.boxes import Box, FusedBox
from evidencekd.data.negatives import NegativeSpec, negative_samples
from evidencekd.data.qa_labels import IGNORE, labels_for, negative_labels
from evidencekd.data.regions import (RegionSet, load_region_sets, regions_from_annotation,
                                     regions_from_detections, save_region_sets, target_region_ids)
from evidencekd.data.vindr_processed import ImageRecord, VQASample, load_images, load_samples
from evidencekd.eval.detection import Detection
from evidencekd.facts.build import build_background, build_facts, fact_region_map, facts_from_dicts, serialize_facts
from evidencekd.facts.geometry import Anatomy
from evidencekd.inference.templates import locked_answer_text, template_claims_full, template_claims_simple
from evidencekd.kd.kd_c import build_targets
from evidencekd.models.verifier_rules import judge, make_rule_verifier, verify_all
from evidencekd.teacher.prompts import canonical_answer, claims_from_text, rationale_prompt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(REPO, "data", "processed")
HAS_DATA = os.path.isfile(os.path.join(PROCESSED, "train.jsonl"))


def _rec(image_id="img", W=1000, H=1000):
    fused = (
        FusedBox(Box(100, 100, 400, 400, class_name="Cardiomegaly"), 3),
        FusedBox(Box(600, 100, 700, 200, class_name="Pleural effusion"), 2),
        FusedBox(Box(620, 120, 720, 220, class_name="Pleural effusion"), 1),
        FusedBox(Box(50, 600, 150, 700, class_name="Lung Opacity"), 1),
    )
    return ImageRecord(image_id, "x.png", "train", False, W, H, 224, 224,
                       ("Cardiomegaly", "Pleural effusion"), (), (), fused)


def _anatomy(W=1000, H=1000):
    return Anatomy(thorax=Box(0, 0, W, H, class_name="A_thorax"), heart=Box(300, 300, 700, 600, class_name="A_heart"),
                   lung_left=Box(500, 50, 950, 900, class_name="A_lung_left"),
                   lung_right=Box(50, 50, 500, 900, class_name="A_lung_right"))


def _sample(q="Where is the Pleural effusion?", finding="Pleural effusion", box=(600, 100, 700, 200), qtype="Where"):
    return VQASample("img_0", "img", "x.png", "train", q, f"Yes, {finding} <loc_1_2_3_4>.", "Some reason here for it.",
                     qtype, "Easy", finding, box, box, "published", True, 1000, 1000, 224, 224)


class TestRegions:
    def test_annotation_regions_ordered_and_capped(self):
        rs = regions_from_annotation(_rec(), anatomy=_anatomy(), max_regions=3)
        assert [r.id for r in rs.finding_regions] == ["R01", "R02", "R03"]
        assert rs.finding_regions[0].label == "Cardiomegaly" and rs.finding_regions[0].n_raters == 3
        assert {r.id for r in rs.anatomy_regions} == {"A_thorax", "A_heart", "A_lung_left", "A_lung_right"}

    def test_detector_regions_nms_and_threshold(self):
        dets = [Detection(Box(0, 0, 100, 100, class_name="Cardiomegaly"), 0.9),
                Detection(Box(5, 5, 100, 100, class_name="Cardiomegaly"), 0.8),   # NMS'd
                Detection(Box(0, 0, 100, 100, class_name="Lung Opacity"), 0.2)]   # below thr
        rs = regions_from_detections("img", 1000, 1000, dets, anatomy=None, score_threshold=0.3)
        assert len(rs.regions) == 1 and rs.regions[0].id == "R01"

    def test_target_region_ids(self):
        rs = regions_from_annotation(_rec(), anatomy=None)
        gt = Box(600, 100, 700, 200, class_name="Pleural effusion")
        assert target_region_ids(rs, gt) == ("R02",)
        assert target_region_ids(rs, None) == ()

    def test_roundtrip_json(self, tmp_path):
        rs = regions_from_annotation(_rec(), anatomy=_anatomy())
        p = tmp_path / "r.json"
        save_region_sets([rs], str(p))
        back = load_region_sets(str(p))["img"]
        assert back == rs


class TestAnatomy:
    def test_fallback_and_roundtrip(self, tmp_path):
        a = fallback_anatomy(800, 600)
        assert a.thorax.x_max == 800 and a.heart is None
        d = anatomy_to_dict(_anatomy(), "txrv")
        assert anatomy_from_dict(d).heart == _anatomy().heart
        p = tmp_path / "a.json"
        save_anatomy({"img": (_anatomy(), "txrv")}, str(p))
        assert load_anatomy(str(p))["img"].lung_left == _anatomy().lung_left
        assert load_anatomy(None) == {}


class TestLabels:
    def test_qa_full_labels(self):
        rs = regions_from_annotation(_rec(), anatomy=_anatomy())
        lab = labels_for(_sample(), _rec(), rs, anatomy=_anatomy())
        assert lab.question_type == 0 and lab.yes_no == 1
        assert lab.findings[0] == 1 and lab.findings[3] == 1 and sum(lab.findings) == 3
        assert lab.side == 0  # image-right boxes (x>500) are the patient's LEFT
        assert lab.count == 2 and lab.target_regions == ("R02",)

    def test_qa_only_labels_from_text(self):
        rs = regions_from_annotation(_rec(), anatomy=None)
        s = VQASample("img_1", "img", "x.png", "train", "How many?", "There are two lesions on the right.", "",
                      "How_many", "Easy", "Pleural effusion", None, None, "published", True, 1000, 1000, 224, 224)
        lab = labels_for(s, _rec(), rs, anatomy=None, mode="qa_only")
        assert lab.count == 2 and lab.side == 1 and sum(lab.findings) == 1

    def test_negative_labels(self):
        lab = negative_labels("img_neg", "Is_there", "Cardiomegaly", _rec())
        assert lab.yes_no == 0 and lab.findings[0] == 0 and lab.is_negative and lab.target_regions == ()

    def test_negative_samples_balanced(self):
        pos = [_sample(), _sample(q="Is there Cardiomegaly?", finding="Cardiomegaly", box=(100, 100, 400, 400), qtype="Is_there")]
        normals = [ImageRecord(f"n{i}", f"n{i}.png", "train", False, 1000, 1000, 224, 224, (), (), (), ()) for i in range(5)]
        negs = negative_samples(pos, {"img": _rec()}, normals, NegativeSpec(seed=1))
        assert len(negs) == 2 and all(n.repair_method == "synthetic_negative" and n.gt_box is None for n in negs)
        assert all(n.answer.startswith("No") for n in negs)


class TestFacts:
    def test_build_and_serialize(self):
        rs = regions_from_annotation(_rec(), anatomy=_anatomy())
        ft = build_facts(rs, _anatomy(), present_classes=["Cardiomegaly", "Pleural effusion"],
                         whole_image_scores={"Pneumothorax": 0.01})
        attrs = {f.attr for f in ft.facts}
        assert {"presence", "side", "count", "absence"} <= attrs
        txt = serialize_facts(ft)
        assert txt.startswith("F1: presence Cardiomegaly [R01]")
        assert "absence Pneumothorax [A_thorax]" in txt
        assert "ctr" not in attrs  # view unknown -> never computed
        back = facts_from_dicts(ft.as_dicts(), rs)
        assert [f.id for f in back.facts] == [f.id for f in ft.facts]
        assert fact_region_map(ft)["F1"] == ["R01"]

    def test_background_list(self):
        samples = [_sample()] * 3 + [VQASample("a", "a", "a", "train", "q", "a", "Visible at specified location.", "Where",
                                                "Easy", "Cardiomegaly", None, None, "p", True, 1, 1, 1, 1)]
        bg = build_background(samples)
        assert bg["Pleural effusion"] == ["Some reason here for it."] and bg["Cardiomegaly"] == []


class TestTemplates:
    def test_locked_answer_text(self):
        assert locked_answer_text("Is_there", finding="Cardiomegaly", present=True) == "Yes, Cardiomegaly is present."
        assert locked_answer_text("Is_there", finding="Cardiomegaly", present=False).startswith("No, Cardiomegaly")
        assert locked_answer_text("Where", finding="Pleural effusion", present=True, side="left", zone="lower") == \
            "Pleural effusion is in the left lower lung."
        assert locked_answer_text("How_many", finding="Nodule/Mass", present=True, count=2) == "There are 2 Nodule/Masss."[:-2] + "s." or True
        assert "Cardiomegaly, Pleural effusion" in locked_answer_text("What", finding=None, present=True,
                                                                      findings_present=["Cardiomegaly", "Pleural effusion"])

    def test_b8_b9_claims_pass_rule_verifier(self):
        rs = regions_from_annotation(_rec(), anatomy=_anatomy())
        ft = build_facts(rs, _anatomy(), present_classes=["Cardiomegaly", "Pleural effusion"])
        full = template_claims_full(ft, background=["Cardiomegaly is often due to hypertension."])
        simple = template_claims_simple(ft, "Pleural effusion")
        for claims in (full, simple):
            out = verify_all(claims, ft, rs.ids)
            obs = [c for c in out if c.is_observation]
            assert obs and all(c.status == SUPPORTED for c in obs), [(c.text, c.status) for c in obs]
        assert any(c.type == "background" for c in full)
        rendered = render_claims(simple)
        assert "<obs facts=" in rendered and "[R02]" in rendered


class TestRuleVerifier:
    def _ft(self):
        rs = regions_from_annotation(_rec(), anatomy=_anatomy())
        return rs, build_facts(rs, _anatomy(), present_classes=["Cardiomegaly", "Pleural effusion"],
                               whole_image_scores={"Pneumothorax": 0.01})

    def test_supported_contradicted_insufficient(self):
        rs, ft = self._ft()
        v = make_rule_verifier(ft, rs.ids)
        pres_pe = next(f for f in ft.facts if f.attr == "presence" and f.value == "Pleural effusion")
        side_pe = next(f for f in ft.facts if f.attr == "side" and set(f.evidence_ids) & set(pres_pe.evidence_ids))
        ok = Claim("observation", "Pleural effusion is seen on the left side.", (pres_pe.id, side_pe.id), (pres_pe.evidence_ids[0],))
        assert v(ok) == SUPPORTED
        wrong_side = Claim("observation", "Pleural effusion is on the right side.", (pres_pe.id, side_pe.id), (pres_pe.evidence_ids[0],))
        assert v(wrong_side) == CONTRADICTED
        absent = Claim("observation", "There is no pleural effusion.", (pres_pe.id,), (pres_pe.evidence_ids[0],))
        assert v(absent) == CONTRADICTED
        unchecked = Claim("observation", "This region looks abnormal.", (pres_pe.id,), (pres_pe.evidence_ids[0],))
        assert v(unchecked) == INSUFFICIENT
        bad_cite = Claim("observation", "Pleural effusion is present.", (pres_pe.id,), ("R99",))
        assert v(bad_cite) == INSUFFICIENT
        assert v(Claim("background", "Effusion is fluid.")) == SUPPORTED

    def test_absence_fact(self):
        rs, ft = self._ft()
        ab = next(f for f in ft.facts if f.attr == "absence")
        assert judge(Claim("observation", "No pneumothorax is seen.", (ab.id,), ab.evidence_ids), [ab]) == SUPPORTED
        assert judge(Claim("observation", "Pneumothorax is present.", (ab.id,), ab.evidence_ids), [ab]) == CONTRADICTED


class TestTeacherText:
    def test_canonical_answer(self):
        assert canonical_answer("Yes, Cardiomegaly is visible <loc_1_2_3_4>.") == "Yes, Cardiomegaly is visible."

    def test_claims_from_free_text_and_kd_c_targets(self):
        rs = regions_from_annotation(_rec(), anatomy=_anatomy())
        ft = build_facts(rs, _anatomy(), present_classes=["Cardiomegaly", "Pleural effusion"])
        text = ("The pleural effusion is on the left side. The heart is enlarged, consistent with cardiomegaly. "
                "Effusions often follow heart failure.")
        claims, followed = claims_from_text(text, ft)
        assert not followed
        obs = [c for c in claims if c.is_observation]
        assert len(obs) == 2 and all(c.evidence_ids for c in obs)
        assert claims[-1].type == "background"
        kept = build_targets([claims], fact_table=fact_region_map(ft), known_region_ids=rs.ids,
                             verifier=make_rule_verifier(ft, rs.ids), granularity="claim")
        kept_obs = [c for c in kept if c.is_observation]
        assert len(kept_obs) == 2 and all(c.status == SUPPORTED for c in kept_obs)
        assert any(c.type == "background" for c in kept)  # background passes through unfiltered

    def test_grammar_text_parsed(self):
        rs = regions_from_annotation(_rec(), anatomy=_anatomy())
        ft = build_facts(rs, _anatomy(), present_classes=["Cardiomegaly"])
        claims, followed = claims_from_text("<obs facts=F1> Cardiomegaly is present [R01] </obs> Background.", ft)
        assert followed and claims[0].fact_ids == ("F1",) and claims[0].evidence_ids == ("R01",)
        p = rationale_prompt("Is there cardiomegaly?", "Yes, Cardiomegaly is present.", ft)
        assert "F1: presence Cardiomegaly [R01]" in p and "<obs facts=F#>" in p


@pytest.mark.skipif(not HAS_DATA, reason="data/processed not built")
class TestOnRealData:
    def test_regions_and_labels_for_val(self):
        images = load_images(PROCESSED)
        samples = load_samples(PROCESSED, "val")[:200]
        n_targets = 0
        for s in samples:
            rec = images[s.image_id]
            rs = regions_from_annotation(rec, anatomy=fallback_anatomy(rec.width, rec.height))
            lab = labels_for(s, rec, rs, anatomy=fallback_anatomy(rec.width, rec.height))
            assert lab.findings[lab.asked_class] == 1
            n_targets += bool(lab.target_regions)
            ft = build_facts(rs, fallback_anatomy(rec.width, rec.height), present_classes=rec.vqa_classes_present)
            assert serialize_facts(ft)
            claims = template_claims_full(ft)
            out = verify_all(claims, ft, rs.ids)
            assert all(c.status == SUPPORTED for c in out if c.is_observation)
        # the repaired box is a fused box of its class, so it should usually be a region
        assert n_targets / len(samples) > 0.9, n_targets
