"""Torch-free tests for the stage-0 path: VinDr readers, mAP, the bridge,
config, and the generated notebooks."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from evidencekd.data.boxes import Box
from evidencekd.data.vindr_processed import (VQASample, load_images, load_samples,
                                             splits_from_images)
from evidencekd.eval.detection import Detection, average_precision, mean_average_precision
from evidencekd.eval.vindr_bridge import (compose, detector_rows, grounding_success,
                                          loc_token, oracle_rows, to_norm1000)
from evidencekd.labels import VQA_CLASSES, finding_in_text
from evidencekd.utils.config import _mini_yaml, load_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(REPO, "data", "processed")
HAS_DATA = os.path.isfile(os.path.join(PROCESSED, "train.jsonl"))


# ---------------------------------------------------------------------------
class TestLabels:
    def test_fourteen_classes(self):
        assert len(VQA_CLASSES) == 14 and len(set(VQA_CLASSES)) == 14

    def test_matches_sibling_constants(self):
        vindr = pytest.importorskip("vindr_cxr_vqa.constants")
        assert tuple(vindr.PATHOLOGIES) == VQA_CLASSES
        for cls, syns in vindr.PATHOLOGY_SYNONYMS.items():
            from evidencekd.labels import VQA_SYNONYMS
            assert tuple(syns) == VQA_SYNONYMS[cls]

    @pytest.mark.parametrize("q,cls", [
        ("Where is the Pleural effusion?", "Pleural effusion"),
        ("Is there lung opacity?", "Lung Opacity"),
        ("Does this X-ray show a nodule?", "Nodule/Mass"),
        ("How many findings are there?", None),
    ])
    def test_finding_in_text(self, q, cls):
        assert finding_in_text(q) == cls


# ---------------------------------------------------------------------------
def _sample(sid="img1_0", finding="Cardiomegaly", box=(100, 100, 300, 300), qtype="Where",
            question=None, W=1000, H=1000):
    return VQASample(sample_id=sid, image_id=sid.split("_")[0], image_path="x.png", split="val",
                     question=question or f"Where is the {finding}?", answer="", reason="",
                     type=qtype, difficulty="Easy", gt_finding=finding, bbox_orig=box,
                     bbox_fixed_orig=box, repair_method="published", box_label_consistent=True,
                     image_width=W, image_height=H, resized_width=224, resized_height=224)


class TestBridge:
    def test_norm1000_matches_vindr(self):
        vindr = pytest.importorskip("vindr_cxr_vqa.boxes")
        b = Box(123.4, 56.7, 890.1, 999.9, class_name="x")
        assert to_norm1000(b, 1024, 1024) == tuple(int(v) for v in vindr.to_normalized((123.4, 56.7, 890.1, 999.9), 1024, 1024))

    def test_loc_token_is_three_digit(self):
        assert loc_token(Box(0, 0, 10, 10, class_name="x"), 1000, 1000) == "<loc_000_000_010_010>"

    def test_compose_without_box_has_no_loc(self):
        assert compose("Cardiomegaly", []) == "Cardiomegaly."
        assert "<loc_" not in compose("", [])

    def test_oracle_rows_replay_answer_key(self):
        s = _sample()
        r = oracle_rows([s])[0]
        assert r.prediction == "Cardiomegaly <loc_100_100_300_300>."
        assert r.gt_location == "<loc_100_100_300_300>"

    def test_detector_rows_pick_named_class(self):
        s = _sample()
        dets = {"img1": [Detection(Box(0, 0, 50, 50, class_name="Pleural effusion"), 0.9),
                         Detection(Box(110, 110, 290, 290, class_name="Cardiomegaly"), 0.6)]}
        r = detector_rows([s], dets, score_threshold=0.3)[0]
        assert r.prediction.startswith("Cardiomegaly <loc_110_110_290_290>")

    def test_detector_rows_list_all_for_what(self):
        s = _sample(qtype="What", question="What findings are there?")
        dets = {"img1": [Detection(Box(0, 0, 50, 50, class_name="Pleural effusion"), 0.9),
                         Detection(Box(110, 110, 290, 290, class_name="Cardiomegaly"), 0.6)]}
        r = detector_rows([s], dets, score_threshold=0.3)[0]
        assert "Cardiomegaly" in r.prediction and "Pleural effusion" in r.prediction
        assert r.prediction.count("<loc_") == 2

    def test_detector_rows_below_threshold_emits_no_box(self):
        s = _sample()
        dets = {"img1": [Detection(Box(110, 110, 290, 290, class_name="Cardiomegaly"), 0.1)]}
        r = detector_rows([s], dets, score_threshold=0.3)[0]
        assert r.prediction == "Cardiomegaly."

    def test_grounding_success(self):
        s = _sample()
        hit = {"img1": [Detection(Box(100, 100, 300, 300, class_name="Cardiomegaly"), 0.9)]}
        miss = {"img1": [Detection(Box(500, 500, 600, 600, class_name="Cardiomegaly"), 0.9)]}
        assert grounding_success([s], hit)["grounding_at_iou"] == 1.0
        assert grounding_success([s], miss)["grounding_at_iou"] == 0.0
        assert grounding_success([s], {})["miou"] == 0.0


# ---------------------------------------------------------------------------
class TestMAP:
    def test_perfect_detections(self):
        gt = {"a": [Box(0, 0, 10, 10, class_name="c"), Box(20, 20, 30, 30, class_name="c")]}
        det = {"a": [Detection(b, 0.9) for b in gt["a"]]}
        assert average_precision(det, gt, "c", 0.5) == pytest.approx(1.0)

    def test_half_recall(self):
        gt = {"a": [Box(0, 0, 10, 10, class_name="c"), Box(20, 20, 30, 30, class_name="c")]}
        det = {"a": [Detection(Box(0, 0, 10, 10, class_name="c"), 0.9)]}
        assert average_precision(det, gt, "c", 0.5) == pytest.approx(0.5)

    def test_false_positive_ranked_first_lowers_ap(self):
        gt = {"a": [Box(0, 0, 10, 10, class_name="c")]}
        det = {"a": [Detection(Box(50, 50, 60, 60, class_name="c"), 0.95),
                     Detection(Box(0, 0, 10, 10, class_name="c"), 0.5)]}
        assert average_precision(det, gt, "c", 0.5) == pytest.approx(0.5)

    def test_absent_class_is_none_not_zero(self):
        gt = {"a": [Box(0, 0, 10, 10, class_name="c")]}
        assert average_precision({}, gt, "d", 0.5) is None
        r = mean_average_precision({}, gt, ["c", "d"])
        assert r["per_class_ap50"]["d"] is None and r["classes_scored"] == 1
        assert r["map_50"] == 0.0

    def test_one_to_one_matching(self):
        gt = {"a": [Box(0, 0, 10, 10, class_name="c")]}
        det = {"a": [Detection(Box(0, 0, 10, 10, class_name="c"), 0.9),
                     Detection(Box(0, 0, 10, 10, class_name="c"), 0.8)]}
        # second duplicate is a FP: precision drops to 0.5 after recall is already 1
        assert average_precision(det, gt, "c", 0.5) == pytest.approx(1.0)
        r = mean_average_precision(det, gt, ["c"])
        assert r["n_detections"] == 2


# ---------------------------------------------------------------------------
class TestConfig:
    def test_mini_yaml_nested(self):
        d = _mini_yaml("a:\n  b: 1\n  c: [x, y]  # comment\n  d: 'q#r'\ne: true\n")
        assert d == {"a": {"b": 1, "c": ["x", "y"], "d": "q#r"}, "e": True}

    def test_detector_config_loads_and_resolves_paths(self):
        cfg = load_config(os.path.join(REPO, "configs", "detector.yaml"))
        assert os.path.isabs(cfg.paths.processed_dir)
        assert cfg.model.min_size == 512
        assert cfg.train.lr_milestones == [6]


# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_DATA, reason="data/processed not built")
class TestProcessedData:
    def test_counts_match_build_summary(self):
        summary = json.load(open(os.path.join(PROCESSED, "build_summary.json")))
        for split in ("train", "val", "test"):
            assert len(load_samples(PROCESSED, split)) == summary["qa_per_split"][split]

    def test_splits_match_vindr_table3(self):
        s = splits_from_images(load_images(PROCESSED))
        assert s.sizes == {"train": 3735, "val": 659, "test": 300, "tune": 359}
        assert set(s.test) <= set(s.val)
        assert not (set(s.train) & set(s.val))
        assert s.held_out == frozenset(s.val)

    def test_split_identical_to_sibling_repo(self):
        sib = os.path.join(REPO, "..", "VinDR-CXR-VQA", "data", "processed")
        if not os.path.isfile(os.path.join(sib, "images.json")):
            pytest.skip("sibling build not present")
        ours = splits_from_images(load_images(PROCESSED))
        theirs = splits_from_images(load_images(sib))
        assert ours.as_dict() == theirs.as_dict()

    def test_held_out_assertion(self):
        from evidencekd.data.splits import LeakageError
        s = splits_from_images(load_images(PROCESSED))
        s.assert_held_out(s.train[:10], stage="x")
        with pytest.raises(LeakageError):
            s.assert_held_out([s.test[0]], stage="x")

    def test_every_sample_has_supervision_box_of_its_finding(self):
        for s in load_samples(PROCESSED, "val"):
            assert s.gt_box is not None and s.gt_box.class_name == s.gt_finding
            assert s.gt_box.area > 0

    def test_fused_boxes_carry_votes(self):
        images = load_images(PROCESSED)
        rec = next(iter(images.values()))
        assert all(f.n_raters >= 1 for f in rec.boxes_fused)

    def test_oracle_bridge_scores_perfectly_with_vindr_scorer(self, tmp_path):
        pytest.importorskip("vindr_cxr_vqa.evaluate")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(REPO, "src"),
                                             os.path.join(REPO, "..", "VinDR-CXR-VQA"),
                                             env.get("PYTHONPATH", "")])
        r = subprocess.run([sys.executable, "-m", "evidencekd.eval.stage0", "--config",
                            os.path.join(REPO, "configs", "detector.yaml"), "--oracle",
                            "--split", "test", "--out-dir", str(tmp_path)],
                           env=env, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        rep = json.load(open(tmp_path / "stage0_test_oracle.json"))
        assert rep["vindr_tables"]["vqa"]["precision"] == 1.0
        assert rep["vindr_tables"]["localization"]["iou_ge_0.5_pct"] == 100.0


# ---------------------------------------------------------------------------
class TestNotebooks:
    def test_generator_matches_disk(self):
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "make_notebooks.py"), "--check"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout

    @pytest.mark.parametrize("name", ["kaggle_stage0_detector", "kaggle_stage0_predict"])
    def test_cells_parse(self, name):
        nb = json.load(open(os.path.join(REPO, "notebooks", f"{name}.ipynb"), encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{name}[{i}]", "exec")
