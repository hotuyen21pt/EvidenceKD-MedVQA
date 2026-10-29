"""Tests for the MedGemma baseline rows — B4 (fine-tuned) and pretrained.

These are the two rows the spec compares the proposed system against
(``docs/method_spec_v4.1.md`` §B4), and they must be produced on the *same*
split as the proposed system, not read off the paper's val-659 constants.
"""

from __future__ import annotations

import json

import pytest

from evidencekd.data.vindr_processed import VQASample
from evidencekd.eval.medgemma_baseline import baseline_rows, done_sample_ids


def sample(sid: str, *, box=(10.0, 20.0, 110.0, 120.0), finding="Cardiomegaly") -> VQASample:
    return VQASample(
        sample_id=sid, image_id=f"img_{sid}", image_path=f"train_data/train/img_{sid}.png",
        split="test", question=f"Is there {finding}?", answer=finding, reason="", type="finding",
        difficulty="easy", gt_finding=finding, bbox_orig=box, bbox_fixed_orig=box,
        repair_method="none", box_label_consistent=True,
        image_width=1000, image_height=1000, resized_width=224, resized_height=224)


class FakeGen:
    """Stands in for ``MedGemmaTeacher.generate``; records the batches it saw."""

    def __init__(self, replies=None):
        self.batches: list[list[str]] = []
        self.replies = replies

    def __call__(self, images, questions):
        qs = list(questions)
        self.batches.append(qs)
        if self.replies is not None:
            return [self.replies[q] for q in qs]
        return [f"Cardiomegaly <loc_010_020_110_120>. for {q}" for q in qs]


class TestBaselineRows:
    def test_one_row_per_sample_in_order(self):
        samples = [sample("a"), sample("b"), sample("c")]
        rows = baseline_rows(samples, lambda s: object(), FakeGen())
        assert [r.sample_id for r in rows] == ["a", "b", "c"]

    def test_prediction_is_the_model_text_verbatim(self):
        gen = FakeGen({"Is there Cardiomegaly?": "  Cardiomegaly <loc_001_002_003_004>.  "})
        rows = baseline_rows([sample("a")], lambda s: object(), gen)
        assert rows[0].prediction == "Cardiomegaly <loc_001_002_003_004>."

    def test_ground_truth_fields_come_from_the_sample_not_the_model(self):
        rows = baseline_rows([sample("a", finding="Pneumothorax")], lambda s: object(), FakeGen())
        assert rows[0].gt_finding == "Pneumothorax"
        # norm1000: 999-scaled then rounded, so 10 -> 010 and 110 -> 110
        assert rows[0].gt_location == "<loc_010_020_110_120>"

    def test_a_sample_without_a_box_has_no_gt_location(self):
        s = sample("a")
        s = VQASample(**{**s.__dict__, "bbox_orig": None, "bbox_fixed_orig": None})
        rows = baseline_rows([s], lambda s: object(), FakeGen())
        assert rows[0].gt_location == ""

    def test_questions_are_sent_in_batches(self):
        gen = FakeGen()
        baseline_rows([sample(str(i)) for i in range(5)], lambda s: object(), gen, batch_size=2)
        assert [len(b) for b in gen.batches] == [2, 2, 1]

    def test_an_empty_reply_still_yields_a_row(self):
        # the scorer's with_loc_pct must be able to see a declined answer
        gen = FakeGen({"Is there Cardiomegaly?": ""})
        rows = baseline_rows([sample("a")], lambda s: object(), gen)
        assert len(rows) == 1 and rows[0].prediction == ""

    def test_images_are_loaded_lazily_one_batch_at_a_time(self):
        opened: list[str] = []

        def loader(s):
            opened.append(s.sample_id)
            return object()

        gen = FakeGen()
        baseline_rows([sample(str(i)) for i in range(4)], loader, gen, batch_size=2)
        assert opened == ["0", "1", "2", "3"]

    def test_skip_ids_are_not_sent_to_the_model(self):
        gen = FakeGen()
        rows = baseline_rows([sample("a"), sample("b")], lambda s: object(), gen, skip_ids={"a"})
        assert [r.sample_id for r in rows] == ["b"]
        assert gen.batches == [["Is there Cardiomegaly?"]]

    def test_skipping_everything_calls_the_model_not_at_all(self):
        gen = FakeGen()
        rows = baseline_rows([sample("a")], lambda s: object(), gen, skip_ids={"a"})
        assert rows == [] and gen.batches == []


class TestResume:
    def test_done_ids_of_a_missing_file_is_empty(self, tmp_path):
        assert done_sample_ids(str(tmp_path / "nope.jsonl")) == set()

    def test_done_ids_reads_back_what_was_written(self, tmp_path):
        p = tmp_path / "preds.jsonl"
        p.write_text(json.dumps({"sample_id": "a"}) + "\n" + json.dumps({"sample_id": "b"}) + "\n",
                     encoding="utf-8")
        assert done_sample_ids(str(p)) == {"a", "b"}

    def test_a_truncated_last_line_is_ignored(self, tmp_path):
        # Kaggle kills the session mid-write; the resume must not die on it
        p = tmp_path / "preds.jsonl"
        p.write_text(json.dumps({"sample_id": "a"}) + '\n{"sample_id": "b"', encoding="utf-8")
        assert done_sample_ids(str(p)) == {"a"}


class TestAdapterChoice:
    def test_pretrained_tag_means_no_adapter(self):
        from evidencekd.eval.medgemma_baseline import adapter_for

        assert adapter_for("pretrained", "/some/adapter") is None

    def test_b4_tag_uses_the_configured_adapter(self):
        from evidencekd.eval.medgemma_baseline import adapter_for

        assert adapter_for("b4", "/some/adapter") == "/some/adapter"

    def test_b4_without_an_adapter_is_refused_up_front(self):
        from evidencekd.eval.medgemma_baseline import adapter_for

        with pytest.raises(SystemExit, match="adapter"):
            adapter_for("b4", "")
