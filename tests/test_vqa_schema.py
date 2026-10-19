"""Tests for VinDr-CXR-VQA record parsing."""

from __future__ import annotations

import pytest

from evidencekd.data.vqa_schema import (
    LocBox,
    SchemaError,
    VQAItem,
    parse_loc_tokens,
    parse_record,
    strip_loc_tokens,
)


def make_vqa(**overrides):
    base = {
        "question": "Where is the Cardiomegaly?",
        "answer": "Enlargement of the cardiac silhouette <loc_691_1375_1653_1831>.",
        "reason": "Cardiomegaly is a nonspecific finding.",
        "type": "Where",
        "difficulty": "Medium",
        "gt_finding": "Cardiomegaly",
        "gt_location": "<loc_691_1375_1653_1831>",
    }
    base.update(overrides)
    return base


class TestLocParsing:
    def test_parses_a_single_box(self):
        boxes = parse_loc_tokens("centrally <loc_691_1375_1653_1831>.")
        assert boxes == [LocBox(691, 1375, 1653, 1831)]

    def test_parses_multiple_boxes_in_order(self):
        boxes = parse_loc_tokens("<loc_1_2_3_4> and <loc_5_6_7_8>")
        assert [b.as_tuple() for b in boxes] == [(1, 2, 3, 4), (5, 6, 7, 8)]

    def test_no_boxes_is_empty_not_error(self):
        # Negative answers legitimately carry no location.
        assert parse_loc_tokens("No evidence of pneumothorax.") == []

    def test_round_trips_through_token(self):
        box = LocBox(691, 1375, 1653, 1831)
        assert parse_loc_tokens(box.to_token()) == [box]

    @pytest.mark.parametrize(
        "box,valid",
        [
            (LocBox(0, 0, 10, 10), True),
            (LocBox(10, 10, 10, 20), False),   # zero width
            (LocBox(10, 10, 20, 10), False),   # zero height
            (LocBox(20, 10, 10, 20), False),   # inverted
            (LocBox(-5, 10, 20, 20), False),   # negative origin
        ],
    )
    def test_validity(self, box, valid):
        assert box.is_valid() is valid

    def test_strip_tidies_whitespace_before_punctuation(self):
        assert (
            strip_loc_tokens("the silhouette <loc_1_2_3_4>.")
            == "the silhouette."
        )


class TestRecordParsing:
    def test_flattens_one_image_into_items(self):
        record = {"image_id": "abc", "num_questions": 2, "vqa": [make_vqa(), make_vqa()]}
        items = parse_record(record)
        assert len(items) == 2
        assert [i.sample_id for i in items] == ["abc#0", "abc#1"]

    def test_detects_truncated_download(self):
        # num_questions disagreeing with len(vqa) is how a partial download shows up.
        record = {"image_id": "abc", "num_questions": 5, "vqa": [make_vqa()]}
        with pytest.raises(SchemaError, match="num_questions=5 but len"):
            parse_record(record)

    def test_missing_field_raises_in_strict_mode(self):
        bad = make_vqa()
        del bad["reason"]
        record = {"image_id": "abc", "num_questions": 1, "vqa": [bad]}
        with pytest.raises(SchemaError, match="missing fields"):
            parse_record(record)

    def test_non_strict_skips_but_keeps_good_items(self):
        bad = make_vqa()
        del bad["reason"]
        record = {"image_id": "abc", "num_questions": 2, "vqa": [bad, make_vqa()]}
        items = parse_record(record, strict=False)
        assert len(items) == 1

    def test_rejects_missing_image_id(self):
        with pytest.raises(SchemaError, match="image_id"):
            parse_record({"num_questions": 1, "vqa": [make_vqa()]})

    def test_rejects_vqa_that_is_not_a_list(self):
        with pytest.raises(SchemaError, match="must be a list"):
            parse_record({"image_id": "a", "num_questions": 1, "vqa": "nope"})


class TestGrounding:
    def test_item_with_valid_box_is_grounded(self):
        item = parse_record(
            {"image_id": "a", "num_questions": 1, "vqa": [make_vqa()]}
        )[0]
        assert item.is_grounded is True

    def test_negative_case_is_not_grounded(self):
        # Nothing in an image shows an absence; this must never be "grounded".
        item = parse_record(
            {
                "image_id": "a",
                "num_questions": 1,
                "vqa": [
                    make_vqa(
                        answer="No, there is no pneumothorax.",
                        gt_location="",
                        gt_finding="Pneumothorax",
                    )
                ],
            }
        )[0]
        assert item.is_grounded is False
        assert item.gt_boxes == []

    def test_degenerate_box_does_not_count_as_grounded(self):
        item = parse_record(
            {
                "image_id": "a",
                "num_questions": 1,
                "vqa": [make_vqa(gt_location="<loc_10_10_10_20>")],
            }
        )[0]
        assert item.is_grounded is False

    def test_answer_text_drops_location_tokens(self):
        item = parse_record(
            {"image_id": "a", "num_questions": 1, "vqa": [make_vqa()]}
        )[0]
        assert "<loc_" not in item.answer_text
        assert item.answer_text.endswith("silhouette.")
