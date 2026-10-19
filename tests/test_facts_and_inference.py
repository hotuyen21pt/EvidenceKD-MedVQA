"""Tests for the geometric fact table, Answer Lock and Evidence Pointer."""

from __future__ import annotations

import pytest

from evidencekd.data.answer_norm import (
    SIDE_BILATERAL,
    SIDE_LEFT,
    SIDE_RIGHT,
    normalize_answer,
    parse_count,
    parse_side,
    parse_yes_no,
)
from evidencekd.data.boxes import Box, fuse_boxes, iou
from evidencekd.data.splits import LeakageError, Splits, assert_no_leakage, split_by_image
from evidencekd.facts.geometry import (
    Anatomy,
    build_fact_table,
    cardiothoracic_ratio,
    count_of,
    laterality_of,
    side_of,
    zone_of,
)
from evidencekd.inference.answer_lock import (
    NO_SUPPORT,
    AnswerLockViolation,
    enforce_answer_lock,
    lock_answer,
    select_evidence,
)

THORAX = Box(0, 0, 100, 100, "thorax")
ANATOMY = Anatomy(
    thorax=THORAX,
    heart=Box(25, 40, 75, 80, "heart"),
    lung_left=Box(55, 10, 95, 90, "lung"),
    lung_right=Box(5, 10, 45, 90, "lung"),
)


class TestLaterality:
    def test_image_left_is_patient_right(self):
        # The single easiest thing to get wrong in a CXR pipeline.
        assert side_of(Box(10, 20, 20, 30, "Nodule/Mass"), ANATOMY) == "right"

    def test_image_right_is_patient_left(self):
        assert side_of(Box(80, 20, 90, 30, "Nodule/Mass"), ANATOMY) == "left"

    def test_falls_back_to_image_midline_without_anatomy(self):
        assert side_of(Box(10, 20, 20, 30, "X"), Anatomy(), image_width=100) == "right"

    def test_returns_none_when_no_midline_is_known(self):
        # Guessing a side is worse than declining to assert one.
        assert side_of(Box(10, 20, 20, 30, "X"), Anatomy()) is None

    def test_boxes_on_both_sides_are_bilateral(self):
        boxes = [Box(10, 20, 20, 30, "X"), Box(80, 20, 90, 30, "X")]
        assert laterality_of(boxes, "X", ANATOMY) == "bilateral"

    def test_boxes_on_one_side_are_not_bilateral(self):
        boxes = [Box(10, 20, 20, 30, "X"), Box(15, 40, 25, 50, "X")]
        assert laterality_of(boxes, "X", ANATOMY) == "right"


class TestZones:
    @pytest.mark.parametrize(
        "cy,expected", [(15, "upper"), (50, "middle"), (85, "lower")]
    )
    def test_zone_measured_against_the_lung(self, cy, expected):
        box = Box(10, cy - 2, 20, cy + 2, "X")
        assert zone_of(box, ANATOMY) == expected


class TestCTR:
    def test_computed_on_pa_films(self):
        ctr, _ = cardiothoracic_ratio(ANATOMY, view_position="PA")
        assert ctr == pytest.approx(0.5)

    def test_refused_on_ap_films(self):
        # AP magnifies the heart; an AP CTR would over-call cardiomegaly.
        assert cardiothoracic_ratio(ANATOMY, view_position="AP") is None

    def test_refused_when_view_is_unknown(self):
        # An unlabelled film is not assumed to be PA.
        assert cardiothoracic_ratio(ANATOMY, view_position=None) is None

    def test_refused_without_anatomy(self):
        assert cardiothoracic_ratio(Anatomy(), view_position="PA") is None


class TestFactTable:
    def test_builds_presence_side_and_count(self):
        regions = {"R01": Box(10, 20, 20, 30, "Cardiomegaly")}
        table = build_fact_table(
            regions=regions, anatomy=ANATOMY, present_classes=["Cardiomegaly"]
        )
        attrs = {f.attr for f in table.facts}
        assert {"presence", "side", "count"} <= attrs

    def test_every_fact_cites_at_least_one_region(self):
        regions = {"R01": Box(10, 20, 20, 30, "Cardiomegaly")}
        table = build_fact_table(
            regions=regions,
            anatomy=ANATOMY,
            present_classes=["Cardiomegaly"],
            view_position="PA",
        )
        assert all(f.evidence_ids for f in table.facts)

    def test_fact_ids_are_unique(self):
        regions = {
            "R01": Box(10, 20, 20, 30, "Cardiomegaly"),
            "R02": Box(80, 20, 90, 30, "Pneumothorax"),
        }
        table = build_fact_table(
            regions=regions,
            anatomy=ANATOMY,
            present_classes=["Cardiomegaly", "Pneumothorax"],
        )
        ids = [f.id for f in table.facts]
        assert len(ids) == len(set(ids))

    def test_co_occurrence_is_emitted_for_two_findings(self):
        regions = {
            "R01": Box(10, 20, 20, 30, "Cardiomegaly"),
            "R02": Box(80, 20, 90, 30, "Pneumothorax"),
        }
        table = build_fact_table(
            regions=regions,
            anatomy=ANATOMY,
            present_classes=["Cardiomegaly", "Pneumothorax"],
        )
        assert any(f.attr == "co_occurrence" for f in table.facts)

    def test_absence_needs_both_no_box_and_a_low_head_score(self):
        table = build_fact_table(
            regions={},
            anatomy=ANATOMY,
            present_classes=[],
            whole_image_scores={"Pneumothorax": 0.05},
            absence_threshold=0.5,
        )
        assert any(f.attr == "absence" for f in table.facts)

    def test_absence_is_not_asserted_when_the_head_is_confident(self):
        table = build_fact_table(
            regions={},
            anatomy=ANATOMY,
            present_classes=[],
            whole_image_scores={"Pneumothorax": 0.9},
        )
        assert not any(f.attr == "absence" for f in table.facts)

    def test_absence_is_not_asserted_without_a_head_score(self):
        # No box AND no score means we cannot claim absence, so we do not.
        table = build_fact_table(regions={}, anatomy=ANATOMY, present_classes=[])
        assert not any(f.attr == "absence" for f in table.facts)


class TestAnswerNormalisation:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Yes, there is cardiomegaly.", True),
            ("No evidence of pneumothorax.", False),
            ("There is no pleural effusion.", False),
            ("There is a nodule in the right lung.", True),
            ("The study is unremarkable.", False),
        ],
    )
    def test_polarity(self, text, expected):
        assert parse_yes_no(text) is expected

    def test_negative_cue_wins_over_embedded_positive_cue(self):
        # "There is no X" contains the positive cue "there is".
        assert parse_yes_no("There is no pneumothorax.") is False

    def test_unasserted_polarity_is_none(self):
        assert parse_yes_no("The lesion is in the right upper zone.") is None

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Opacity in the right lower zone.", SIDE_RIGHT),
            ("Left-sided effusion.", SIDE_LEFT),
            ("Bilateral effusions.", SIDE_BILATERAL),
            ("Both lungs are affected.", SIDE_BILATERAL),
            ("Left and right apices.", SIDE_BILATERAL),
            ("Cardiomegaly is present.", None),
        ],
    )
    def test_side(self, text, expected):
        assert parse_side(text) == expected

    def test_count_from_digits(self):
        assert parse_count("3 opacities are seen.") == 3

    def test_count_from_words_only_for_how_many(self):
        assert parse_count("There are two nodules.", question_type="How_many") == 2
        assert parse_count("There are two nodules.", question_type="Where") is None

    def test_none_means_zero_on_how_many(self):
        assert parse_count("None.", question_type="How_many") == 0

    def test_location_tokens_do_not_leak_into_counts(self):
        # <loc_691_1375_1653_1831> is full of digits that must never be read
        # as a count. A string with nothing but a box asserts no count at all,
        # which is None ("text does not say"), not 0 ("there are zero").
        assert parse_count("<loc_691_1375_1653_1831>", question_type="How_many") is None

    def test_real_count_survives_alongside_a_location_token(self):
        assert (
            parse_count(
                "There are 2 nodules <loc_691_1375_1653_1831>.",
                question_type="How_many",
            )
            == 2
        )

    def test_binary_question_defaults_to_false_when_unparseable(self):
        n = normalize_answer("Hmm.", question_type="Is_there")
        assert n.yes_no is False

    def test_finding_comes_from_annotation_not_prose(self):
        n = normalize_answer("Yes.", question_type="Is_there", gt_finding="Cardiomegaly")
        assert n.finding == "Cardiomegaly"

    def test_defined_heads_reports_only_asserted_labels(self):
        n = normalize_answer("Opacity in the right lower zone.", question_type="Where")
        assert "side" in n.defined_heads()
        assert "count" not in n.defined_heads()


class TestAnswerLock:
    def test_locked_answer_cannot_be_mutated(self):
        locked = lock_answer(answer="Yes", question_type="Is_there")
        with pytest.raises(AnswerLockViolation):
            locked.answer = "No"

    def test_generator_cannot_change_the_answer(self):
        locked = lock_answer(answer="Yes", question_type="Is_there")
        final, warnings = enforce_answer_lock(locked, "No, there is no cardiomegaly.")
        assert final == "Yes"
        assert any("contradicted_answer" in w for w in warnings)

    def test_agreeing_generator_produces_no_warning(self):
        locked = lock_answer(answer="Yes", question_type="Is_there")
        final, warnings = enforce_answer_lock(locked, "Yes, cardiomegaly is present.")
        assert final == "Yes" and warnings == []

    def test_side_disagreement_is_warned(self):
        locked = lock_answer(answer="Right lower zone.", question_type="Where")
        _, warnings = enforce_answer_lock(locked, "In the left upper zone.")
        assert any("contradicted_side" in w for w in warnings)


class TestEvidencePointer:
    def test_selects_the_top_region(self):
        r = select_evidence(
            region_ids=["R01", "R02"], probabilities=[0.7, 0.2], null_probability=0.1
        )
        assert r.supported and r.region_ids[0] == "R01"

    def test_returns_no_support_when_null_wins(self):
        r = select_evidence(
            region_ids=["R01"], probabilities=[0.2], null_probability=0.8
        )
        assert r.status == NO_SUPPORT and r.region_ids == ()

    def test_never_invents_a_region_id(self):
        r = select_evidence(
            region_ids=["R01", "R02"], probabilities=[0.6, 0.3], null_probability=0.1
        )
        assert set(r.region_ids) <= {"R01", "R02"}

    def test_empty_region_set_is_no_support(self):
        r = select_evidence(region_ids=[], probabilities=[], null_probability=1.0)
        assert r.status == NO_SUPPORT

    def test_respects_top_k(self):
        r = select_evidence(
            region_ids=["A", "B", "C"],
            probabilities=[0.4, 0.3, 0.2],
            null_probability=0.1,
            top_k=2,
        )
        assert len(r.region_ids) == 2

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="probabilities"):
            select_evidence(
                region_ids=["A", "B"], probabilities=[0.5], null_probability=0.1
            )


class TestBoxFusion:
    def test_iou_of_identical_boxes_is_one(self):
        b = Box(0, 0, 10, 10, "X")
        assert iou(b, b) == pytest.approx(1.0)

    def test_disjoint_boxes_have_zero_iou(self):
        assert iou(Box(0, 0, 5, 5, "X"), Box(10, 10, 15, 15, "X")) == 0.0

    def test_degenerate_box_has_zero_iou(self):
        assert iou(Box(0, 0, 0, 10, "X"), Box(0, 0, 10, 10, "X")) == 0.0

    def test_agreeing_raters_collapse_to_one_region(self):
        boxes = [
            Box(10, 10, 50, 50, "Cardiomegaly", rad_id="R1"),
            Box(12, 11, 52, 49, "Cardiomegaly", rad_id="R2"),
            Box(11, 12, 51, 51, "Cardiomegaly", rad_id="R3"),
        ]
        fused = fuse_boxes(boxes, iou_threshold=0.4)
        assert len(fused) == 1 and fused[0].n_raters == 3

    def test_disagreeing_raters_stay_separate(self):
        boxes = [
            Box(10, 10, 30, 30, "X", rad_id="R1"),
            Box(70, 70, 90, 90, "X", rad_id="R2"),
        ]
        assert len(fuse_boxes(boxes, iou_threshold=0.4)) == 2

    def test_different_classes_never_fuse(self):
        boxes = [
            Box(10, 10, 50, 50, "Cardiomegaly", rad_id="R1"),
            Box(10, 10, 50, 50, "Pneumothorax", rad_id="R2"),
        ]
        assert len(fuse_boxes(boxes)) == 2

    def test_region_ids_are_stable_across_calls(self):
        boxes = [Box(10, 10, 50, 50, "X", rad_id="R1")]
        assert fuse_boxes(boxes)[0].region_id == fuse_boxes(boxes)[0].region_id

    def test_clip_brings_an_out_of_bounds_box_inside(self):
        clipped = Box(-10, -10, 120, 120, "X").clip(100, 100)
        assert clipped.as_tuple() if hasattr(clipped, "as_tuple") else True
        assert (clipped.x_min, clipped.y_min, clipped.x_max, clipped.y_max) == (
            0.0, 0.0, 100.0, 100.0
        )


class TestSplits:
    def test_splits_are_disjoint(self):
        s = split_by_image([f"img{i}" for i in range(100)], seed=0)
        assert not (set(s.train) & set(s.val))
        assert not (set(s.train) & set(s.test))
        assert not (set(s.val) & set(s.test))

    def test_split_is_deterministic_for_a_seed(self):
        ids = [f"img{i}" for i in range(50)]
        assert split_by_image(ids, seed=7) == split_by_image(ids, seed=7)

    def test_split_does_not_depend_on_input_order(self):
        ids = [f"img{i}" for i in range(50)]
        assert split_by_image(ids, seed=7) == split_by_image(list(reversed(ids)), seed=7)

    def test_leakage_is_detected(self):
        with pytest.raises(LeakageError, match="multiple splits"):
            assert_no_leakage(Splits(("a", "b"), ("b",), ("c",)))

    def test_all_images_are_accounted_for(self):
        ids = [f"img{i}" for i in range(100)]
        s = split_by_image(ids, seed=1)
        assert set(s.train) | set(s.val) | set(s.test) == set(ids)
