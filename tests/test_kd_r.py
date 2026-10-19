"""Tests for KD-R counterfactual region distillation (spec §4.4)."""

from __future__ import annotations

import math

import pytest

from evidencekd.kd.kd_r import (
    KDRError,
    counterfactual_relevance,
    gate_weight,
    kl_divergence,
    target_mass,
    teacher_distribution,
    weighted_kd_r,
)


class TestRelevance:
    def test_drop_in_logprob_is_the_relevance(self):
        assert counterfactual_relevance(-0.5, -2.0) == pytest.approx(1.5)

    def test_negative_drop_is_clipped_to_zero(self):
        # Masking made the teacher MORE confident: that region is not evidence
        # for the answer, and must not get negative mass.
        assert counterfactual_relevance(-2.0, -0.5) == 0.0

    def test_identical_logprobs_give_zero(self):
        assert counterfactual_relevance(-1.0, -1.0) == 0.0

    @pytest.mark.parametrize(
        "orig,masked",
        [(float("nan"), -1.0), (-1.0, float("inf")), (float("-inf"), -1.0)],
    )
    def test_non_finite_inputs_raise(self, orig, masked):
        with pytest.raises(KDRError, match="non-finite"):
            counterfactual_relevance(orig, masked)


class TestTeacherDistribution:
    def test_returns_m_plus_one_entries(self):
        assert len(teacher_distribution([1.0, 2.0, 3.0])) == 4

    def test_sums_to_one(self):
        p = teacher_distribution([0.3, 1.7, 0.0], r_null=0.5, tau=0.7)
        assert sum(p) == pytest.approx(1.0)

    def test_all_zero_relevance_is_defined_not_a_division_by_zero(self):
        # This is exactly the case a plain r_i/sum(r_j) normalisation cannot
        # handle. The null slot is why the spec's softmax form is used.
        p = teacher_distribution([0.0, 0.0, 0.0], r_null=0.0)
        assert sum(p) == pytest.approx(1.0)
        assert all(x == pytest.approx(0.25) for x in p)

    def test_higher_relevance_gets_more_mass(self):
        p = teacher_distribution([5.0, 0.0], r_null=0.0)
        assert p[0] > p[1]

    def test_null_slot_is_last(self):
        p = teacher_distribution([0.0, 0.0], r_null=10.0)
        assert p[-1] > p[0] and p[-1] > p[1]

    def test_lower_temperature_sharpens(self):
        soft = teacher_distribution([2.0, 1.0], r_null=0.0, tau=2.0)
        sharp = teacher_distribution([2.0, 1.0], r_null=0.0, tau=0.2)
        assert sharp[0] > soft[0]

    def test_large_relevance_does_not_overflow(self):
        p = teacher_distribution([1000.0, 0.0], r_null=0.0)
        assert sum(p) == pytest.approx(1.0)
        assert all(math.isfinite(x) for x in p)

    def test_empty_region_set_raises(self):
        with pytest.raises(KDRError, match="empty region set"):
            teacher_distribution([])

    def test_non_positive_temperature_raises(self):
        with pytest.raises(KDRError, match="tau must be positive"):
            teacher_distribution([1.0], tau=0.0)


class TestTargetMass:
    def test_sums_mass_over_target_regions(self):
        assert target_mass([0.5, 0.2, 0.3], [0, 1]) == pytest.approx(0.7)

    def test_no_targets_is_zero(self):
        assert target_mass([0.5, 0.2, 0.3], []) == 0.0

    def test_null_slot_cannot_be_a_target(self):
        # index 2 is the null slot for a 2-region distribution
        with pytest.raises(KDRError, match="null slot is not a target"):
            target_mass([0.5, 0.2, 0.3], [2])


class TestGate:
    def test_passing_sample_weight_is_the_confidence(self):
        g = gate_weight(teacher_correct=True, mass=0.7, confidence=0.9, delta=0.5)
        assert g.passed and g.weight == pytest.approx(0.9)

    def test_wrong_teacher_is_gated_out(self):
        g = gate_weight(teacher_correct=False, mass=0.9, confidence=0.9, delta=0.5)
        assert not g.passed and g.weight == 0.0
        assert g.reason == "teacher_wrong"

    def test_low_target_mass_is_gated_out(self):
        g = gate_weight(teacher_correct=True, mass=0.1, confidence=0.9, delta=0.5)
        assert not g.passed and g.reason == "mass_below_delta"

    def test_mass_exactly_at_delta_passes(self):
        g = gate_weight(teacher_correct=True, mass=0.5, confidence=0.9, delta=0.5)
        assert g.passed

    def test_ablation_disabling_correctness_keeps_a_wrong_teacher(self):
        # Table E "KD-R không chọn lọc" needs every gate component switchable.
        g = gate_weight(
            teacher_correct=False,
            mass=0.9,
            confidence=1.0,
            delta=0.5,
            use_correctness=False,
        )
        assert g.passed and g.weight == pytest.approx(1.0)

    def test_ablation_mass_only(self):
        g = gate_weight(
            teacher_correct=False,
            mass=0.9,
            confidence=0.4,
            delta=0.5,
            use_correctness=False,
            use_confidence=False,
        )
        assert g.passed and g.weight == pytest.approx(1.0)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(KDRError, match="confidence"):
            gate_weight(teacher_correct=True, mass=0.9, confidence=1.5, delta=0.5)


class TestKL:
    def test_identical_distributions_give_zero(self):
        assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_is_non_negative(self):
        assert kl_divergence([0.7, 0.3], [0.5, 0.5]) > 0

    def test_is_asymmetric(self):
        p, q = [0.9, 0.1], [0.5, 0.5]
        assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))

    def test_zero_in_target_contributes_nothing(self):
        # 0 log 0 limit, not a nan
        assert math.isfinite(kl_divergence([1.0, 0.0], [0.5, 0.5]))

    def test_zero_in_student_against_nonzero_target_is_infinite(self):
        assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf")

    def test_length_mismatch_raises(self):
        with pytest.raises(KDRError, match="length mismatch"):
            kl_divergence([0.5, 0.5], [1.0])


class TestWeightedLoss:
    def test_weights_the_mean_by_gate(self):
        targets = [[1.0, 0.0], [0.0, 1.0]]
        students = [[0.5, 0.5], [0.5, 0.5]]
        # Both samples have identical KL; weighting cannot change the mean.
        equal = weighted_kd_r(targets, students, [1.0, 1.0])
        single = weighted_kd_r(targets, students, [1.0, 0.0])
        assert equal == pytest.approx(single, rel=1e-6)

    def test_gated_out_sample_does_not_contribute(self):
        targets = [[1.0, 0.0], [1.0, 0.0]]
        students = [[0.9, 0.1], [0.1, 0.9]]
        # Second sample has a much larger KL but weight 0.
        loss = weighted_kd_r(targets, students, [1.0, 0.0])
        assert loss == pytest.approx(kl_divergence(targets[0], students[0]), rel=1e-6)

    def test_all_gated_out_is_zero_not_an_error(self):
        # A fully-gated batch contributes no gradient. That is correct.
        assert weighted_kd_r([[1.0, 0.0]], [[0.5, 0.5]], [0.0]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(KDRError, match="equal length"):
            weighted_kd_r([[1.0, 0.0]], [[0.5, 0.5]], [1.0, 1.0])


class TestEndToEnd:
    def test_a_region_that_matters_dominates_the_target(self):
        # Teacher is confident; masking region 0 destroys the answer,
        # masking region 1 does nothing.
        logp_orig = -0.1
        r = [
            counterfactual_relevance(logp_orig, -4.0),
            counterfactual_relevance(logp_orig, -0.1),
        ]
        pi = teacher_distribution(r, r_null=0.0, tau=1.0)
        assert pi[0] > pi[1]
        assert target_mass(pi, [0]) > 0.5

    def test_teacher_insensitive_to_masking_puts_mass_on_null(self):
        # Spec §8 stopping rule: if the teacher is not sensitive to masking,
        # KD-R has nothing to transfer. That shows up as mass on the null slot.
        r = [counterfactual_relevance(-0.1, -0.1) for _ in range(4)]
        pi = teacher_distribution(r, r_null=0.0)
        assert target_mass(pi, [0]) < 0.5
        g = gate_weight(
            teacher_correct=True, mass=target_mass(pi, [0]), confidence=0.9, delta=0.5
        )
        assert not g.passed
