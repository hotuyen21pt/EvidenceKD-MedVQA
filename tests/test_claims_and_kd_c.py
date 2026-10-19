"""Tests for claim grammar, citation constraints and KD-C filtering."""

from __future__ import annotations

import pytest

from evidencekd.claims.parser import (
    CONTRADICTED,
    INSUFFICIENT,
    SUPPORTED,
    Claim,
    allowed_citations,
    can_close_observation,
    parse_claims,
    render_claims,
    validate_claim,
)
from evidencekd.kd.kd_c import FilterReport, build_targets, filter_rationale

FACT_TABLE = {"F3": ["R01", "A_thorax"], "F4": ["R02"]}
KNOWN_REGIONS = ["R01", "R02", "A_thorax", "A_heart"]


def obs(text="The cardiac silhouette is enlarged.", facts=("F3",), cites=("R01",)):
    return Claim(type="observation", text=text, fact_ids=facts, evidence_ids=cites)


class TestParsing:
    def test_parses_an_observation_claim(self):
        claims = parse_claims("<obs facts=F3> Heart is enlarged. [R01] </obs>")
        assert len(claims) == 1
        c = claims[0]
        assert c.type == "observation"
        assert c.fact_ids == ("F3",)
        assert c.evidence_ids == ("R01",)
        assert "[R01]" not in c.text

    def test_parses_multiple_facts_and_citations(self):
        claims = parse_claims("<obs facts=F3,F4> Both. [R01] [R02] </obs>")
        assert claims[0].fact_ids == ("F3", "F4")
        assert claims[0].evidence_ids == ("R01", "R02")

    def test_deduplicates_repeated_citations(self):
        claims = parse_claims("<obs facts=F3> X. [R01] [R01] </obs>")
        assert claims[0].evidence_ids == ("R01",)

    def test_text_outside_obs_becomes_background(self):
        claims = parse_claims(
            "Cardiomegaly is often chronic. <obs facts=F3> Enlarged. [R01] </obs>"
        )
        assert [c.type for c in claims] == ["background", "observation"]

    def test_background_splits_on_sentences(self):
        claims = parse_claims("First fact. Second fact.")
        assert len(claims) == 2
        assert all(c.type == "background" for c in claims)

    def test_round_trips(self):
        original = "<obs facts=F3> Heart is enlarged. [R01] </obs>"
        assert parse_claims(render_claims(parse_claims(original))) == parse_claims(
            original
        )


class TestConstrainedDecoding:
    def test_allowed_citations_come_from_declared_facts(self):
        assert allowed_citations(["F3"], FACT_TABLE) == {"R01", "A_thorax"}

    def test_undeclared_fact_permits_nothing(self):
        assert allowed_citations(["F9"], FACT_TABLE) == frozenset()

    def test_cannot_close_before_citing(self):
        assert can_close_observation("Heart is enlarged.") is False

    def test_can_close_after_citing(self):
        assert can_close_observation("Heart is enlarged. [R01]") is True


class TestValidation:
    def test_valid_claim_passes_unchanged(self):
        c = validate_claim(obs(), fact_table=FACT_TABLE, known_region_ids=KNOWN_REGIONS)
        assert c.reject_reason is None

    def test_observation_without_citation_is_rejected(self):
        c = validate_claim(
            obs(cites=()), fact_table=FACT_TABLE, known_region_ids=KNOWN_REGIONS
        )
        assert c.status == INSUFFICIENT
        assert "no citation" in c.reject_reason

    def test_hallucinated_region_is_rejected(self):
        c = validate_claim(
            obs(cites=("R99",)),
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
        )
        assert "unknown region" in c.reject_reason

    def test_borrowed_evidence_is_rejected(self):
        # R02 exists, but fact F3 does not reference it.
        c = validate_claim(
            obs(facts=("F3",), cites=("R02",)),
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
        )
        assert "not supported by declared facts" in c.reject_reason

    def test_undeclared_fact_is_rejected(self):
        c = validate_claim(
            obs(facts=("F9",)), fact_table=FACT_TABLE, known_region_ids=KNOWN_REGIONS
        )
        assert "undeclared fact" in c.reject_reason

    def test_background_claims_are_not_validated(self):
        c = Claim(type="background", text="Cardiomegaly is often chronic.")
        assert validate_claim(
            c, fact_table=FACT_TABLE, known_region_ids=KNOWN_REGIONS
        ).reject_reason is None


def always(status):
    return lambda claim: status


class TestKDCFiltering:
    def test_supported_claim_is_kept(self):
        r = filter_rationale(
            [obs()],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(SUPPORTED),
        )
        assert r.usable and len(r.kept_claims) == 1

    @pytest.mark.parametrize("status", [CONTRADICTED, INSUFFICIENT])
    def test_non_supported_claims_are_dropped(self, status):
        r = filter_rationale(
            [obs()],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(status),
        )
        assert not r.usable and len(r.kept_claims) == 0

    def test_insufficient_is_never_promoted_to_supported(self):
        r = filter_rationale(
            [obs()],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(INSUFFICIENT),
        )
        assert all(c.status != SUPPORTED for c in r.kept_claims + r.dropped_claims)

    def test_claim_granularity_keeps_good_siblings(self):
        good, bad = obs(text="Good."), obs(text="Bad.", cites=("R99",))
        r = filter_rationale(
            [good, bad],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(SUPPORTED),
            granularity="claim",
        )
        assert r.usable
        assert [c.text for c in r.kept_claims] == ["Good."]

    def test_sample_granularity_discards_the_whole_rationale(self):
        # This is the control that claim-level filtering must beat.
        good, bad = obs(text="Good."), obs(text="Bad.", cites=("R99",))
        r = filter_rationale(
            [good, bad],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(SUPPORTED),
            granularity="sample",
        )
        assert not r.usable and r.kept_claims == ()

    def test_none_granularity_skips_the_verifier(self):
        r = filter_rationale(
            [obs()],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(CONTRADICTED),
            granularity="none",
        )
        assert r.usable and len(r.kept_claims) == 1

    def test_none_granularity_still_rejects_malformed_citations(self):
        # Unfiltered means "no verifier", not "no grammar".
        r = filter_rationale(
            [obs(cites=("R99",))],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(SUPPORTED),
            granularity="none",
        )
        assert not r.usable

    def test_background_claims_survive_filtering(self):
        bg = Claim(type="background", text="Often chronic.")
        r = filter_rationale(
            [obs(), bg],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(SUPPORTED),
        )
        assert bg in r.kept_claims


class TestReporting:
    def test_report_counts_retention(self):
        report = FilterReport()
        filter_rationale(
            [obs(text="A."), obs(text="B.")],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=lambda c: SUPPORTED if c.text == "A." else CONTRADICTED,
            report=report,
        )
        assert report.total_observations == 2
        assert report.kept == 1
        assert report.retention == pytest.approx(0.5)
        assert report.dropped_by_status[CONTRADICTED] == 1

    def test_report_separates_format_from_verifier_drops(self):
        report = FilterReport()
        filter_rationale(
            [obs(cites=("R99",))],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(SUPPORTED),
            report=report,
        )
        assert sum(report.dropped_by_format.values()) == 1
        assert sum(report.dropped_by_status.values()) == 0


class TestBuildTargets:
    def test_deduplicates_across_the_three_rationales(self):
        # The teacher generates 3 rationales; identical observations must not
        # be repeated three times (Table C scores concision).
        targets = build_targets(
            [[obs()], [obs()], [obs()]],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(SUPPORTED),
        )
        assert len(targets) == 1

    def test_keeps_distinct_claims_from_different_rationales(self):
        targets = build_targets(
            [[obs(text="A.")], [obs(text="B.")]],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(SUPPORTED),
        )
        assert {c.text for c in targets} == {"A.", "B."}

    def test_all_rationales_failing_yields_no_targets(self):
        targets = build_targets(
            [[obs()], [obs()]],
            fact_table=FACT_TABLE,
            known_region_ids=KNOWN_REGIONS,
            verifier=always(CONTRADICTED),
        )
        assert targets == []
