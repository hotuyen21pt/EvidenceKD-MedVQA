"""Pure helpers of ``scripts/make_pipeline_pdf.py``.

The document itself needs ``data/processed`` and the VinDr PNGs, so it is not
built here. These are the parts that turn numbers into what the reader sees --
where a formatting slip is silent and ends up printed on every page.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import make_pipeline_pdf as mp  # noqa: E402


class TestVietnameseNumbers:
    """'.' groups thousands and ',' marks the decimal -- the opposite of C."""

    @pytest.mark.parametrize("value,dec,expected", [
        (2045.33, 0, "2.045"),      # the regression: used to print "2.045,"
        (2045.33, 1, "2.045,3"),
        (2500, None, "2.500"),
        (0.7157, 4, "0,7157"),
        (0.0896, 4, "0,0896"),
        (13634.4, None, "13.634"),
        (0.5, 1, "0,5"),
    ])
    def test_separators(self, value, dec, expected):
        assert mp.vn(value, dec) == expected

    def test_zero_decimals_never_leaves_a_dangling_comma(self):
        for v in (0.4, 1.5, 999.99, 2045.33):
            assert not mp.vn(v, 0).endswith(",")

    def test_a_million_groups_every_three_digits(self):
        assert mp.vn(1234567) == "1.234.567"


class TestDistributions:
    """The document draws these; wrong normalisation is invisible in a bar."""

    def test_softmax_sums_to_one(self):
        out = mp.softmax([0.53, 0.02, 0.0, 0.08, 0.0])
        assert math.isclose(sum(out), 1.0, abs_tol=1e-12)

    def test_a_lower_temperature_sharpens(self):
        xs = [0.53, 0.02, 0.0, 0.08, 0.0]
        assert mp.softmax(xs, 0.2)[0] > mp.softmax(xs, 1.0)[0]

    def test_the_kd_r_example_crosses_the_gate_only_at_the_low_temperature(self):
        # the point section 8 of the document makes: delta = 0.5 decides this
        logp = mp.ILLUSTRATIVE["teacher_logp"]
        r = [max(0.0, logp["orig"] - logp[f"R0{i}"]) for i in (1, 2, 3, 4)] + [0.0]
        assert mp.softmax(r, 1.0)[0] < 0.5
        assert mp.softmax(r, 0.2)[0] >= 0.5

    def test_masking_that_helps_the_teacher_is_clipped_to_zero(self):
        logp = mp.ILLUSTRATIVE["teacher_logp"]
        assert logp["R03"] > logp["orig"], "R03 is the example where masking raises confidence"
        assert max(0.0, logp["orig"] - logp["R03"]) == 0.0

    def test_sigmoid_matches_the_relevance_logits_shown(self):
        assert math.isclose(mp.sigmoid(2.1), 0.8909031788043871, rel_tol=1e-9)
