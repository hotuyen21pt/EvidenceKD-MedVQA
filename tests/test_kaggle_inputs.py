"""Model auto-detection in the Kaggle notebook (``notebook_cells.FULL_MODELS``).

Detection runs against whatever happens to be attached to the session, so it is
exercised here against a fake ``/kaggle/input`` tree instead of the real one.

The regression it guards: the teacher directory was "any attached directory
holding a ``config.json``", with ``medgemma`` used only as a *sort* key. A
session with flan-T5 attached for generator S but no MedGemma therefore picked
the flan-T5 as the teacher, and all 30 KD-R shards plus all 30 KD-C shards died
inside ``AutoModelForImageTextToText``::

    ValueError: Unrecognized configuration class
    <class 'transformers.models.t5.configuration_t5.T5Config'>
    for this kind of AutoModel: AutoModelForImageTextToText.
"""

from __future__ import annotations

import functools
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import notebook_cells as nc  # noqa: E402

# What `AutoModelForImageTextToText` accepts and what it does not. Trimmed to
# the keys detection looks at.
MEDGEMMA_CONFIG = {
    "model_type": "gemma3",
    "architectures": ["Gemma3ForConditionalGeneration"],
    "text_config": {"model_type": "gemma3_text"},
    "vision_config": {"model_type": "siglip_vision_model", "image_size": 896},
}
T5_CONFIG = {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"], "d_model": 768}
GEMMA_TEXT_CONFIG = {"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"]}


def model_dir(root, name, config, *, weights=True, extra=()):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if weights:
        (d / "model.safetensors").write_bytes(b"")
    for f in extra:
        (d / f).write_bytes(b"")
    return str(d)


def detect(root, *, medgemma="", adapter="/kaggle/input/adapter", gen_s=""):
    """Run the FULL_MODELS cell over ``root`` as if it were /kaggle/input."""
    ns: dict = {}
    exec(nc.FINDERS, ns)
    ns["find_inputs"] = functools.partial(ns["find_inputs"], root=str(root))
    ns.update(MEDGEMMA=medgemma, ADAPTER=adapter, GEN_S_MODEL=gen_s)
    exec(nc.FULL_MODELS, ns)
    return ns


class TestTeacherDetection:
    def test_a_flan_t5_alone_is_never_taken_for_the_teacher(self, tmp_path):
        # the exact Kaggle attachment that killed 60 shards
        model_dir(tmp_path, "flan-t5-base", T5_CONFIG, extra=("spiece.model",))
        assert detect(tmp_path)["MEDGEMMA"] == ""

    def test_the_teacher_is_found_next_to_a_flan_t5(self, tmp_path):
        model_dir(tmp_path, "flan-t5-base", T5_CONFIG, extra=("spiece.model",))
        want = model_dir(tmp_path, "medgemma-4b-it", MEDGEMMA_CONFIG)
        assert detect(tmp_path)["MEDGEMMA"] == want

    def test_a_vision_model_with_an_unrelated_name_still_counts(self, tmp_path):
        # Kaggle datasets get renamed; the config decides, not the path
        want = model_dir(tmp_path, "cxr-weights", MEDGEMMA_CONFIG)
        assert detect(tmp_path)["MEDGEMMA"] == want

    def test_a_text_only_gemma_is_rejected_too(self, tmp_path):
        # same family, no vision tower -> same ValueError from transformers
        model_dir(tmp_path, "gemma-3-1b-it", GEMMA_TEXT_CONFIG)
        assert detect(tmp_path)["MEDGEMMA"] == ""

    def test_a_vision_model_without_weights_is_not_usable(self, tmp_path):
        model_dir(tmp_path, "medgemma-4b-it", MEDGEMMA_CONFIG, weights=False)
        assert detect(tmp_path)["MEDGEMMA"] == ""

    def test_an_unreadable_config_is_not_usable(self, tmp_path):
        d = tmp_path / "broken"
        d.mkdir()
        (d / "config.json").write_text("{not json", encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"")
        assert detect(tmp_path)["MEDGEMMA"] == ""

    def test_a_path_naming_the_teacher_breaks_a_tie(self, tmp_path):
        model_dir(tmp_path, "aaa-some-vlm", MEDGEMMA_CONFIG)
        want = model_dir(tmp_path, "medgemma-4b-it", MEDGEMMA_CONFIG)
        assert detect(tmp_path)["MEDGEMMA"] == want

    def test_a_hand_set_path_that_is_not_a_vision_model_stops_the_run(self, tmp_path):
        bad = model_dir(tmp_path, "flan-t5-base", T5_CONFIG)
        with pytest.raises(SystemExit) as e:
            detect(tmp_path, medgemma=bad)
        assert bad in str(e.value) and "t5" in str(e.value)

    def test_a_hand_set_teacher_path_is_kept(self, tmp_path):
        want = model_dir(tmp_path, "my-medgemma", MEDGEMMA_CONFIG)
        assert detect(tmp_path, medgemma=want)["MEDGEMMA"] == want


class TestGeneratorSDetection:
    def test_the_flan_t5_still_goes_to_generator_s(self, tmp_path):
        want = model_dir(tmp_path, "flan-t5-base", T5_CONFIG, extra=("spiece.model",))
        model_dir(tmp_path, "medgemma-4b-it", MEDGEMMA_CONFIG)
        ns = detect(tmp_path)
        assert ns["GEN_S_MODEL"] == want
        assert ns["MEDGEMMA"] != want

    def test_generator_s_is_not_filled_with_the_teacher(self, tmp_path):
        model_dir(tmp_path, "medgemma-4b-it", MEDGEMMA_CONFIG)
        assert detect(tmp_path)["GEN_S_MODEL"] == ""
