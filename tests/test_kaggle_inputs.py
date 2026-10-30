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


class TestKaggleInputLayouts:
    """Kaggle nests inputs differently per account and per input kind.

    Observed here: datasets mount as `/kaggle/input/datasets/<owner>/<slug>/`
    rather than `/kaggle/input/<slug>/`. Nothing in the finders hardcodes a
    level, but they do carry a depth budget, and exhausting it is silent --
    the input just looks unattached.
    """

    FLAT = ""
    NESTED = "datasets/tuyennguyen21pt"

    def tree(self, root, prefix, model_rel):
        code = root / prefix / "evidencekd-kaggle-code" if prefix else root / "evidencekd-kaggle-code"
        (code / "src" / "evidencekd").mkdir(parents=True)
        (code / "src" / "evidencekd" / "__init__.py").write_text("", encoding="utf-8")
        (code / "data" / "processed").mkdir(parents=True)
        (code / "data" / "processed" / "train.jsonl").write_text("{}", encoding="utf-8")
        images = code.parent / "evidencekd-kaggle-images" / "dataset" / "train_data" / "train"
        images.mkdir(parents=True)
        (images / "a.png").write_bytes(b"")
        return str(code), model_dir(root, model_rel, MEDGEMMA_CONFIG)

    def finders(self, root):
        ns: dict = {}
        exec(nc.FINDERS, ns)
        return functools.partial(ns["find_inputs"], root=str(root))

    @pytest.mark.parametrize("prefix", [FLAT, NESTED])
    def test_the_code_bundle_and_images_resolve_either_way(self, tmp_path, prefix):
        code, _ = self.tree(tmp_path, prefix, "medgemma/transformers/medgemma-4b-it/1")
        find = self.finders(tmp_path)
        assert find("src/evidencekd") == [code]
        assert find("data/processed/train.jsonl") == [code]
        assert find("dataset/train_data/train")

    @pytest.mark.parametrize("model_rel,depth", [
        ("models/tuyennguyen21pt/mg/transformers/mg-4b-it/1", 6),
        ("datasets/tuyennguyen21pt/models/mg/transformers/mg-4b-it/1", 7),
    ])
    def test_a_model_under_owner_framework_variation_version_is_reached(self, tmp_path, model_rel, depth):
        # depth 7 was past the old budget: the teacher silently "was not attached"
        assert model_rel.count("/") + 1 == depth
        _, want = self.tree(tmp_path, self.NESTED, model_rel)
        assert detect(tmp_path)["MEDGEMMA"] == want

    def test_the_image_folders_are_never_descended(self, tmp_path):
        # thousands of PNGs; walking them would cost minutes per marker
        deep = tmp_path / "evidencekd-kaggle-images" / "dataset" / "train_data" / "train" / "nested"
        deep.mkdir(parents=True)
        (deep / "config.json").write_text(json.dumps(MEDGEMMA_CONFIG), encoding="utf-8")
        assert self.finders(tmp_path)("config.json") == []
