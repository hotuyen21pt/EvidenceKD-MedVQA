"""Tests for the config EvidenceKD hands ``vindr_cxr_vqa``'s model loader.

``MedGemmaTeacher`` builds a small vindr-shaped config; ``load_base_model``
falls back to ``cfg.train.load_in_4bit`` whenever its ``load_in_4bit``
argument is ``None``, so that block must exist and the decision must be made
here — a missing ``train`` key killed every KD-R and KD-C shard on Kaggle with
``AttributeError: config has no key 'train'``.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from evidencekd.teacher.medgemma import build_vindr_config, check_image_text_model, resolve_4bit
from evidencekd.utils.config import Config


def teacher_cfg(load_in_4bit=None) -> Config:
    return Config({
        "model": {"name": "google/medgemma-1.5-4b-it", "attn_implementation": "eager",
                  "torch_dtype": "auto", "adapter": "/a", "load_in_4bit": load_in_4bit},
        "generation": {"max_new_tokens": 256, "rationale_max_new_tokens": 200,
                       "rationale_n": 3, "rationale_temperature": 0.7},
    })


class TestResolve4bit:
    def test_explicit_argument_wins_over_everything(self):
        assert resolve_4bit(teacher_cfg(load_in_4bit=False), True, vram_gb=80.0) is True
        assert resolve_4bit(teacher_cfg(load_in_4bit=True), False, vram_gb=8.0) is False

    def test_config_value_wins_over_vram(self):
        assert resolve_4bit(teacher_cfg(load_in_4bit=True), None, vram_gb=80.0) is True
        assert resolve_4bit(teacher_cfg(load_in_4bit=False), None, vram_gb=8.0) is False

    @pytest.mark.parametrize("vram,expected", [(15.0, True), (16.0, True), (23.9, True),
                                               (24.0, False), (40.0, False), (80.0, False)])
    def test_null_means_decide_by_vram(self, vram, expected):
        # configs/teacher.yaml: "null = 4-bit on GPUs under ~24 GB"
        assert resolve_4bit(teacher_cfg(), None, vram_gb=vram) is expected

    def test_a_cpu_run_reports_zero_vram_and_gets_4bit(self):
        # not useful, but it must not crash before saying so
        assert resolve_4bit(teacher_cfg(), None, vram_gb=0.0) is True

    def test_a_config_without_the_key_at_all_still_resolves(self):
        cfg = Config({"model": {"name": "m", "attn_implementation": "eager", "torch_dtype": "auto"},
                      "generation": {"max_new_tokens": 1}})
        assert resolve_4bit(cfg, None, vram_gb=8.0) is True


class TestVindrConfig:
    def test_it_carries_a_train_block_with_a_concrete_bool(self):
        # the regression: load_base_model reads cfg.train.load_in_4bit
        vcfg = build_vindr_config(teacher_cfg(), load_in_4bit=True)
        assert vcfg.train.load_in_4bit is True

    def test_the_train_block_is_never_none(self):
        vcfg = build_vindr_config(teacher_cfg(), load_in_4bit=False)
        assert vcfg.train.load_in_4bit is False, "None would send the loader back to the missing key"

    def test_model_block_is_passed_through(self):
        vcfg = build_vindr_config(teacher_cfg(), load_in_4bit=False)
        assert vcfg.model.name == "google/medgemma-1.5-4b-it"
        assert vcfg.model.attn_implementation == "eager"
        assert vcfg.model.torch_dtype == "auto"

    def test_generation_is_greedy_and_bounded(self):
        vcfg = build_vindr_config(teacher_cfg(), load_in_4bit=False)
        assert vcfg.eval.max_new_tokens == 256
        assert vcfg.generate.do_sample is False and vcfg.generate.num_beams == 1

    def test_every_key_the_vindr_loader_touches_is_present(self):
        vcfg = build_vindr_config(teacher_cfg(), load_in_4bit=False)
        for path in (("model", "name"), ("model", "attn_implementation"), ("model", "torch_dtype"),
                     ("train", "load_in_4bit"), ("eval", "max_new_tokens")):
            node = vcfg
            for k in path:
                node = getattr(node, k)


class TestBaseCheckpointIsAVisionModel:
    """The teacher loads through ``AutoModelForImageTextToText``.

    A Kaggle session auto-detected the flan-T5 attached for generator S as the
    teacher; all 60 KD shards died inside transformers with ``Unrecognized
    configuration class T5Config``, which names neither the path nor the
    expectation. The check belongs here, before a 4 GB load.
    """

    @staticmethod
    def ckpt(tmp_path, name, config):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return str(d)

    def test_a_t5_checkpoint_is_refused_by_path_and_model_type(self, tmp_path):
        bad = self.ckpt(tmp_path, "flan-t5-base", {"model_type": "t5", "d_model": 768})
        with pytest.raises(ValueError) as e:
            check_image_text_model(bad)
        assert bad in str(e.value) and "t5" in str(e.value)

    def test_a_text_only_gemma_is_refused_too(self, tmp_path):
        bad = self.ckpt(tmp_path, "gemma-3-1b-it", {"model_type": "gemma3_text"})
        with pytest.raises(ValueError):
            check_image_text_model(bad)

    def test_a_vision_language_checkpoint_passes(self, tmp_path):
        ok = self.ckpt(tmp_path, "medgemma-4b-it",
                       {"model_type": "gemma3", "vision_config": {"image_size": 896}})
        check_image_text_model(ok)

    def test_an_older_style_vision_tower_key_passes(self, tmp_path):
        ok = self.ckpt(tmp_path, "vlm", {"model_type": "llava", "vision_tower": "openai/clip"})
        check_image_text_model(ok)

    def test_a_hub_id_is_left_to_the_loader(self):
        # offline check only; nothing is downloaded to find out
        check_image_text_model("google/medgemma-1.5-4b-it")

    def test_a_directory_without_a_config_is_left_to_the_loader(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        check_image_text_model(str(d))


class TestEveryKeyTheVindrPathReads:
    """The vindr config shape, checked against vindr's source, not a memory of it.

    This has now bitten twice on Kaggle, both times after the model had
    loaded, both times costing a whole teacher stage:

        AttributeError: config has no key 'train'; keys: [eval, generate, model]
        AttributeError: config has no key 'data';  keys: [eval, generate, model, train]

    So rather than add the missing key and wait for the third, read the
    functions ``MedGemmaTeacher`` actually calls and require the config to
    cover every ``cfg.<block>.<key>`` they touch.
    """

    #: (file, function) of each vindr entry point the teacher goes through.
    ENTRY_POINTS = (("infer.py", "load_model"), ("infer.py", "load_processor"),
                    ("infer.py", "answer"), ("modeling.py", "load_base_model"))

    @staticmethod
    def vindr_dir():
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(os.path.join(here, "..", "VinDR-CXR-VQA", "vindr_cxr_vqa"))

    @classmethod
    def keys_read_by(cls, filename, function):
        """``{(block, key)}`` that one vindr function reads off its config.

        Parsed from source: importing it would pull in torch, which this
        suite deliberately runs without.
        """
        path = os.path.join(cls.vindr_dir(), filename)
        src = open(path, encoding="utf-8").read()
        body = re.search(rf"^def {function}\(.*?(?=^(?:@|def )|\Z)", src, re.M | re.S)
        assert body, f"{function} not found in {path}"
        return set(re.findall(r"\bcfg\.(\w+)\.(\w+)", body.group(0)))

    @pytest.mark.parametrize("filename,function", ENTRY_POINTS)
    def test_the_config_carries_every_key(self, filename, function):
        if not os.path.isdir(self.vindr_dir()):
            pytest.skip("sibling VinDR-CXR-VQA checkout not present")
        vcfg = build_vindr_config(teacher_cfg(), load_in_4bit=False)
        for block, key in sorted(self.keys_read_by(filename, function)):
            node = getattr(vcfg, block, None)
            assert node is not None, f"{function} reads cfg.{block}.{key}; no {block!r} block"
            assert key in node, f"{function} reads cfg.{block}.{key}; block {block!r} lacks it"

    def test_the_scan_actually_finds_something(self):
        # a regex that silently matched nothing would make the test above vacuous
        if not os.path.isdir(self.vindr_dir()):
            pytest.skip("sibling VinDR-CXR-VQA checkout not present")
        assert ("data", "loc_format") in self.keys_read_by("infer.py", "answer")
        assert ("train", "load_in_4bit") in self.keys_read_by("modeling.py", "load_base_model")

    def test_the_loc_format_is_the_one_the_adapter_was_trained_with(self):
        # vindr configs/default.yaml: loc_format: norm1000. A mismatch here
        # changes the prompt the LoRA adapter sees, silently.
        vcfg = build_vindr_config(teacher_cfg(), load_in_4bit=False)
        assert vcfg.data.loc_format == "norm1000"
