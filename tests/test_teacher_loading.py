"""Tests for the config EvidenceKD hands ``vindr_cxr_vqa``'s model loader.

``MedGemmaTeacher`` builds a small vindr-shaped config; ``load_base_model``
falls back to ``cfg.train.load_in_4bit`` whenever its ``load_in_4bit``
argument is ``None``, so that block must exist and the decision must be made
here — a missing ``train`` key killed every KD-R and KD-C shard on Kaggle with
``AttributeError: config has no key 'train'``.
"""

from __future__ import annotations

import pytest

from evidencekd.teacher.medgemma import build_vindr_config, resolve_4bit
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
