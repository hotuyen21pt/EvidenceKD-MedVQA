"""The MedGemma teacher — the sibling repo's fine-tuned adapter, frozen.

Loads through ``vindr_cxr_vqa.infer`` (same base loader, same processor, same
chat template the adapter was trained with) and exposes the two things KD
needs and the spec's audit demands be verified (README "No teacher"):

* :meth:`answer_logprob` — teacher-forced ``log p_T(a* | I, Q)`` of a given
  answer string, mean over its tokens, batched over images (original + masked);
* :meth:`generate` — free answers (B4 rows, teacher correctness) and
  :meth:`generate_text` — rationales from an arbitrary prompt (KD-C).

Masking (``mask_region``) is done on the 224 px PNG before the processor,
so the hidden rectangle survives the processor's resize.
Torch / transformers imported lazily.
"""
from __future__ import annotations

import os
from typing import Sequence

from evidencekd.data.boxes import Box
from evidencekd.utils.config import Config


def mask_region(image, box: Box, fx: float, fy: float, *, method: str = "constant_fill",
                fill: int = 0, blur_sigma: float = 8.0):
    """A copy of the PIL image with the region hidden (box in original px)."""
    from PIL import ImageDraw, ImageFilter

    im = image.copy()
    x0, y0 = max(0, int(box.x_min * fx)), max(0, int(box.y_min * fy))
    x1, y1 = min(im.width, int(round(box.x_max * fx))), min(im.height, int(round(box.y_max * fy)))
    if x1 <= x0 or y1 <= y0:
        return im
    if method == "blur":
        patch = im.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(blur_sigma))
        im.paste(patch, (x0, y0))
    else:
        ImageDraw.Draw(im).rectangle([x0, y0, x1 - 1, y1 - 1], fill=(fill, fill, fill))
    return im


class MedGemmaTeacher:
    def __init__(self, cfg, *, adapter: str | None = None, load_in_4bit: bool | None = None):
        import torch
        from vindr_cxr_vqa.infer import load_model, load_processor

        self.cfg = cfg
        vcfg = Config({"model": {"name": cfg.model.name,
                                 "attn_implementation": cfg.model.get("attn_implementation", "eager"),
                                 "torch_dtype": cfg.model.get("torch_dtype", "auto")},
                       "eval": {"max_new_tokens": int(cfg.generation.max_new_tokens)},
                       "generate": {"do_sample": False, "num_beams": 1}})
        self.vcfg = vcfg
        adapter = adapter if adapter is not None else cfg.model.get("adapter")
        if adapter and not os.path.isdir(adapter):
            raise FileNotFoundError(f"adapter directory not found: {adapter}")
        self.adapter = adapter
        four = load_in_4bit if load_in_4bit is not None else cfg.model.get("load_in_4bit")
        self.model = load_model(vcfg, adapter, four)
        self.processor = load_processor(vcfg, adapter)
        self.tokenizer = self.processor.tokenizer
        self.device = next(self.model.parameters()).device
        self.torch = torch

    # ------------------------------------------------------------------
    def _prompt_text(self, question: str, *, system: str | None = None) -> str:
        from vindr_cxr_vqa.prompts import build_messages
        msgs = build_messages({"question": question}, None, "norm1000")
        if system is not None:
            msgs[0] = {"role": "system", "content": [{"type": "text", "text": system}]}
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def _prompt_from_user_text(self, user_text: str, system: str) -> str:
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @staticmethod
    def _image_token_ids(processor) -> list[int]:
        from vindr_cxr_vqa.data import _image_token_ids
        return _image_token_ids(processor)

    # ------------------------------------------------------------------
    def answer_logprob(self, images: Sequence, question: str, answer: str) -> list[float]:
        """Mean token log-probability of ``answer`` after the prompt, one per image.

        All images share the question, so the batch is (original + masked
        copies) of one sample — the KD-R inner loop.
        """
        torch = self.torch
        prompt = self._prompt_text(question)
        full = prompt + answer + (self.tokenizer.eos_token or "")
        proc = self.processor
        enc = proc(text=[full] * len(images), images=[[im] for im in images], return_tensors="pt", padding=True)
        p_only = proc(text=[prompt], images=[[images[0]]], return_tensors="pt")
        n_prompt = int(p_only["input_ids"].shape[1])
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits.float()
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        # left padding: the answer occupies the last (T - n_prompt_i) real tokens
        T = ids.shape[1]
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)  # (B, T-1)
        out = []
        for b in range(ids.shape[0]):
            real = int(attn[b].sum())
            start = T - real + n_prompt          # first answer token index
            tok_lp = lp[b, start - 1:T - 1]
            if tok_lp.numel() == 0:
                out.append(float("-inf"))
            else:
                out.append(float(tok_lp.mean()))
        return out

    def generate(self, images: Sequence, questions: Sequence[str], *, max_new_tokens: int | None = None) -> list[str]:
        from vindr_cxr_vqa.infer import answer
        records = [{"question": q} for q in questions]
        return answer(self.model, self.processor, self.vcfg, records, list(images),
                      max_new_tokens=max_new_tokens or int(self.cfg.generation.max_new_tokens))

    def generate_text(self, image, user_text: str, *, system: str, n: int = 1,
                      max_new_tokens: int = 200, temperature: float = 0.7) -> list[str]:
        torch = self.torch
        prompt = self._prompt_from_user_text(user_text, system)
        enc = self.processor(text=[prompt] * n, images=[[image]] * n, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                      temperature=max(temperature, 1e-5), top_p=0.95,
                                      pad_token_id=self.tokenizer.pad_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tokenizer.batch_decode(gen, skip_special_tokens=True)]
