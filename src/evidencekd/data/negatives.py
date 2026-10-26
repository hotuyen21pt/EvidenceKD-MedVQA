"""Synthetic negative questions for QA+Full training — design §5.4.

The release has no negative sample (DATA_NOTES §3). Without any, the Yes/No
head and the ``absence`` fact cannot be learned. This module writes templated
``Is_there`` / ``Yes_No`` questions whose answer is "No":

* on VQA **training** images, for classes that have no fused box there;
* on VinDr-CXR "No finding" images (outside the VQA set), for any class.

Every synthetic record carries ``source: "synthetic_negative"`` and is
counted, never silently mixed in. Val/test are untouched: Tables 4-6 stay on
the released data; Yes/No F1 is reported on a separate "A-neg" set built from
VinDr-CXR's test partition (consensus of five readers).

Pure Python.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from evidencekd.data.vindr_processed import ImageRecord, VQASample
from evidencekd.labels import VQA_CLASSES

TEMPLATES = {
    "Is_there": ("Is there {finding} in this chest X-ray?",
                 "Is there any evidence of {finding}?",
                 "Does this X-ray show {finding}?"),
    "Yes_No": ("Can {finding} be seen on this image?",
               "Is {finding} present in this chest X-ray?",
               "Are there signs of {finding}?"),
}
NEGATIVE_ANSWER = "No, there is no evidence of {finding} in this image."


@dataclass(frozen=True)
class NegativeSpec:
    per_positive_ratio: float = 1.0     # negatives per released QA pair, overall
    on_vqa_images: bool = True
    on_normal_images: bool = True
    max_normal_images: int = 4000
    seed: int = 42


def negative_samples(positives: Sequence[VQASample], images: Mapping[str, ImageRecord],
                     normal_images: Sequence[ImageRecord], spec: NegativeSpec) -> list[VQASample]:
    """Balanced per class: as many "No" questions about class c as "Yes" ones."""
    rng = random.Random(spec.seed)
    want = {c: 0 for c in VQA_CLASSES}
    for s in positives:
        if s.gt_finding in want:
            want[s.gt_finding] += 1
    want = {c: int(round(n * spec.per_positive_ratio)) for c, n in want.items()}

    pools: dict[str, list[ImageRecord]] = {c: [] for c in VQA_CLASSES}
    if spec.on_vqa_images:
        for s in positives:
            rec = images.get(s.image_id)
            if rec is None:
                continue
            present = {f.box.class_name for f in rec.boxes_fused}
            for c in VQA_CLASSES:
                if c not in present:
                    pools[c].append(rec)
    if spec.on_normal_images:
        normals = list(normal_images)[: spec.max_normal_images]
        for c in VQA_CLASSES:
            pools[c].extend(normals)

    out: list[VQASample] = []
    for c in VQA_CLASSES:
        pool = list({r.image_id: r for r in pools[c]}.values())
        rng.shuffle(pool)
        for k, rec in enumerate(pool[: want[c]]):
            qtype = "Is_there" if k % 2 == 0 else "Yes_No"
            template = TEMPLATES[qtype][k % len(TEMPLATES[qtype])]
            out.append(VQASample(
                sample_id=f"{rec.image_id}_neg_{c.replace(' ', '').replace('/', '')}",
                image_id=rec.image_id, image_path=rec.image_path, split="train",
                question=template.format(finding=c), answer=NEGATIVE_ANSWER.format(finding=c),
                reason="", type=qtype, difficulty="Easy", gt_finding=c,
                bbox_orig=None, bbox_fixed_orig=None, repair_method="synthetic_negative",
                box_label_consistent=True, image_width=rec.width, image_height=rec.height,
                resized_width=rec.resized_width, resized_height=rec.resized_height))
    return out


def normal_image_records(dataset_dir: str, *, exclude: set[str]) -> list[ImageRecord]:
    """VinDr-CXR train images whose majority label is 'No finding' and that are
    not in the VQA set, as ImageRecords with no boxes."""
    from vindr_cxr_vqa import vindr  # sibling readers
    metas = vindr.load_images(dataset_dir, "train")
    out = []
    for m in metas.values():
        if m.image_id in exclude or m.boxes:
            continue
        if m.labels_majority:          # any majority finding -> not normal
            continue
        out.append(ImageRecord(m.image_id, os.path.relpath(m.path, dataset_dir).replace("\\", "/"),
                               "train", False, m.width, m.height, m.resized_width, m.resized_height,
                               (), (), (), ()))
    out.sort(key=lambda r: r.image_id)
    return out


def write_negatives(samples: Sequence[VQASample], path: str) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    per_class: dict[str, int] = {}
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            per_class[s.gt_finding] = per_class.get(s.gt_finding, 0) + 1
            d = s.__dict__.copy()
            d["source"] = "synthetic_negative"
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return {"n_negatives": len(samples), "per_class": per_class, "path": path}
