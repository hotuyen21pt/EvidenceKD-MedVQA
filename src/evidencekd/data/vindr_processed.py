"""Readers for the dataset built by ``vindr_cxr_vqa.build_dataset``.

The sibling repository ``VinDR-CXR-VQA`` joins the HuggingFace QA release with
the local VinDr-CXR images, repairs the 60 % of boxes that were paired with the
wrong pathology, and writes one JSON object per QA pair into
``data/processed/{train,val,test}.jsonl`` plus a per-image index
``images.json``. This module reads those files and nothing else; it does not
re-derive any of it.

Coordinate convention (design §5.1): every box exposed here is in **original
DICOM pixel space**, the same space as ``data.vqa_schema.LocBox`` and
``data.boxes.Box``. The 224x224 PNG space is reached through ``fx``/``fy``.

Pure Python: no torch, no numpy, no pandas.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, Sequence

from evidencekd.data.boxes import Box, FusedBox
from evidencekd.data.splits import Splits, assert_no_leakage
from evidencekd.labels import VQA_CLASSES, finding_in_text

SPLIT_FILES = ("train", "val", "test")


class ProcessedDataError(RuntimeError):
    """The processed directory is missing or does not look like VinDr output."""


@dataclass(frozen=True)
class VQASample:
    """One QA pair of the release, with the repaired supervision box."""

    sample_id: str
    image_id: str
    image_path: str          # relative to the VinDr dataset dir, forward slashes
    split: str
    question: str
    answer: str
    reason: str
    type: str
    difficulty: str
    gt_finding: str
    bbox_orig: tuple[float, float, float, float] | None        # as published
    bbox_fixed_orig: tuple[float, float, float, float] | None  # re-joined to gt_finding
    repair_method: str
    box_label_consistent: bool
    image_width: int
    image_height: int
    resized_width: int
    resized_height: int

    @property
    def fx(self) -> float:
        return self.resized_width / max(self.image_width, 1)

    @property
    def fy(self) -> float:
        return self.resized_height / max(self.image_height, 1)

    @property
    def gt_box(self) -> Box | None:
        """The supervision box (design §5.1: ``gt_location_fixed``)."""
        if self.bbox_fixed_orig is None:
            return None
        x0, y0, x1, y1 = self.bbox_fixed_orig
        return Box(x0, y0, x1, y1, class_name=self.gt_finding)

    @property
    def question_finding(self) -> str | None:
        """The class the question names, if it names one."""
        return finding_in_text(self.question)


@dataclass(frozen=True)
class ImageRecord:
    """One image of the release, with every radiologist box."""

    image_id: str
    image_path: str
    split: str
    width: int
    height: int
    resized_width: int
    resized_height: int
    labels_majority: tuple[str, ...]
    labels_union: tuple[str, ...]
    boxes_raw: tuple[Box, ...]
    boxes_fused: tuple[FusedBox, ...]

    @property
    def fx(self) -> float:
        return self.resized_width / max(self.width, 1)

    @property
    def fy(self) -> float:
        return self.resized_height / max(self.height, 1)

    def fused_of(self, class_name: str) -> tuple[FusedBox, ...]:
        return tuple(f for f in self.boxes_fused if f.box.class_name == class_name)

    @property
    def vqa_classes_present(self) -> tuple[str, ...]:
        """Classes with at least one raw box, restricted to the 14 VQA classes."""
        seen = {b.class_name for b in self.boxes_raw}
        return tuple(c for c in VQA_CLASSES if c in seen)


# ---------------------------------------------------------------------------
def _tuple4(v) -> tuple[float, float, float, float] | None:
    if not v:
        return None
    if len(v) != 4:
        raise ProcessedDataError(f"box with {len(v)} coordinates: {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def _sample_from_dict(d: dict) -> VQASample:
    try:
        return VQASample(
            sample_id=d["sample_id"],
            image_id=d["image_id"],
            image_path=d["image_path"],
            split=d["split"],
            question=d["question"],
            answer=d["answer"],
            reason=d.get("reason", ""),
            type=d["type"],
            difficulty=d.get("difficulty", ""),
            gt_finding=d["gt_finding"],
            bbox_orig=_tuple4(d.get("bbox_orig")),
            bbox_fixed_orig=_tuple4(d.get("bbox_fixed_orig")),
            repair_method=d.get("repair_method", "published"),
            box_label_consistent=bool(d.get("box_label_consistent", True)),
            image_width=int(d["image_width"]),
            image_height=int(d["image_height"]),
            resized_width=int(d["resized_width"]),
            resized_height=int(d["resized_height"]),
        )
    except KeyError as e:
        raise ProcessedDataError(f"record {d.get('sample_id')!r} lacks {e}") from None


def iter_samples(processed_dir: str, split: str) -> Iterator[VQASample]:
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {SPLIT_FILES}, got {split!r}")
    path = os.path.join(processed_dir, f"{split}.jsonl")
    if not os.path.isfile(path):
        raise ProcessedDataError(
            f"{path} not found — build it with "
            "`python scripts/build_data.py` (design §5.2)")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield _sample_from_dict(json.loads(line))


def load_samples(processed_dir: str, split: str) -> list[VQASample]:
    return list(iter_samples(processed_dir, split))


def _image_from_dict(d: dict) -> ImageRecord:
    raw = tuple(
        Box(*map(float, b["bbox_orig"]), class_name=b["class_name"],
            rad_id=b.get("rad_id"))
        for b in d.get("boxes_raw", []))
    fused = tuple(
        FusedBox(box=Box(*map(float, b["bbox_orig"]), class_name=b["class_name"]),
                 n_raters=int(b.get("votes", 1)))
        for b in d.get("boxes_fused", []))
    return ImageRecord(
        image_id=d["image_id"],
        image_path=d["image_path"],
        split=d["split"],
        width=int(d["width"]), height=int(d["height"]),
        resized_width=int(d["resized_width"]),
        resized_height=int(d["resized_height"]),
        labels_majority=tuple(d.get("labels_majority", [])),
        labels_union=tuple(d.get("labels_union", [])),
        boxes_raw=raw, boxes_fused=fused,
    )


def load_images(processed_dir: str) -> dict[str, ImageRecord]:
    """``images.json`` -> image_id -> :class:`ImageRecord`."""
    path = os.path.join(processed_dir, "images.json")
    if not os.path.isfile(path):
        raise ProcessedDataError(f"{path} not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data.values() if isinstance(data, dict) else data
    out = {}
    for d in records:
        rec = _image_from_dict(d)
        out[rec.image_id] = rec
    return out


def splits_from_images(images: dict[str, ImageRecord]) -> Splits:
    """The image-level split recorded in ``images.json``.

    Raises ``LeakageError`` if an image is listed in two splits — which cannot
    happen for a file written by ``build_dataset --disjoint-test``, and is
    exactly the thing to catch when someone builds without the flag.
    """
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for rec in images.values():
        if rec.split not in buckets:
            raise ProcessedDataError(f"unknown split {rec.split!r} on {rec.image_id}")
        buckets[rec.split].append(rec.image_id)
    s = Splits(train=tuple(sorted(buckets["train"])),
               val=tuple(sorted(buckets["val"])),
               test=tuple(sorted(buckets["test"])))
    assert_no_leakage(s)
    return s


def samples_by_image(samples: Sequence[VQASample]) -> dict[str, list[VQASample]]:
    out: dict[str, list[VQASample]] = {}
    for s in samples:
        out.setdefault(s.image_id, []).append(s)
    return out


def build_summary(processed_dir: str) -> dict:
    path = os.path.join(processed_dir, "build_summary.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
