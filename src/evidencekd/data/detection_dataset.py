"""Detector training/inference samples from the VinDr-CXR copy (design §7.1).

Reads the radiologist CSVs through ``vindr_cxr_vqa.vindr`` (the sibling
repository's tested readers), fuses the three independent readings of each
lesion with the same IoU the VQA boxes were repaired with, maps boxes into the
224 px PNG space, and keeps only the 14 VQA classes.

The index-building half (:func:`build_index`) is pure Python and tested; the
``torch.utils.data.Dataset`` half imports torch lazily.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Sequence

from evidencekd.data.vindr_processed import ProcessedSplits
from evidencekd.labels import VQA_CLASSES, VQA_CLASS_TO_ID


@dataclass(frozen=True)
class DetSample:
    image_id: str
    path: str                                  # absolute PNG path
    width: int                                 # original DICOM
    height: int
    resized_width: int
    resized_height: int
    boxes_224: tuple[tuple[float, float, float, float], ...] = ()
    labels: tuple[int, ...] = ()               # 1-based (0 = background)
    class_names: tuple[str, ...] = field(default=())

    @property
    def fx(self) -> float:
        return self.resized_width / max(self.width, 1)

    @property
    def fy(self) -> float:
        return self.resized_height / max(self.height, 1)

    @property
    def has_boxes(self) -> bool:
        return bool(self.boxes_224)


def _load_vindr(dataset_dir: str, partition: str = "train"):
    try:
        from vindr_cxr_vqa import vindr  # the sibling repo's readers
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "vindr_cxr_vqa is not importable. Add ../VinDR-CXR-VQA to PYTHONPATH "
            "or build the Kaggle bundle, which ships it.") from e
    return vindr, vindr.load_images(dataset_dir, partition)


def build_index(
    dataset_dir: str,
    *,
    splits: ProcessedSplits | None,
    image_source: str = "all",
    fuse_iou: float = 0.3,
    min_votes: int = 1,
    normal_ratio: float = 1.0,
    seed: int = 42,
    stage: str = "detector_pretrain",
    partition: str = "train",
    image_ids: Sequence[str] | None = None,
) -> list[DetSample]:
    """Training (or inference) samples in 224 px space.

    * ``image_source="vqa_train"`` restricts to ``splits.train``; ``"all"``
      takes every PNG of the partition on disk.
    * ``splits`` val/test images are asserted absent (``LeakageError``).
    * ``normal_ratio`` subsamples box-free images relative to boxed ones.
    * ``image_ids`` overrides both selections (used for inference on a split).
    """
    vindr, metas = _load_vindr(dataset_dir, partition)
    if image_ids is not None:
        chosen = [metas[i] for i in image_ids if i in metas]
    elif image_source == "vqa_train":
        if splits is None:
            raise ValueError("image_source='vqa_train' needs splits")
        chosen = [metas[i] for i in splits.train if i in metas]
    elif image_source == "all":
        chosen = list(metas.values())
    else:
        raise ValueError(f"unknown image_source {image_source!r}")

    if splits is not None and image_ids is None:
        # Drop the VQA val/test images from whatever source was chosen, then
        # assert it — the assertion documents the guarantee, the filter makes it.
        held = splits.held_out  # all of val (test is a subset of it)
        chosen = [m for m in chosen if m.image_id not in held]
        splits.assert_held_out((m.image_id for m in chosen), stage=stage)

    boxed, normal = [], []
    for m in chosen:
        fused = vindr.fuse_boxes([b for b in m.boxes if b.class_name in VQA_CLASS_TO_ID],
                                 thr=fuse_iou)
        fused = [f for f in fused if f["votes"] >= min_votes]
        boxes, labels, names = [], [], []
        for f in fused:
            x0, y0, x1, y1 = f["bbox"]
            bx = (x0 * m.fx, y0 * m.fy, x1 * m.fx, y1 * m.fy)
            if bx[2] - bx[0] < 1.0 or bx[3] - bx[1] < 1.0:
                continue  # degenerate at 224 px
            boxes.append(bx)
            labels.append(VQA_CLASS_TO_ID[f["class_name"]] + 1)
            names.append(f["class_name"])
        s = DetSample(m.image_id, os.path.abspath(m.path), m.width, m.height,
                      m.resized_width, m.resized_height,
                      tuple(boxes), tuple(labels), tuple(names))
        (boxed if s.has_boxes else normal).append(s)

    if image_ids is None:
        # Training selection: subsample background images relative to boxed
        # ones. Sorted before shuffling so the result does not depend on
        # os.listdir order.
        normal.sort(key=lambda s: s.image_id)
        if normal_ratio <= 0:
            normal = []
        else:
            random.Random(seed).shuffle(normal)
            normal = normal[: int(round(len(boxed) * normal_ratio))]
    out = boxed + normal
    out.sort(key=lambda s: s.image_id)
    return out


def summarize(samples: Sequence[DetSample]) -> dict:
    per_class = {c: 0 for c in VQA_CLASSES}
    for s in samples:
        for n in s.class_names:
            per_class[n] += 1
    return {
        "n_images": len(samples),
        "n_with_boxes": sum(s.has_boxes for s in samples),
        "n_boxes": sum(len(s.boxes_224) for s in samples),
        "per_class_boxes": per_class,
    }


# ---------------------------------------------------------------------------
# torch half
# ---------------------------------------------------------------------------
class VinDrDetectionDataset:  # torch.utils.data.Dataset duck-type
    """PNG -> float tensor [3, H, W] in [0, 1] + torchvision target dict."""

    def __init__(self, samples: Sequence[DetSample], *, train: bool = True,
                 hflip_prob: float = 0.0):
        import torch  # noqa: F401  (fail early, clearly)
        self.samples = list(samples)
        self.train = train
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        w, h = img.size
        t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        t = t.view(h, w, 3).permute(2, 0, 1).float().div_(255.0)
        boxes = torch.tensor(s.boxes_224, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(s.labels, dtype=torch.int64)
        # No horizontal flip by default: image-left is the patient's right and
        # laterality facts depend on it (README "Decisions that must not be reverted").
        if self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            t = t.flip(-1)
            if len(boxes):
                boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
        target = {"boxes": boxes, "labels": labels,
                  "image_id": torch.tensor([i])}
        return t, target


def collate(batch):
    return tuple(zip(*batch))
