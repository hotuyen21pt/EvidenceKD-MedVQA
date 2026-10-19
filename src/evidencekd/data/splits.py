"""Train/val/test splitting, by image (method spec §2).

Spec §2: *"chia theo ảnh; ảnh validation/test không xuất hiện ở tiền huấn
luyện detector, tinh chỉnh teacher hay sinh mục tiêu KD."*

Two requirements, both enforced here:

1. **Split by image, not by question.** Each image carries 3–5 questions. A
   question-level split would put questions about the *same radiograph* in
   train and test, which leaks the image.
2. **One split serves every stage.** The same held-out images must be absent
   from detector pre-training, teacher fine-tuning and KD target generation.
   :func:`assert_no_leakage` is meant to be called at the top of each of those
   scripts, not only once at split time.

The HF release exposes a single ``train`` split, so these splits are ours and
must be recorded: :func:`split_manifest` produces the hash that goes into
``reports/experiment_registry.csv``.

Pure Python.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


class LeakageError(RuntimeError):
    """Raised when an image appears in more than one split."""


@dataclass(frozen=True)
class Splits:
    """Image ids per split."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def of(self, name: str) -> tuple[str, ...]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, name)

    def as_dict(self) -> dict[str, list[str]]:
        return {name: list(self.of(name)) for name in SPLIT_NAMES}

    @property
    def sizes(self) -> dict[str, int]:
        return {name: len(self.of(name)) for name in SPLIT_NAMES}


def split_by_image(
    image_ids: Iterable[str],
    *,
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Splits:
    """Partition image ids into train/val/test.

    Deterministic given ``seed``: ids are sorted before shuffling, so the
    result does not depend on the order they arrived in (which varies with
    dataset streaming and worker count).

    >>> s = split_by_image([f"img{i}" for i in range(10)], seed=0)
    >>> s.sizes
    {'train': 8, 'val': 1, 'test': 1}
    >>> set(s.train) & set(s.test)
    set()
    """
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("fractions must be non-negative")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val+test fractions must be < 1.0, got {val_fraction + test_fraction}"
        )

    unique = sorted(set(image_ids))
    if not unique:
        raise ValueError("no image ids to split")

    rng = random.Random(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))
    if n_val + n_test >= n:
        raise ValueError(f"split leaves no training images: n={n}")

    test = tuple(sorted(shuffled[:n_test]))
    val = tuple(sorted(shuffled[n_test : n_test + n_val]))
    train = tuple(sorted(shuffled[n_test + n_val :]))
    splits = Splits(train=train, val=val, test=test)
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: Splits) -> None:
    """Raise if any image id appears in two splits.

    Call this at the start of every stage that touches data, not just once.

    >>> assert_no_leakage(Splits(("a",), ("b",), ("c",))) is None
    True
    >>> assert_no_leakage(Splits(("a",), ("a",), ("c",)))
    Traceback (most recent call last):
        ...
    evidencekd.data.splits.LeakageError: image(s) in multiple splits: ['a']
    """
    seen: dict[str, str] = {}
    duplicates: set[str] = set()
    for name in SPLIT_NAMES:
        for image_id in splits.of(name):
            if image_id in seen and seen[image_id] != name:
                duplicates.add(image_id)
            seen[image_id] = name
    if duplicates:
        raise LeakageError(f"image(s) in multiple splits: {sorted(duplicates)}")


def assert_images_held_out(
    used_image_ids: Iterable[str], splits: Splits, *, stage: str
) -> None:
    """Raise if a stage touched a val/test image.

    Guards the specific requirement that held-out images never appear in
    detector pre-training, teacher fine-tuning or KD target generation.
    """
    held_out = set(splits.val) | set(splits.test)
    violations = sorted(set(used_image_ids) & held_out)
    if violations:
        raise LeakageError(
            f"stage {stage!r} used {len(violations)} held-out image(s), "
            f"first few: {violations[:5]}"
        )


def split_manifest(splits: Splits) -> dict[str, object]:
    """A hashable record of the split, for the experiment registry."""
    payload = splits.as_dict()
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sizes": splits.sizes,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def questions_per_split(
    items_by_image: Mapping[str, Sequence[object]], splits: Splits
) -> dict[str, int]:
    """Count QA items per split, which differs from image counts (3–5 each)."""
    return {
        name: sum(len(items_by_image.get(i, ())) for i in splits.of(name))
        for name in SPLIT_NAMES
    }
