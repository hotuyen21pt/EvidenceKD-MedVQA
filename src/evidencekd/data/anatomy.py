"""Anatomy (lungs, heart, thorax) for the fact table — design §5.5.

The data carries no anatomy masks. ``segment_images`` runs torchxrayvision's
PSPNet (frozen) once per image and caches boxes + mask areas in
``derived/anatomy.json``; everything downstream reads the cache through
:func:`load_anatomy` and never imports torch. When the segmenter is not
available, :func:`fallback_anatomy` gives a thorax = whole-image box and no
heart/lungs, so side facts still work (midline = image centre) and zone /
extent / CTR facts are simply absent rather than invented.

View position is unknown for PNGs, so CTR is never computed (README rule).
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Mapping

from evidencekd.data.boxes import Box
from evidencekd.facts.geometry import Anatomy

TXRV_TARGETS = {"Left Lung": "lung_left", "Right Lung": "lung_right", "Heart": "heart"}
FALLBACK = "fallback"


def fallback_anatomy(width: int, height: int) -> Anatomy:
    return Anatomy(thorax=Box(0.0, 0.0, float(width), float(height), class_name="A_thorax"))


def anatomy_to_dict(a: Anatomy, source: str) -> dict:
    def b(x: Box | None):
        return None if x is None else [x.x_min, x.y_min, x.x_max, x.y_max]
    return {"thorax": b(a.thorax), "heart": b(a.heart), "lung_left": b(a.lung_left),
            "lung_right": b(a.lung_right), "lung_left_area": a.lung_left_area,
            "lung_right_area": a.lung_right_area, "source": source}


def anatomy_from_dict(d: dict) -> Anatomy:
    def b(v, name):
        return None if not v else Box(float(v[0]), float(v[1]), float(v[2]), float(v[3]), class_name=name)
    return Anatomy(thorax=b(d.get("thorax"), "A_thorax"), heart=b(d.get("heart"), "A_heart"),
                   lung_left=b(d.get("lung_left"), "A_lung_left"),
                   lung_right=b(d.get("lung_right"), "A_lung_right"),
                   lung_left_area=d.get("lung_left_area"), lung_right_area=d.get("lung_right_area"))


def save_anatomy(entries: Mapping[str, tuple[Anatomy, str]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: anatomy_to_dict(a, src) for k, (a, src) in entries.items()}, f)


def load_anatomy(path: str | None) -> dict[str, Anatomy]:
    """``{image_id: Anatomy}``; an empty dict when there is no cache."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {k: anatomy_from_dict(v) for k, v in json.load(f).items()}


def anatomy_for(cache: Mapping[str, Anatomy], image_id: str, width: int, height: int) -> Anatomy:
    return cache.get(image_id) or fallback_anatomy(width, height)


# ---------------------------------------------------------------------------
# torch half — torchxrayvision
# ---------------------------------------------------------------------------
def _mask_to_box_and_area(mask, fx: float, fy: float) -> tuple[Box | None, float | None]:
    """A binary mask in model space -> box in original pixels + area in original px^2."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    x0, x1, y0, y1 = xs.min() / fx, (xs.max() + 1) / fx, ys.min() / fy, (ys.max() + 1) / fy
    area = float(mask.sum()) / (fx * fy)
    return Box(float(x0), float(y0), float(x1), float(y1), class_name=""), area


def segment_images(items: Iterable[tuple[str, str, int, int]], *, out_path: str,
                   weights: str | None = None, device: str | None = None,
                   threshold: float = 0.5, batch_size: int = 8) -> dict[str, tuple[Anatomy, str]]:
    """Run torchxrayvision PSPNet over ``(image_id, png_path, orig_w, orig_h)`` items.

    Falls back to :func:`fallback_anatomy` per image when the package or its
    weights are unavailable, recording ``source`` so the fact notes can say
    "widths from boxes" vs masks. Writes and returns the cache.
    """
    import numpy as np
    import torch
    from PIL import Image

    entries: dict[str, tuple[Anatomy, str]] = {}
    model = None
    try:
        import torchxrayvision as xrv  # type: ignore
        model = xrv.baseline_models.chestx_det.PSPNet(weights_url=weights) if weights else xrv.baseline_models.chestx_det.PSPNet()
        model.eval()
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(dev)
        targets = list(model.targets)
    except Exception as e:  # noqa: BLE001
        print(f"[anatomy] torchxrayvision unavailable ({e}); writing fallback anatomy")
        model = None

    items = list(items)
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        if model is None:
            for iid, _p, w, h in chunk:
                entries[iid] = (fallback_anatomy(w, h), FALLBACK)
            continue
        xs = []
        for _iid, p, _w, _h in chunk:
            im = Image.open(p).convert("L").resize((512, 512))
            arr = np.asarray(im, dtype=np.float32)
            arr = xrv.datasets.normalize(arr, 255)  # to [-1024, 1024]
            xs.append(torch.from_numpy(arr)[None])
        x = torch.stack(xs).to(dev)
        with torch.no_grad():
            pred = torch.sigmoid(model(x)).cpu().numpy()  # (B, T, 512, 512)
        for (iid, _p, w, h), pr in zip(chunk, pred):
            fx, fy = 512 / max(w, 1), 512 / max(h, 1)
            parts: dict[str, tuple[Box | None, float | None]] = {}
            for name, key in TXRV_TARGETS.items():
                if name in targets:
                    parts[key] = _mask_to_box_and_area(pr[targets.index(name)] > threshold, fx, fy)
                else:
                    parts[key] = (None, None)
            ll, lr, he = parts["lung_left"], parts["lung_right"], parts["heart"]
            # torchxrayvision names sides anatomically (patient's left); a
            # patient-left lung sits on the image's right. Keep as given.
            boxes = [b for b, _ in (ll, lr, he) if b is not None]
            if boxes:
                thorax = Box(min(b.x_min for b in boxes), min(b.y_min for b in boxes),
                             max(b.x_max for b in boxes), max(b.y_max for b in boxes), class_name="A_thorax")
            else:
                thorax = Box(0.0, 0.0, float(w), float(h), class_name="A_thorax")
            a = Anatomy(thorax=thorax,
                        heart=None if he[0] is None else Box(*(he[0].x_min, he[0].y_min, he[0].x_max, he[0].y_max), class_name="A_heart"),
                        lung_left=None if ll[0] is None else Box(*(ll[0].x_min, ll[0].y_min, ll[0].x_max, ll[0].y_max), class_name="A_lung_left"),
                        lung_right=None if lr[0] is None else Box(*(lr[0].x_min, lr[0].y_min, lr[0].x_max, lr[0].y_max), class_name="A_lung_right"),
                        lung_left_area=ll[1], lung_right_area=lr[1])
            entries[iid] = (a, "txrv" if boxes else FALLBACK)
    save_anatomy(entries, out_path)
    n_real = sum(1 for _, s in entries.values() if s != FALLBACK)
    print(f"[anatomy] {len(entries)} images, {n_real} segmented, {len(entries) - n_real} fallback -> {out_path}")
    return entries
