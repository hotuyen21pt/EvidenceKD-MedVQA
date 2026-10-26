"""Cell sources shared by the Kaggle notebook generators.

Same pattern as ``VinDR-CXR-VQA/scripts/notebook_cells.py``: the notebooks are
GENERATED (``make_notebooks.py``) and never hand-edited; ``tests/test_notebooks.py``
fails when the ``.ipynb`` on disk drifts from what the generator produces.

Every cell is plain Python that runs inside Kaggle. Markdown is written in
Vietnamese, code comments in English.
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 0. Finders — locate attached inputs however Kaggle nested them
# ---------------------------------------------------------------------------
FINDERS = r'''
import os, sys, glob, json, shutil, zipfile, subprocess, time

def find_inputs(marker, root="/kaggle/input", max_depth=6):
    """Every directory under root that contains `marker` (a relative path)."""
    hits = []
    if not os.path.isdir(root):
        return hits
    root_depth = root.rstrip("/").count("/")
    for cur, dirs, files in os.walk(root):
        depth = cur.count("/") - root_depth
        if depth >= max_depth:
            dirs[:] = []
        # never descend into the image folders themselves (thousands of files)
        dirs[:] = [d for d in dirs if d not in ("train", "test")]
        if os.path.exists(os.path.join(cur, marker)):
            hits.append(cur)
    hits.sort(key=lambda p: (p.count("/"), p))
    return hits

def tree(root="/kaggle/input", max_depth=4, limit=60):
    """Print what is attached, shallow, so a wrong attachment is obvious."""
    n = 0
    root_depth = root.rstrip("/").count("/")
    for cur, dirs, files in os.walk(root):
        depth = cur.count("/") - root_depth
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in ("train", "test")]
        print("  " * depth + os.path.basename(cur) + "/" + ("" if depth else f"   ({cur})"))
        for f in sorted(files)[:5]:
            print("  " * (depth + 1) + f)
        if len(files) > 5:
            print("  " * (depth + 1) + f"... {len(files) - 5} more")
        n += 1
        if n >= limit:
            print("  ..."); break

def unpack(marker, dest="/kaggle/working/unpacked"):
    """Extract any attached .zip whose listing contains `marker`; return the dir."""
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                if not any(n.startswith(marker) or ("/" + marker) in n for n in names):
                    continue
                out = os.path.join(dest, os.path.splitext(os.path.basename(z))[0])
                if not os.path.isdir(out):
                    print(f"unpacking {z} -> {out}")
                    zf.extractall(out)
                return out
        except zipfile.BadZipFile:
            continue
    return None

def locate(marker, what):
    hits = find_inputs(marker)
    if not hits:
        up = unpack(marker)
        if up:
            hits = find_inputs(marker, root=up) or [up]
    if not hits:
        tree()
        raise SystemExit(f"Không tìm thấy {what} (cần thư mục chứa `{marker}`). "
                         f"Hãy Add Input dataset tương ứng rồi chạy lại.")
    return hits[0]
'''

# ---------------------------------------------------------------------------
# 1. Code freshness check
# ---------------------------------------------------------------------------
CODE_CHECK = r'''
NEEDS = {
    "src/evidencekd/models/detector.py": ["def train", "def predict", "def latest_checkpoint"],
    "src/evidencekd/eval/stage0.py": ["def main", "def _write_vindr_config"],
    "src/evidencekd/eval/vindr_bridge.py": ["def detector_rows", "def oracle_rows", "def grounding_success"],
    "src/evidencekd/eval/detection.py": ["def mean_average_precision"],
    "src/evidencekd/data/detection_dataset.py": ["def build_index", "class VinDrDetectionDataset"],
    "src/evidencekd/data/vindr_processed.py": ["def load_images", "def splits_from_images"],
    "src/evidencekd/preflight.py": ["def run"],
    "vindr_cxr_vqa/evaluate.py": ["def eval_boxes", "def eval_vqa"],
    "vindr_cxr_vqa/vindr.py": ["def fuse_boxes", "def load_images"],
}

def missing_markers(root):
    out = []
    for rel, snippets in NEEDS.items():
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            out.append(f"{rel}: file missing"); continue
        src = open(p, encoding="utf-8").read()
        for s in snippets:
            if s not in src:
                out.append(f"{rel}: `{s}` not found")
    return out
'''

# ---------------------------------------------------------------------------
# 2. Resolve inputs and copy code into a writable work dir
# ---------------------------------------------------------------------------
RESOLVE = r'''
WORK = "/kaggle/working/evidencekd"

CODE_ROOT = BUNDLE or next((h for h in find_inputs("src/evidencekd") if not missing_markers(h)), None)
if CODE_ROOT is None:
    CODE_ROOT = locate("src/evidencekd", "code bundle (evidencekd-kaggle-code.zip)")
bad = missing_markers(CODE_ROOT)
if bad:
    raise SystemExit("Code đính kèm đã CŨ hoặc thiếu — dựng lại bằng "
                     "`python scripts/make_kaggle_bundle.py --part code` rồi upload phiên bản mới:\n  " + "\n  ".join(bad))

PROCESSED_ROOT = locate("data/processed/train.jsonl", "dữ liệu đã dựng (data/processed)")
IMAGES_ROOT = IMAGES or locate("dataset/train_data/train", "ảnh VinDr (dataset/train_data/train)")

WEIGHTS_PATH = WEIGHTS
if not WEIGHTS_PATH:
    cands = glob.glob("/kaggle/input/**/fasterrcnn_resnet50_fpn_coco*.pth", recursive=True)
    if not cands:
        up = unpack("weights/")
        if up:
            cands = glob.glob(os.path.join(up, "**/*.pth"), recursive=True)
    WEIGHTS_PATH = cands[0] if cands else ""

if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)
for item in ("src", "configs", "scripts", "vindr_cxr_vqa", "pyproject.toml", "README.md", "docs"):
    s = os.path.join(CODE_ROOT, item)
    if os.path.isdir(s):
        shutil.copytree(s, os.path.join(WORK, item), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    elif os.path.isfile(s):
        shutil.copy2(s, WORK)
os.chdir(WORK)
for p in (os.path.join(WORK, "src"), WORK):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.path.join(WORK, "src") + os.pathsep + WORK + os.pathsep + os.environ.get("PYTHONPATH", "")

n_png = len(glob.glob(os.path.join(IMAGES_ROOT, "dataset/train_data/train/*.png")))
print("CODE_ROOT      ", CODE_ROOT)
print("PROCESSED_ROOT ", PROCESSED_ROOT)
print("IMAGES_ROOT    ", IMAGES_ROOT, f"({n_png} PNG)")
print("WEIGHTS_PATH   ", WEIGHTS_PATH or "(none — backbone from scratch)")
print("WORK           ", WORK)
'''

# ---------------------------------------------------------------------------
# 3. Offline detection + install
# ---------------------------------------------------------------------------
OFFLINE = r'''
import socket

def has_internet(host="pypi.org", port=443, timeout=4):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False

OFFLINE = (not has_internet()) if OFFLINE is None else OFFLINE
print("OFFLINE =", OFFLINE)
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
'''

INSTALL = r'''
if not OFFLINE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--retries", "1", "--timeout", "15", "pyyaml"], check=False)
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__,
      "cuda", torch.cuda.is_available(),
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB" if torch.cuda.is_available() else "")
'''

# ---------------------------------------------------------------------------
# 4. Config rewrite for Kaggle
# ---------------------------------------------------------------------------
CONFIG = r'''
from evidencekd.utils.config import load_config, dump_config

cfg = load_config(os.path.join(WORK, "configs", "detector.yaml"))
cfg["paths"]["dataset_dir"] = os.path.join(IMAGES_ROOT, "dataset")
cfg["paths"]["processed_dir"] = os.path.join(PROCESSED_ROOT, "data", "processed")
cfg["paths"]["output_dir"] = "/kaggle/working/outputs"
cfg["paths"]["reports_dir"] = "/kaggle/working/reports"
cfg["model"]["coco_weights"] = WEIGHTS_PATH

vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
cfg["train"]["batch_size"] = 8 if vram >= 14 else 4
cfg["train"]["num_workers"] = 2
cfg["train"]["amp"] = bool(torch.cuda.is_available())
# train on every VinDr-CXR image attached when the full 15,000 are there,
# else on the 3,735 VQA training images only
cfg["data"]["image_source"] = "all" if n_png >= 10000 else "vqa_train"
if EPOCHS is not None:
    cfg["train"]["epochs"] = int(EPOCHS)

CFG_PATH = os.path.join(WORK, "configs", "kaggle.yaml")
dump_config(cfg, CFG_PATH)
print(open(CFG_PATH).read())
assert os.path.isdir(cfg["paths"]["dataset_dir"]), cfg["paths"]["dataset_dir"]
assert os.path.isfile(os.path.join(cfg["paths"]["processed_dir"], "train.jsonl"))
'''

PREFLIGHT = r'''
r = subprocess.run([sys.executable, "-m", "evidencekd.preflight", "--config", CFG_PATH])
if r.returncode != 0:
    raise SystemExit("preflight FAIL — sửa các dòng [FAIL] ở trên trước khi train")
'''

VERIFY = r'''
# Bridge self-test: the answer key replayed through evidencekd -> vindr scorer
# must score VQA precision 1.0 and localization 100% at IoU >= 0.5.
r = subprocess.run([sys.executable, "-m", "evidencekd.eval.stage0", "--config", CFG_PATH,
                    "--oracle", "--split", "val", "--out-dir", "/kaggle/working/reports"])
if r.returncode != 0:
    raise SystemExit("Cầu nối sang scorer VinDr bị hỏng — không train tiếp")
manifest = os.path.join(WORK, "reports", "split_manifest.json")
print(open(manifest).read() if os.path.isfile(manifest) else "(no split manifest in bundle)")
'''

# ---------------------------------------------------------------------------
# 5. Train / checkpoints / predict / eval / vis / save
# ---------------------------------------------------------------------------
TRAIN = r'''
OUT = "/kaggle/working/outputs/detector"
os.makedirs(OUT, exist_ok=True)
if RESUME:
    # bring checkpoints from a previous session's output (attached as input)
    prev = sorted(glob.glob("/kaggle/input/**/detector/checkpoint-*.pt", recursive=True)) + \
           sorted(glob.glob("/kaggle/input/**/checkpoint-*.pt", recursive=True))
    seen = set()
    for p in prev:
        if os.path.basename(p) in seen:
            continue
        seen.add(os.path.basename(p))
        dst = os.path.join(OUT, os.path.basename(p))
        if not os.path.exists(dst):
            print("resume: copying", p)
            shutil.copy2(p, dst)
cmd = [sys.executable, "-m", "evidencekd.models.detector", "train", "--config", CFG_PATH,
       "--output-dir", OUT, "--max-steps", str(MAX_STEPS), "--resume", "auto" if RESUME else "none"]
if EPOCHS is not None:
    cmd += ["--epochs", str(EPOCHS)]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
print("$", " ".join(cmd))
t0 = time.time()
subprocess.run(cmd, check=True)
print(f"train took {(time.time() - t0) / 60:.1f} min")
'''

CHECKPOINTS = r'''
for p in sorted(glob.glob(os.path.join(OUT, "*.pt"))):
    print(f"{os.path.getsize(p) / 2**20:8.0f} MB  {p}")
assert os.path.isfile(os.path.join(OUT, "detector.pt")), "detector.pt chưa được ghi"
'''

PREDICT = r'''
CKPT = os.path.join(OUT, "detector.pt")
DETS = {}
for split in ("val", "test"):
    DETS[split] = f"/kaggle/working/outputs/detections_{split}.json"
    subprocess.run([sys.executable, "-m", "evidencekd.models.detector", "predict", "--config", CFG_PATH,
                    "--checkpoint", CKPT, "--split", split, "--out", DETS[split],
                    "--batch-size", str(cfg["train"]["batch_size"])], check=True)
'''

EVAL = r'''
REPORTS = "/kaggle/working/reports"
for split in ("val", "test"):
    subprocess.run([sys.executable, "-m", "evidencekd.eval.stage0", "--config", CFG_PATH,
                    "--detections", DETS[split], "--split", split, "--out-dir", REPORTS], check=True)
'''

COMPARE = r'''
from vindr_cxr_vqa.results import PAPER

def row(name, v, d, l):
    return (f"| {name} | {v['f1']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | "
            f"{d['precision']:.3f} | {d['recall']:.3f} | {d['miou_tp']:.3f} | "
            f"{l['iou_ge_0.5_pct']:.1f} | {l['iou_ge_0.3_pct']:.1f} |")

for split in ("val", "test"):
    rep = json.load(open(os.path.join(REPORTS, f"stage0_{split}_b7.json")))
    t = rep.get("vindr_tables")
    print(f"\n### {split}  (B7 = detector giai đoạn 0, hộp theo tên finding trong câu hỏi)\n")
    print("| Hệ thống | T4 F1 | T4 P | T4 R | T5 P | T5 R | T5 mIoU | T6 IoU≥.5 % | T6 IoU≥.3 % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in ("Pretrained", "Fine-tuned"):
        print(row(f"Paper {name} (val-659)", PAPER["vqa"][name], PAPER["detection"][name], PAPER["localization"][name]))
    if t:
        print(row(f"B7 detector ({split})", t["vqa"], t["detection"], t["localization"]))
    m, g = rep.get("map", {}), rep.get("grounding", {})
    print(f"\nmAP@.5 {m.get('map_50', float('nan')):.3f} · mAP@[.5:.95] {m.get('map_50_95', float('nan')):.3f} · "
          f"Grounding@.5 {g.get('grounding_at_iou', float('nan')):.3f} · mIoU {g.get('miou', float('nan')):.3f}")
'''

VIS = r'''
from PIL import Image, ImageDraw
from evidencekd.data.vindr_processed import load_images, splits_from_images
from evidencekd.eval.vindr_bridge import load_detections

FIG = "/kaggle/working/outputs/figures"; os.makedirs(FIG, exist_ok=True)
images = load_images(cfg["paths"]["processed_dir"])
dets = load_detections(DETS["val"])
picked = sorted(dets, key=lambda k: -len(dets[k]))[:6]
UP = 3
for iid in picked:
    rec = images[iid]
    im = Image.open(os.path.join(cfg["paths"]["dataset_dir"], rec.image_path)).convert("RGB")
    im = im.resize((im.width * UP, im.height * UP))
    dr = ImageDraw.Draw(im)
    for f in rec.boxes_fused:
        b = f.box
        dr.rectangle([b.x_min * rec.fx * UP, b.y_min * rec.fy * UP, b.x_max * rec.fx * UP, b.y_max * rec.fy * UP], outline=(60, 220, 60), width=2)
        dr.text((b.x_min * rec.fx * UP + 2, b.y_min * rec.fy * UP + 2), b.class_name, fill=(60, 220, 60))
    for d in dets[iid]:
        if d.score < cfg["eval"]["score_threshold"]:
            continue
        b = d.box
        dr.rectangle([b.x_min * rec.fx * UP, b.y_min * rec.fy * UP, b.x_max * rec.fx * UP, b.y_max * rec.fy * UP], outline=(240, 60, 60), width=2)
        dr.text((b.x_min * rec.fx * UP + 2, b.y_max * rec.fy * UP - 12), f"{b.class_name} {d.score:.2f}", fill=(240, 60, 60))
    im.save(os.path.join(FIG, f"{iid}.png"))
print("figures ->", FIG, sorted(os.listdir(FIG)))
from IPython.display import display
for iid in picked[:3]:
    display(Image.open(os.path.join(FIG, f"{iid}.png")))
'''

SAVE = r'''
# detector.pt (model only) + the newest checkpoint (for resume) + reports
stage = "/kaggle/working/detector_bundle"
shutil.rmtree(stage, ignore_errors=True); os.makedirs(os.path.join(stage, "detector"))
shutil.copy2(os.path.join(OUT, "detector.pt"), os.path.join(stage, "detector"))
from evidencekd.models.detector import latest_checkpoint
ck = latest_checkpoint(OUT)
if ck:
    shutil.copy2(ck, os.path.join(stage, "detector"))
for extra in ("train_config.json", "train_log.jsonl", "train_index_summary.json"):
    p = os.path.join(OUT, extra)
    if os.path.isfile(p):
        shutil.copy2(p, os.path.join(stage, "detector"))
shutil.copytree("/kaggle/working/reports", os.path.join(stage, "reports"), dirs_exist_ok=True)
for split in ("val", "test"):
    shutil.copy2(DETS[split], stage)
zip_path = shutil.make_archive("/kaggle/working/evidencekd-detector", "zip", stage)
print(f"{os.path.getsize(zip_path) / 2**20:.0f} MB ->", zip_path)
print("\nTải file này ở tab Output; phiên sau Add Input nó để resume hoặc chạy notebook predict.")
for p in sorted(glob.glob("/kaggle/working/reports/*.json")):
    print(" ", p)
'''

# ---------------------------------------------------------------------------
# Predict-notebook specific
# ---------------------------------------------------------------------------
FIND_DETECTOR = r'''
CKPT = CHECKPOINT
if not CKPT:
    cands = glob.glob("/kaggle/input/**/detector.pt", recursive=True)
    if not cands:
        up = unpack("detector/")
        if up:
            cands = glob.glob(os.path.join(up, "**/detector.pt"), recursive=True)
    if not cands:
        cands = sorted(glob.glob("/kaggle/input/**/checkpoint-*.pt", recursive=True))
    if not cands:
        tree()
        raise SystemExit("Không tìm thấy detector.pt / checkpoint-*.pt — Add Input output của notebook train (evidencekd-detector.zip)")
    CKPT = cands[0]
print("CHECKPOINT", CKPT, f"({os.path.getsize(CKPT) / 2**20:.0f} MB)")
OUT = os.path.dirname(CKPT)
'''

ASK = r'''
# Run the detector on one image of your own (or a dataset image) and print
# every box in original-pixel space and as the <loc_...> token VinDr uses.
from PIL import Image
from evidencekd.models.detector import load_for_inference
from evidencekd.labels import VQA_CLASSES
from evidencekd.eval.vindr_bridge import loc_token
from evidencekd.data.boxes import Box

model, device = load_for_inference(cfg, CKPT)
img = Image.open(IMAGE_PATH).convert("RGB")
W, H = ORIGINAL_SIZE or img.size
x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(img.height, img.width, 3).permute(2, 0, 1).float().div(255).to(device)
with torch.no_grad():
    p = model([x])[0]
fx, fy = img.width / W, img.height / H
print(f"{IMAGE_PATH}  ({img.width}x{img.height}, original {W}x{H})")
for b, l, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
    if s < cfg["eval"]["score_threshold"]:
        continue
    box = Box(b[0] / fx, b[1] / fy, b[2] / fx, b[3] / fy, class_name=VQA_CLASSES[l - 1])
    print(f"  {box.class_name:20s} {s:.2f}  px {[round(v) for v in (box.x_min, box.y_min, box.x_max, box.y_max)]}  {loc_token(box, W, H)}")
'''


# ---------------------------------------------------------------------------
# notebook writers
# ---------------------------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    # No `kaggle` metadata block and no accelerator key on purpose: importing
    # a notebook that carries them resets the user's attached inputs.
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(nb: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
