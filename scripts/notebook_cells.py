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
    "src/evidencekd/inference/pipeline.py": ["def run_split", "def resolve_generator_config"],
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


# ===========================================================================
# Full-pipeline notebook cells (stages 0-3 + evaluation in one session chain)
# ===========================================================================
FULL_NEEDS = r'''
NEEDS.update({
    "src/evidencekd/models/decision_branch.py": ["def train", "def predict"],
    "src/evidencekd/data/build_regions.py": ["def main"],
    "src/evidencekd/inference/pipeline.py": ["def run_split", "def tune_thresholds"],
    "src/evidencekd/inference/oracle_decisions.py": ["def write_oracle"],
    "src/evidencekd/eval/tables_spec.py": ["def compute"],
    "src/evidencekd/eval/vindr_tables.py": ["def evaluate"],
    "src/evidencekd/eval/medgemma_baseline.py": ["def baseline_rows", "def adapter_for"],
    "src/evidencekd/teacher/kd_r_targets.py": ["def generate_shard", "def gate_shards"],
    "src/evidencekd/teacher/kd_c_targets.py": ["def generate_shard"],
    "src/evidencekd/models/generator_s.py": ["def train"],
    "src/evidencekd/models/verifier.py": ["def train"],
})
'''

FULL_MODELS = r'''
# Optional inputs for the later stages. Leave "" to auto-detect; a stage is
# skipped (with a message) when its model is not attached.
def _find_dir(markers, prefer=None):
    for m in markers:
        hits = find_inputs(m)
        if prefer:
            hits = sorted(hits, key=lambda h: (prefer not in h.lower(), h.count("/"), h))
        if hits:
            return hits[0]
    return ""

MEDGEMMA = MEDGEMMA or _find_dir(["config.json"], prefer="medgemma")
if MEDGEMMA and not glob.glob(os.path.join(MEDGEMMA, "*.safetensors")):
    MEDGEMMA = ""
ADAPTER = ADAPTER or next((os.path.dirname(p) for p in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
                           if "checkpoint-" not in p), "")
if not ADAPTER:
    up = unpack("adapter_config.json")
    if up:
        ADAPTER = next((os.path.dirname(p) for p in glob.glob(os.path.join(up, "**/adapter_config.json"), recursive=True)
                        if "checkpoint-" not in p), "")
GEN_S_MODEL = GEN_S_MODEL or _find_dir(["spiece.model"], prefer="t5")
print("MEDGEMMA   ", MEDGEMMA or "(none -> teacher / KD / generator L skipped)")
print("ADAPTER    ", ADAPTER or "(none -> teacher skipped)")
print("GEN_S_MODEL", GEN_S_MODEL or "(none -> generator S skipped; template generator only)")
'''

FULL_CONFIG = r'''
from evidencekd.utils.config import load_config, dump_config
OUT = "/kaggle/working/outputs"; REPORTS = "/kaggle/working/reports"; DERIVED = "/kaggle/working/derived"
CACHE = "/kaggle/working/cache"
for d in (OUT, REPORTS, DERIVED, CACHE):
    os.makedirs(d, exist_ok=True)
# bring back everything a previous session produced (evidencekd-full.zip attached as input)
for sub in ("outputs", "reports", "derived", "cache"):
    for p in glob.glob(f"/kaggle/input/**/evidencekd_full/{sub}", recursive=True):
        shutil.copytree(p, f"/kaggle/working/{sub}", dirs_exist_ok=True); print("restored", p)
for p in glob.glob("/kaggle/input/**/detector/detector.pt", recursive=True):
    os.makedirs(os.path.join(OUT, "detector"), exist_ok=True)
    if not os.path.exists(os.path.join(OUT, "detector", "detector.pt")):
        shutil.copy2(p, os.path.join(OUT, "detector")); print("restored", p)

vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
CFG = {}
common = {"dataset_dir": os.path.join(IMAGES_ROOT, "dataset"), "processed_dir": os.path.join(PROCESSED_ROOT, "data", "processed"),
          "output_dir": OUT, "reports_dir": REPORTS, "derived_dir": DERIVED, "cache_dir": CACHE}
for name in ("detector", "decision", "teacher", "generator"):
    cfg = load_config(os.path.join(WORK, "configs", f"{name}.yaml"))
    for k, v in common.items():
        cfg["paths"][k] = v
    if name == "detector":
        cfg["model"]["coco_weights"] = WEIGHTS_PATH
        cfg["train"]["batch_size"] = 8 if vram >= 14 else 4
        cfg["data"]["image_source"] = "all" if n_png >= 10000 else "vqa_train"
    if name == "decision":
        cfg["paths"]["detector_checkpoint"] = os.path.join(OUT, "detector", "detector.pt")
        cfg["paths"]["kd_r_cache"] = os.path.join(CACHE, "kd_r")
        cfg["train"]["batch_size"] = 16 if vram >= 14 else 8
    if name == "teacher":
        cfg["model"]["name"] = MEDGEMMA or cfg["model"]["name"]
        cfg["model"]["adapter"] = ADAPTER
        cfg["kd_r"]["subset_fraction"] = KD_R_SUBSET
    if name == "generator":
        cfg["generator_s"]["model"] = GEN_S_MODEL or cfg["generator_s"]["model"]
        cfg["generator_l"]["model"] = MEDGEMMA or cfg["generator_l"]["model"]
    CFG[name] = os.path.join(WORK, "configs", f"kaggle_{name}.yaml")
    dump_config(cfg, CFG[name])
print(CFG)
assert os.path.isfile(os.path.join(common["processed_dir"], "train.jsonl"))

def sh(*args, check=True):
    cmd = [sys.executable, "-m", *[str(a) for a in args]]
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"  -> exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(cmd)}")
    return r.returncode
'''

FULL_STAGE0 = r'''
DET = os.path.join(OUT, "detector", "detector.pt")
if os.path.isfile(DET) and not RETRAIN_DETECTOR:
    print("detector.pt present -> stage 0 training skipped")
else:
    sh("evidencekd.preflight", "--config", CFG["detector"])
    sh("evidencekd.models.detector", "train", "--config", CFG["detector"], "--output-dir", os.path.join(OUT, "detector"),
       "--resume", "auto", *(["--epochs", str(DETECTOR_EPOCHS)] if DETECTOR_EPOCHS else []),
       *(["--limit", str(SMOKE)] if SMOKE else []))
for split in ("val", "test"):
    p = os.path.join(OUT, f"detections_{split}.json")
    if not os.path.isfile(p) or RETRAIN_DETECTOR:
        sh("evidencekd.models.detector", "predict", "--config", CFG["detector"], "--checkpoint", DET, "--split", split, "--out", p)
    sh("evidencekd.eval.stage0", "--config", CFG["detector"], "--detections", p, "--split", split, "--out-dir", REPORTS)
'''

FULL_DERIVED = r'''
try:
    import torchxrayvision  # noqa: F401
    anat_flag = []
except Exception:
    anat_flag = ["--no-anatomy"]
    print("torchxrayvision not available -> fallback anatomy (no zone/extent facts)")
sh("evidencekd.data.build_regions", "--config", CFG["decision"], *anat_flag)
# self-test of the whole inference chain with the answer key: must be ~perfect
sh("evidencekd.inference.oracle_decisions", "--config", CFG["decision"], "--split", "val")
sh("evidencekd.inference.pipeline", "--config", CFG["decision"], "--split", "val", "--generator", "template",
   "--decision", os.path.join(OUT, "decision_val_oracle.jsonl"), "--regions", os.path.join(DERIVED, "regions_val_annotation.json"), "--tag", "oracle")
sh("evidencekd.eval.vindr_tables", "--config", CFG["decision"], "--pred", os.path.join(REPORTS, "preds_val_oracle.jsonl"), "--split", "val")
'''

FULL_STAGE1 = r'''
DEC_DIR = os.path.join(OUT, "decision")
S1 = os.path.join(DEC_DIR, "decision_stage1.pt")
if not os.path.isfile(S1) or RETRAIN_DECISION:
    sh("evidencekd.models.decision_branch", "train", "--config", CFG["decision"], "--stage", "1", "--output-dir", DEC_DIR,
       "--resume", "auto", *(["--epochs", str(DECISION_EPOCHS)] if DECISION_EPOCHS else []), *(["--limit", str(SMOKE)] if SMOKE else []))
for split in ("val", "test"):
    sh("evidencekd.models.decision_branch", "predict", "--config", CFG["decision"], "--checkpoint", S1, "--split", split,
       "--out", os.path.join(OUT, f"decision_{split}.jsonl"))
sh("evidencekd.inference.pipeline", "--config", CFG["decision"], "--tune")
'''

FULL_TEACHER = r'''
HAVE_TEACHER = bool(MEDGEMMA and ADAPTER) and RUN_TEACHER
if not HAVE_TEACHER:
    print("teacher skipped (needs MedGemma + the VinDr adapter attached, and RUN_TEACHER=True)")
else:
    t_start = time.time()
    def budget_left():
        return (time.time() - t_start) / 60 < TEACHER_BUDGET_MIN
    for shard in range(KD_SHARDS):
        if not budget_left():
            print("KD-R: time budget reached; resume next session"); break
        sh("evidencekd.teacher.kd_r_targets", "--config", CFG["teacher"], "--shard", shard, "--n-shards", KD_SHARDS,
           *(["--limit", str(SMOKE)] if SMOKE else []), check=False)
    sh("evidencekd.teacher.kd_r_targets", "--config", CFG["teacher"], "--gate", check=False)
    sh("evidencekd.teacher.kd_r_targets", "--config", CFG["teacher"], "--sensitivity", check=False)
    for shard in range(KD_SHARDS):
        if not budget_left():
            print("KD-C: time budget reached; resume next session"); break
        sh("evidencekd.teacher.kd_c_targets", "--config", CFG["teacher"], "--shard", shard, "--n-shards", KD_SHARDS,
           *(["--limit", str(SMOKE)] if SMOKE else []), check=False)
'''

FULL_STAGE2 = r'''
FINAL_DEC = S1
if glob.glob(os.path.join(CACHE, "kd_r", "*.jsonl")):
    S2 = os.path.join(DEC_DIR, "decision_stage2.pt")
    if not os.path.isfile(S2) or RETRAIN_DECISION:
        sh("evidencekd.models.decision_branch", "train", "--config", CFG["decision"], "--stage", "2", "--output-dir", DEC_DIR,
           "--init", S1, "--resume", "auto", *(["--limit", str(SMOKE)] if SMOKE else []))
    for split in ("val", "test"):
        sh("evidencekd.models.decision_branch", "predict", "--config", CFG["decision"], "--checkpoint", S2, "--split", split,
           "--out", os.path.join(OUT, f"decision_{split}.jsonl"))
    sh("evidencekd.inference.pipeline", "--config", CFG["decision"], "--tune")
    FINAL_DEC = S2
else:
    print("no KD-R cache -> stage 2 skipped; the stage-1 decision branch is evaluated")
print("decision checkpoint for evaluation:", FINAL_DEC)
'''

FULL_STAGE3 = r'''
VERIFIER = ""
GEN_S = ""
V_PT = os.path.join(OUT, "verifier", "verifier.pt")
GS_FINAL = os.path.join(OUT, "generator_s", "final")
if os.path.isfile(V_PT) and not RETRAIN_GENERATOR:
    VERIFIER = V_PT; print("verifier.pt da co -> bo qua train")
elif glob.glob(os.path.join(CACHE, "kd_c", "*.jsonl")):
    sh("evidencekd.models.verifier", "train", "--config", CFG["generator"], "--output-dir", os.path.join(OUT, "verifier"), check=False)
    if os.path.isfile(V_PT):
        VERIFIER = V_PT
if os.path.isdir(GS_FINAL) and not RETRAIN_GENERATOR:
    GEN_S = GS_FINAL; print("generator_s/final da co -> bo qua train")
elif GEN_S_MODEL:
    rc = sh("evidencekd.models.generator_s", "train", "--config", CFG["generator"], "--output-dir", os.path.join(OUT, "generator_s"),
            *(["--limit", str(SMOKE)] if SMOKE else []), check=False)
    if rc == 0 and os.path.isdir(GS_FINAL):
        GEN_S = GS_FINAL
print("verifier:", VERIFIER or "rules only", "| generator S:", GEN_S or "not trained (template generator)")
'''

FULL_BASELINES = r'''
# B4 (MedGemma + adapter VinDr) va pretrained (cung base, khong adapter) CHAY TREN
# CHINH SPLIT nay -- de "co cai thien khong" tra loi duoc, thay vi dat canh hang so
# val-659 cua paper. Ton kem: ~959 sinh/tag cho ca hai split, nen co resume.
BASELINES = []
if not RUN_BASELINES:
    print("baselines bi tat (RUN_BASELINES=False)")
elif not MEDGEMMA:
    print("khong co MedGemma -> khong co B4/pretrained; bang so sanh chi con hang so val-659 cua paper")
else:
    BASELINES = ["pretrained"] + (["b4"] if ADAPTER else [])
    if not ADAPTER:
        print("co MedGemma nhung thieu adapter -> chi chay pretrained, KHONG co B4")
    for split in BASELINE_SPLITS:
        for tag in BASELINES:
            sh("evidencekd.eval.medgemma_baseline", "--config", CFG["teacher"], "--split", split, "--tag", tag,
               "--batch-size", BASELINE_BATCH, *(["--limit", str(SMOKE)] if SMOKE else []), check=False)
print("baselines:", BASELINES or "(none)")
'''

FULL_EVAL = r'''
preds = {}
for split in ("val", "test"):
    gens = [("template", [])] + ([("s", ["--generator-dir", GEN_S, "--generator-config", CFG["generator"]])] if GEN_S else [])
    for gen, extra in gens:
        sh("evidencekd.inference.pipeline", "--config", CFG["decision"], "--split", split, "--generator", gen, *extra,
           *(["--verifier", VERIFIER] if VERIFIER else []))
        sh("evidencekd.eval.tables_spec", "--config", CFG["decision"], "--pipeline", os.path.join(OUT, f"pipeline_{split}_{gen}.jsonl"), "--split", split)
        preds.setdefault(split, []).append(os.path.join(REPORTS, f"preds_{split}_{gen}.jsonl"))
    # B7 (ablation noi bo) va hai baseline MedGemma, deu tren split nay
    for tag in ("b7", "pretrained", "b4"):
        p = os.path.join(REPORTS, f"preds_{split}_{tag}.jsonl")
        if os.path.isfile(p):
            preds[split].append(p)
    sh("evidencekd.eval.vindr_tables", "--config", CFG["decision"], "--pred", *preds[split], "--split", split, check=False)
sh("evidencekd.eval.faithfulness", "--config", CFG["decision"], "--checkpoint", FINAL_DEC,
   "--pipeline", os.path.join(OUT, "pipeline_test_template.jsonl"), "--split", "test", "--limit", "300", check=False)
'''

FULL_SAVE = r'''
stage = "/kaggle/working/evidencekd_full"
shutil.rmtree(stage, ignore_errors=True)
for sub in ("reports", "derived", "cache"):
    if os.path.isdir(f"/kaggle/working/{sub}"):
        shutil.copytree(f"/kaggle/working/{sub}", os.path.join(stage, sub))
os.makedirs(os.path.join(stage, "outputs"), exist_ok=True)
keep = ["detector/detector.pt", "decision/decision_stage1.pt", "decision/decision_stage2.pt", "decision/vocab.json",
        "verifier/verifier.pt", "generator_s/final", "detections_val.json", "detections_test.json",
        "decision_val.jsonl", "decision_test.jsonl"] + [os.path.basename(p) for p in glob.glob(os.path.join(OUT, "pipeline_*.jsonl"))]
for k in keep:
    p = os.path.join(OUT, k)
    if os.path.isdir(p):
        shutil.copytree(p, os.path.join(stage, "outputs", k), dirs_exist_ok=True)
    elif os.path.isfile(p):
        os.makedirs(os.path.dirname(os.path.join(stage, "outputs", k)), exist_ok=True); shutil.copy2(p, os.path.join(stage, "outputs", k))
zip_path = shutil.make_archive("/kaggle/working/evidencekd-full", "zip", "/kaggle/working", "evidencekd_full")
print(f"{os.path.getsize(zip_path) / 2**20:.0f} MB ->", zip_path)
print("Phiên sau: Add Input file này để resume mọi giai đoạn (checkpoint, cache KD, report).")
for p in sorted(glob.glob(os.path.join(REPORTS, "*.json"))):
    print(" ", os.path.basename(p))
'''
