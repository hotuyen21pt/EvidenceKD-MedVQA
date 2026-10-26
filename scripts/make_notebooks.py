"""Generate the Kaggle notebooks. Never edit the .ipynb files by hand.

    python scripts/make_notebooks.py          # writes notebooks/*.ipynb
    python scripts/make_notebooks.py --check  # exit 1 if the files on disk differ
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import notebook_cells as nc  # noqa: E402

TRAIN_PATH = os.path.join(nc.REPO, "notebooks", "kaggle_stage0_detector.ipynb")
PREDICT_PATH = os.path.join(nc.REPO, "notebooks", "kaggle_stage0_predict.ipynb")

MD_INTRO_TRAIN = """
# EvidenceKD-MedVQA — giai đoạn 0: detector trên VinDr-CXR (Kaggle)

Notebook này huấn luyện **Faster R-CNN R50-FPN** (14 lớp VinDr-CXR-VQA) — nhánh
quyết định đầu tiên của thiết kế EvidenceKD — rồi chấm nó bằng **chính scorer
Bảng 4–6 của VinDR-CXR-VQA** (dòng B7: hộp theo tên finding trong câu hỏi).

**Cần Add Input trước khi chạy:**

| # | Input | Ở đâu | Bắt buộc |
|---|---|---|---|
| 1 | `evidencekd-kaggle-code.zip` (code + `data/processed`) | Datasets → New Dataset → upload zip | có |
| 2 | Ảnh VinDr: dataset `vpa-repo` đã có, **hoặc** `evidencekd-kaggle-images.zip` | Add Input → Datasets | có |
| 3 | `evidencekd-kaggle-weights.zip` (COCO Faster R-CNN, 160 MB) | Add Input → Datasets | nên có (không có thì train từ đầu) |
| 4 | Output phiên trước `evidencekd-detector.zip` | Add Input → Notebooks/Output | chỉ khi resume |

Settings → **Accelerator: GPU T4 x2** (hoặc P100). Internet có thể tắt — cell
dưới tự phát hiện. Rồi **Run all**.

Ước lượng: 15.000 ảnh ở 512 px ≈ 45–60 phút/epoch trên T4; 8 epoch ≈ 1 phiên.
Checkpoint được ghi **mỗi epoch** và cell train tự resume từ checkpoint mới nhất
tìm thấy trong input, nên có thể chia nhiều phiên (Save Version → Save & Run All,
rồi Add Input output của phiên trước).
"""

SETUP_TRAIN = """
# Để trống để tự dò; hoặc dán đường dẫn /kaggle/input/... tương ứng
BUNDLE  = ""   # thư mục chứa src/evidencekd (code bundle)
IMAGES  = ""   # thư mục chứa dataset/train_data/train
WEIGHTS = ""   # file fasterrcnn_resnet50_fpn_coco-*.pth

EPOCHS    = None   # None = theo configs/detector.yaml (8)
MAX_STEPS = -1     # >0 để chạy thử ngắn
LIMIT     = None   # số ảnh (smoke test), None = tất cả
RESUME    = True
OFFLINE   = None   # None = tự phát hiện; True/False để ép
""" + nc.FINDERS + nc.CODE_CHECK + nc.RESOLVE

MD_TRAIN = """
## 3. Train

`--resume auto` lấy `checkpoint-N.pt` mới nhất trong thư mục output (kể cả bản
được copy từ input của phiên trước). Loss từng 50 bước ghi vào `train_log.jsonl`.
"""

MD_EVAL = """
## 4. Dự đoán và chấm

Detector chạy trên val (659 ảnh, đúng tập paper chấm) và test (300 ảnh, tập con
của val như Bảng 3 của paper — y hệt VinDR-CXR-VQA). `evidencekd.eval.stage0`
tính mAP, Grounding@IoU≥0,5 (Bảng B của spec), rồi dịch hộp thành chuỗi
`"{finding} <loc_...>."` và gọi `vindr_cxr_vqa.evaluate` — Bảng 4–6 y hệt paper.
"""

MD_INTRO_PREDICT = """
# EvidenceKD-MedVQA — giai đoạn 0: chạy detector đã train

Add Input: (1) `evidencekd-kaggle-code.zip`, (2) ảnh VinDr (`vpa-repo` hoặc
`evidencekd-kaggle-images.zip`), (3) output của notebook train
(`evidencekd-detector.zip`, chứa `detector/detector.pt`). GPU không bắt buộc
nhưng nhanh hơn nhiều.
"""

SETUP_PREDICT = """
BUNDLE  = ""
IMAGES  = ""
WEIGHTS = ""          # không cần cho predict
CHECKPOINT = ""       # để trống: tự tìm detector.pt trong input
EPOCHS = None
OFFLINE = None
""" + nc.FINDERS + nc.CODE_CHECK + nc.RESOLVE

ASK_HEADER = """
# Ảnh của anh: đường dẫn PNG/JPG bất kỳ. Với ảnh dataset, ORIGINAL_SIZE = (width, height)
# DICOM gốc để <loc> ra đúng hệ; với ảnh ngoài để None (coi ảnh là gốc).
IMAGE_PATH = sorted(glob.glob(os.path.join(cfg["paths"]["dataset_dir"], "train_data/train/*.png")))[0]
ORIGINAL_SIZE = None
"""


def build_train() -> dict:
    return nc.notebook([
        nc.md(MD_INTRO_TRAIN),
        nc.md("## 0. Setup — tìm input, kiểm tra code, copy vào /kaggle/working"),
        nc.code(SETUP_TRAIN),
        nc.code(nc.OFFLINE),
        nc.code(nc.INSTALL),
        nc.md("## 1. Config cho Kaggle"),
        nc.code(nc.CONFIG),
        nc.md("## 2. Preflight và tự kiểm tra cầu nối sang scorer VinDr"),
        nc.code(nc.PREFLIGHT),
        nc.code(nc.VERIFY),
        nc.md(MD_TRAIN),
        nc.code(nc.TRAIN),
        nc.code(nc.CHECKPOINTS),
        nc.md(MD_EVAL),
        nc.code(nc.PREDICT),
        nc.code(nc.EVAL),
        nc.code(nc.COMPARE),
        nc.md("## 5. Hình: hộp bác sĩ (xanh) và detector (đỏ)"),
        nc.code(nc.VIS),
        nc.md("## 6. Lưu output"),
        nc.code(nc.SAVE),
    ])


def build_predict() -> dict:
    return nc.notebook([
        nc.md(MD_INTRO_PREDICT),
        nc.md("## 0. Setup"),
        nc.code(SETUP_PREDICT),
        nc.code(nc.OFFLINE),
        nc.code(nc.INSTALL),
        nc.code(nc.CONFIG),
        nc.md("## 1. Checkpoint"),
        nc.code(nc.FIND_DETECTOR),
        nc.md("## 2. Dự đoán val + test và chấm"),
        nc.code(nc.PREDICT),
        nc.code(nc.EVAL),
        nc.code(nc.COMPARE),
        nc.md("## 3. Hỏi trên một ảnh"),
        nc.code(ASK_HEADER + nc.ASK),
        nc.md("## 4. Hình"),
        nc.code(nc.VIS),
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    targets = {TRAIN_PATH: build_train(), PREDICT_PATH: build_predict()}
    if args.check:
        bad = []
        for path, nb in targets.items():
            on_disk = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else None
            if on_disk != nb:
                bad.append(path)
        for p in bad:
            print("out of date:", p)
        return 1 if bad else 0
    for path, nb in targets.items():
        nc.write(nb, path)
        print("wrote", path, f"({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
