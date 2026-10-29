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
FULL_PATH = os.path.join(nc.REPO, "notebooks", "kaggle_full_pipeline.ipynb")

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


MD_INTRO_FULL = """
# EvidenceKD-MedVQA — full pipeline trên Kaggle (giai đoạn 0 → 3 → chấm bảng)

Một notebook chạy lần lượt: **detector** (giai đoạn 0) → vùng ứng viên + giải phẫu +
mẫu âm → **self-test oracle** → **decision branch** giai đoạn 1 → chọn ngưỡng →
**teacher MedGemma**: KD-R + KD-C (nếu attach MedGemma + adapter VinDr) →
giai đoạn 2 (KD-R) → **verifier** + **generator S** (nếu attach flan-T5) →
pipeline suy luận → **Bảng A–C, F** của spec và **Bảng 4–6** của VinDr → zip.

Mỗi giai đoạn tự bỏ qua khi output đã có (resume), nên chạy nhiều phiên:
Save Version → Save & Run All, rồi phiên sau Add Input `evidencekd-full.zip`.

| # | Input | Bắt buộc |
|---|---|---|
| 1 | `evidencekd-kaggle-code.zip` | có |
| 2 | ảnh VinDr: `vpa-repo` hoặc `evidencekd-kaggle-images(-all).zip` | có |
| 3 | `evidencekd-kaggle-weights.zip` (COCO Faster R-CNN) | nên |
| 4 | Kaggle Models → MedGemma 1.5 4B-it (Transformers) | cho teacher / KD |
| 5 | adapter VinDr `medgemma-lora.zip` (output notebook train của VinDR-CXR-VQA) | cho teacher / KD |
| 6 | flan-t5-base (upload thư mục HF làm Dataset) | cho generator S |
| 7 | `evidencekd-full.zip` của phiên trước | để resume |

GPU T4 x2 hoặc P100; Internet có thể tắt. Ước lượng: giai đoạn 0 ≈ 1 phiên;
giai đoạn 1 ≈ 2–3 h; teacher KD ≈ 2–3 phiên (có ngân sách thời gian, tự dừng và resume).
"""

SETUP_FULL = """
BUNDLE = ""; IMAGES = ""; WEIGHTS = ""
MEDGEMMA = ""        # thư mục MedGemma (config.json + *.safetensors); "" = tự tìm
ADAPTER = ""         # thư mục adapter VinDr (adapter_config.json); "" = tự tìm
GEN_S_MODEL = ""     # thư mục flan-t5-base; "" = tự tìm

SMOKE = None             # ví dụ 64: chạy thử toàn chuỗi trên vài mẫu
DETECTOR_EPOCHS = None   # None = theo config (8)
DECISION_EPOCHS = None   # None = theo config (6)
RETRAIN_DETECTOR = False
RETRAIN_DECISION = False
RETRAIN_GENERATOR = False   # True = train lai verifier + generator S dù outputs/ đã có
RUN_TEACHER = True
KD_SHARDS = 30
KD_R_SUBSET = 0.5        # 0.5 = nửa số ảnh train (design Q3); 1.0 = tất cả
TEACHER_BUDGET_MIN = 420 # phút tối đa dành cho KD-R + KD-C trong phiên này
RUN_BASELINES = True     # chạy MedGemma pretrained + B4 trên CHÍNH split này, để
                         # bảng 4-6 so sánh được cùng chế độ (spec §"Sau giai đoạn 1")
BASELINE_SPLITS = ("test", "val")   # test (300) trước; val (659) để đặt cạnh số paper
BASELINE_BATCH = 4
OFFLINE = None
EPOCHS = None
""" + nc.FINDERS + nc.CODE_CHECK + nc.FULL_NEEDS + nc.RESOLVE


def build_full() -> dict:
    return nc.notebook([
        nc.md(MD_INTRO_FULL),
        nc.md("## 0. Setup"),
        nc.code(SETUP_FULL),
        nc.code(nc.OFFLINE),
        nc.code(nc.INSTALL),
        nc.code(nc.FULL_MODELS),
        nc.md("## 1. Config cho mọi giai đoạn + khôi phục output phiên trước"),
        nc.code(nc.FULL_CONFIG),
        nc.md("## 2. Giai đoạn 0 — detector (train nếu chưa có) + B7 Bảng 4–6"),
        nc.code(nc.FULL_STAGE0),
        nc.md("## 3. Vùng ứng viên, giải phẫu, mẫu âm, nhãn — và self-test oracle qua scorer VinDr"),
        nc.code(nc.FULL_DERIVED),
        nc.md("## 4. Giai đoạn 1 — decision branch, dự đoán val/test, chọn ngưỡng trên val∖test"),
        nc.code(nc.FULL_STAGE1),
        nc.md("## 5. Teacher MedGemma — KD-R (che vùng) và KD-C (rationale), có ngân sách thời gian"),
        nc.code(nc.FULL_TEACHER),
        nc.md("## 6. Giai đoạn 2 — KD-R + faithfulness (khi có cache KD-R)"),
        nc.code(nc.FULL_STAGE2),
        nc.md("## 7. Giai đoạn 3 — verifier học (khi có KD-C) và generator S (khi có flan-T5)"),
        nc.code(nc.FULL_STAGE3),
        nc.md("## 8. Baseline MedGemma trên chính split này — pretrained và B4 (fine-tuned)"),
        nc.code(nc.FULL_BASELINES),
        nc.md("## 9. Suy luận Proposed-S và chấm: Bảng A–C, F (spec) + Bảng 4–6 (VinDr) + faithfulness"),
        nc.code(nc.FULL_EVAL),
        nc.md("## 10. Lưu output để resume / tải về"),
        nc.code(nc.FULL_SAVE),
    ])


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
    targets = {TRAIN_PATH: build_train(), PREDICT_PATH: build_predict(), FULL_PATH: build_full()}
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
