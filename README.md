# EvidenceKD-MedThink v4.1

Evidence-grounded medical VQA on chest X-rays: a decision branch produces the
answer, findings and scores; a generator only *explains* them, under a
citation constraint; a verifier checks each claim.

Implementation of the method spec `docs/method_spec_v4.1.md` (CV2 —
EvidenceKD-MedThink v4.1).

> **Status: stage 0 (detector) is runnable on Kaggle; nothing has been trained
> yet.** Every number in the spec's result tables is still blank, and nothing
> in this repository should be read as an experimental finding.

**Design for this repository on VinDr-CXR-VQA:**
[`docs/superpowers/specs/2026-10-26-vindr-cxr-vqa-on-evidencekd-design.md`](docs/superpowers/specs/2026-10-26-vindr-cxr-vqa-on-evidencekd-design.md)
— keeps the sibling repo's purpose (answer + `<loc>` box + reasoning, scored by
its Tables 4–6, one-image UI) and realises it with the EvidenceKD flow.

---

## Current state

| | |
|---|---|
| Research logic (schema, fusion, KD-R math, claim grammar, Answer Lock, splits) | **Implemented, tested** |
| Data on disk | **Yes** — `data/processed/` built from `../VinDR-CXR-VQA`, same split as that repo (3,735 train / 659 val / 300 test ⊂ val) |
| Bridge to VinDr's Tables 4–6 scorer | **Implemented, tested** (oracle replay scores P=1.0, 100 % IoU≥0.5 through `vindr_cxr_vqa.evaluate`) |
| Stage 0: Faster R-CNN detector, train/predict/eval, Kaggle notebooks + bundles | **Implemented, not yet run** (no GPU here) |
| Stages 1–3 (decision branch, teacher/KD, generator, verifier) | **Not written** |
| Any training run / any reported metric | **No** |

```
$ python -m pytest -q
180 passed in 2.14s
```

Run on MSYS2 CPython 3.14.3, no GPU, no torch installed. That is deliberate —
see [Why the core is torch-free](#why-the-core-is-torch-free).

---

## Luồng xử lý chi tiết của repo (tiếng Việt)

Mục này lần theo **một câu hỏi** của bản phát hành — ví dụ
`"Where is the Pleural effusion?"` trên ảnh `9a5094b2…` — qua mọi giai đoạn,
ghi rõ module nào làm gì, đọc file nào, ghi file nào, và lệnh chạy tương ứng.
Thứ tự dưới đây đúng là thứ tự các cell trong `notebooks/kaggle_full_pipeline.ipynb`.

```
VinDR-CXR-VQA (repo bên cạnh)                 EvidenceKD-MedVQA (repo này)
┌─────────────────────────────┐
│ data/processed/*.jsonl      │──(1) đọc────▶ data/vindr_processed      ──▶ VQASample, ImageRecord, ProcessedSplits
│ images.json (boxes_raw/fused)│
│ dataset/*.png 224px + CSV   │──(2) train──▶ models/detector (GĐ 0)     ──▶ outputs/detector/detector.pt, detections_{val,test}.json
│ vindr_cxr_vqa.evaluate      │◀─(9) chấm───  eval/vindr_bridge           ◀── chuỗi "{answer} <loc_…>. Reasoning: …"
│ outputs/medgemma-lora       │──(6) teacher▶ teacher/medgemma            ──▶ cache/kd_r, cache/kd_c
└─────────────────────────────┘
        (3) data/build_regions ──▶ derived/regions_*.json, anatomy.json, negatives.jsonl, labels_train.jsonl, background.json
        (4) inference/oracle_decisions + pipeline --generator template ──▶ self-test: P = 1.0, IoU≥.5 ≈ 100 %
        (5) models/decision_branch --stage 1 ──▶ outputs/decision/decision_stage1.pt, decision_{val,test}.jsonl, reports/thresholds.json
        (6) teacher/kd_r_targets, teacher/kd_c_targets ──▶ cache/kd_r/*.jsonl, cache/kd_c/*.jsonl
        (7) models/decision_branch --stage 2 ──▶ decision_stage2.pt
        (8) models/verifier, models/generator_s (hoặc generator_l) ──▶ outputs/verifier/verifier.pt, outputs/generator_s/final
        (9) inference/pipeline ──▶ outputs/pipeline_{split}_{gen}.jsonl + reports/preds_*.jsonl ──▶ eval/tables_spec, eval/vindr_tables, eval/faithfulness
       (10) serve + ui ──▶ hỏi từng ảnh, xem hộp, dữ kiện, mệnh đề và vùng được trích
```

### (1) Dữ liệu vào — `data/vindr_processed.py`

Không dựng lại gì. `scripts/build_data.py` gọi `vindr_cxr_vqa.build_dataset`
với cấu hình mặc định nên `data/processed/` giống **từng byte** với repo VinDR
(kiểm bằng sha256 trong test): 17.597 QA / 4.394 ảnh; train 3.735 ảnh, val 659,
test 300 ⊂ val. Mỗi `VQASample` mang câu hỏi, câu trả lời công bố, `reason`,
loại câu hỏi, `gt_finding`, và **hai hộp**: `bbox_orig` (công bố, 60 % sai lớp) và
`bbox_fixed_orig` (đã nối lại đúng lớp — hộp giám sát duy nhất ở đây). Mỗi
`ImageRecord` mang `boxes_raw` (từng bác sĩ) và `boxes_fused` (đồng thuận IoU 0,3,
có `votes`). `ProcessedSplits.held_out` = toàn bộ val; `tune` = val ∖ test (359 ảnh)
là nơi chọn mọi ngưỡng. Tọa độ nội bộ luôn là **pixel DICOM gốc**.

### (2) Giai đoạn 0 — detector, `models/detector.py`

`data/detection_dataset.build_index` đọc CSV VinDr qua `vindr_cxr_vqa.vindr`,
gộp ba lần đọc của bác sĩ ở IoU 0,3 (đúng tham số VinDR dùng để sửa hộp), giữ
14 lớp VQA, đổi hộp về không gian 224 px, **loại mọi ảnh val/test** rồi assert.
Faster R-CNN R50-FPN (torchvision, khởi tạo COCO từ `weights/`) upsample 224→512
bên trong model. Checkpoint mỗi epoch, `--resume auto`. `predict` ghi
`detections_{split}.json` với hộp đã quy về pixel gốc. `eval/stage0` chấm mAP,
Grounding@0,5 và — qua cầu nối — dòng **B7** trên Bảng 4–6 của VinDR.

### (3) Dữ liệu dẫn xuất — `data/build_regions.py`

Ghi vào `derived/`:

| File | Từ đâu | Dùng ở đâu |
|---|---|---|
| `anatomy.json` | torchxrayvision PSPNet (phổi/tim) hoặc fallback thorax = cả ảnh | bảng dữ kiện (bên, tầng, mức lan rộng) |
| `regions_train.json` | `boxes_fused` ≤ 8 vùng `R01..` theo `votes` rồi diện tích + 4 vùng giải phẫu `A_*` | huấn luyện GĐ 1–2, teacher KD-R che vùng, KD-C |
| `regions_{val,test}.json` | detector (`detections_*.json`, NMS theo lớp, ngưỡng điểm) | suy luận thật |
| `regions_{val,test}_annotation.json` | annotation | oracle, phân tích |
| `labels_train.jsonl` | `data/qa_labels`: loại câu hỏi, Yes/No, finding đa nhãn (lớp có hộp trên ảnh), bên/đếm **tính hình học** từ hộp của `gt_finding`, `target_regions` = vùng IoU ≥ 0,5 với `bbox_fixed_orig` | mọi head của decision branch, pointer, cổng KD-R |
| `negatives.jsonl` | `data/negatives`: câu hỏi mẫu "Is there X?" trả lời "No" trên ảnh train không có X và ảnh "No finding" của VinDr | head Yes/No, fact `absence` (chỉ train, chỉ QA+Full) |
| `background.json` | câu `reason` distinct theo finding | mệnh đề `background` đã duyệt |

`RegionSet` là **hợp đồng dữ liệu chung**: mã vùng được chốt ở đây một lần;
pointer chọn, mệnh đề trích, KD-R che — đều theo cùng mã.

### (4) Self-test oracle — `inference/oracle_decisions.py`

Giả lập đầu ra decision branch từ annotation (xác suất finding 1/0, relevance và
pointer đặt lên vùng đúng), chạy pipeline với generator mẫu câu, chấm bằng
`vindr_cxr_vqa.evaluate`. Phải cho precision 1,0 và ≈ 100 % ảnh IoU ≥ 0,5; nếu
không, cầu nối hỏng chứ không phải model. Đã chạy thật trên máy local: P = 1,0,
99,85 % IoU ≥ 0,5, Grounding@0,5 = 0,967.

### (5) Giai đoạn 1 — decision branch, `models/decision_branch.py`

Một câu hỏi vào model dưới dạng: ảnh 224 px, danh sách hộp vùng (finding + giải
phẫu) ở 224 px, mã lớp/loại vùng, token câu hỏi.

1. **Backbone đóng băng** của detector GĐ 0 → FPN; `MultiScaleRoIAlign` 7×7 trên hộp
   gốc **và** hộp mở rộng ×1,5 → ghép → token vùng d=256, cộng embedding loại
   (finding/giải phẫu/ảnh), lớp, tọa độ chuẩn hóa; thêm **token toàn ảnh**.
2. **Text encoder**: `simple` (từ vựng dựng từ câu hỏi train + 2 lớp Transformer,
   không cần tải gì) hoặc `hf` (CXR-BERT attach offline).
3. **Fusion** 2 lớp cross-attention hai chiều vùng ↔ token câu hỏi.
4. **Evidence encoder** 2 lớp Transformer trên [vùng; câu hỏi].
5. **Heads**: relevance ρ̂_j (vùng nào trả lời câu hỏi), loại câu hỏi, finding đa
   nhãn (từ token toàn ảnh), Yes/No, bên, đếm (từ [câu hỏi; vùng gộp theo ρ̂]),
   **pointer** π_c trên vùng finding + slot `NO_SUPPORT`.
6. **Loss GĐ 1**: BCE relevance + InfoNCE alignment + CE/BCE các head +
   −log Σ_{G} π (mẫu âm → slot null).

`predict` ghi `decision_{split}.jsonl` (xác suất từng head, relevance, pointer
theo mã vùng của `regions_{split}.json`). `pipeline --tune` chọn ngưỡng finding
trên `tune` theo F1 tập finding mỗi ảnh → `reports/thresholds.json`.

### (6) Teacher MedGemma — `teacher/`

`teacher/medgemma.py` nạp MedGemma-1.5-4B-it + **adapter LoRA của VinDR** qua
chính `vindr_cxr_vqa.infer` (cùng prompt, cùng chat template adapter được train).

* **KD-R** (`kd_r_targets.py`): với mỗi câu train, `a*` = câu trả lời công bố bỏ
  `<loc>`; tính `log p_T(a*|I,Q)` teacher-forced trên ảnh gốc và trên ảnh **che
  từng vùng** `R01..R08` (tô đen trên PNG 224 px); `r_i = max(0, sụt)`;
  `π^T = softmax([r, r_∅]/τ)`; sinh câu trả lời tự do để biết teacher đúng
  (`evaluate.mentioned` của VinDR). Ghi `cache/kd_r/shard-XXXX.jsonl`, resume theo
  shard. `--gate` tính trọng số `w_n = 1[đúng]·1[m_n ≥ δ]·c_n` (các cờ `--no-*`,
  `--random-gate` = các dòng Bảng E); `--sensitivity` là điều kiện vào của spec §8.
* **KD-C** (`kd_c_targets.py`): teacher nhận câu hỏi + đáp án mẫu + **bảng dữ kiện
  tính từ annotation**, sinh 3 rationale; `teacher/prompts.claims_from_text`
  parse ngữ pháp `<obs facts=F1> … [R01] </obs>` hoặc gán câu tự do vào fact khớp;
  `kd/kd_c.build_targets` + verifier luật giữ từng mệnh đề đạt →
  `cache/kd_c/shard-XXXX.jsonl`.

### (7) Giai đoạn 2 — KD-R và faithfulness

`decision_branch --stage 2 --init decision_stage1.pt` thêm `kd/kd_r.kd_r_loss`
(KL(π^T‖π^S) có trọng số cổng, cùng tập mã vùng vì train dùng `regions_train.json`)
và số hạng faithfulness: che **token** từng vùng trong evidence encoder, đo sụt
logit của finding được hỏi → π^cf → KL(π^cf‖π^S).

### (8) Giai đoạn 3 — verifier và generator

* `models/verifier_rules.py` (v1, luật): mệnh đề `supported` khi ngữ pháp sạch và
  mọi thuộc tính nêu trong câu (finding, bên, tầng, số, vắng mặt) khớp fact được
  trích; `contradicted` khi mâu thuẫn; `insufficient` khi nêu thứ không fact nào
  phủ. Dùng để **lọc** KD-C.
* `models/verifier.py` (v2, học): phân loại 3 trạng thái trên "mệnh đề ‖ dữ kiện
  của vùng trích", mẫu âm tổng hợp theo cấu trúc (đổi finding, đảo bên, thêm tầng
  không có, đổi số). Dùng để **chấm**, nên bộ lọc không tự chấm mình.
* `models/generator_s.py`: flan-T5 đọc `question | answer | facts | regions`,
  sinh mệnh đề; `claims/constrain.CitationLogitsProcessor` chặn `</obs>` khi chưa
  trích và chỉ cho mã vùng hợp lệ sau `[`; `enforce_citations` bảo đảm sau giải mã.
  Target = mệnh đề KD-C đã lọc (hoặc mẫu câu B9 cho dòng "S không KD").
* `models/generator_l.py`: MedGemma + **LoRA mới** (không dùng adapter teacher),
  cùng input + ảnh — mốc trần Proposed-L.

### (9) Suy luận và chấm — `inference/pipeline.py`

Với mỗi câu hỏi của val/test (vùng từ detector):

1. `decide()`: loại câu hỏi; finding được hỏi = tên trong câu hỏi, nếu câu hỏi
   không nêu tên thì = lớp của vùng có relevance cao nhất; có mặt ⇔ xác suất
   finding ≥ ngưỡng (và Yes/No với câu Is_there/Yes_No); bên, đếm từ head.
2. `facts/build.build_facts` → bảng dữ kiện từ **vùng detector + giải phẫu + đầu
   ra head** (presence, side, zone, count, extent, co-occurrence, absence; CTR
   không bao giờ tính vì không biết tư thế chụp).
3. `inference/templates.locked_answer_text` → chuỗi trả lời; `answer_lock.lock_answer`
   khóa lại — generator không thể sửa (`enforce_answer_lock` chỉ sinh cảnh báo).
4. `answer_lock.select_evidence` trên π_c → vùng bằng chứng hoặc `NO_SUPPORT`.
5. Chọn dữ kiện: về finding được hỏi + dữ kiện trích vùng pointer.
6. Generator (`template_simple` = B8, `template` = B9, `s`, `l`) → mệnh đề →
   `enforce_citations` → verifier → trạng thái từng mệnh đề.
7. JSON theo spec §2 (`answer, finding, scores, regions, facts, pointer, claims`)
   vào `outputs/pipeline_{split}_{gen}.jsonl`; **cầu nối** `eval/vindr_bridge.compose`
   dựng `"{answer} <loc_norm1000 của hộp trả lời>. Reasoning: {mệnh đề supported}"`
   vào `reports/preds_{split}_{gen}.jsonl` — hộp trả lời = vùng pointer cùng lớp,
   không có thì vùng relevance cao nhất cùng lớp, không có nữa thì **không phát
   `<loc>`** (không bao giờ bịa hộp).

Chấm: `eval/vindr_tables` gọi `vindr_cxr_vqa.evaluate` → Bảng 4–6 cạnh số của
paper; `eval/tables_spec` → Bảng A (Yes/No F1 — N/A nếu không có mẫu âm, finding
macro-F1, side acc, count MAE), B (Grounding@0,5, mIoU, answer&grounding),
C (supported-claim P, citation P/R, coverage, contradiction, question relevance,
độ gọn — **máy chấm**), F (ECE, latency); `eval/faithfulness` che vùng được trích
và so tỉ lệ đổi đáp án với 4 đối chứng (ngẫu nhiên, vùng điểm cao không trích,
vùng giải phẫu, vùng ghép từ ảnh khác).

### (10) Phục vụ một ảnh — `serve.py`

Cùng hợp đồng HTTP với `vindr_cxr_vqa.serve` (health, meta, samples, image,
predict) nên React UI của VinDR dùng được ngay; thêm trường `evidence`
(vùng, dữ kiện, pointer, mệnh đề kèm trạng thái). `--mock` chạy không cần torch;
`Runner` trong `inference/pipeline.py` chạy detector → decision → pipeline cho
một ảnh tải lên.

### Những gì tạo ra ở mỗi lần chạy Kaggle

`evidencekd-full.zip` = `outputs/` (checkpoint cuối, detections, decision, pipeline),
`reports/` (mọi bảng, thresholds, kd_r_gate, faithfulness), `derived/`, `cache/`
(KD-R, KD-C). Attach nó ở phiên sau: mọi giai đoạn có output sẵn sẽ tự bỏ qua,
shard KD dở dang chạy tiếp.

## Running the full pipeline on Kaggle

`notebooks/kaggle_full_pipeline.ipynb` chains every stage in one notebook and
resumes across sessions (attach the previous session's `evidencekd-full.zip`):

| Stage | What runs | Needs attached |
|---|---|---|
| 0 | Faster R-CNN detector → detections → B7 Tables 4–6 | code zip, images, COCO weights |
| derived | regions, anatomy (torchxrayvision if present), synthetic negatives, labels; **oracle self-test** through the bridge | — |
| 1 | decision branch (relevance, finding, Yes/No, side, count, pointer) → thresholds on val∖test | — |
| teacher | MedGemma + VinDr adapter: KD-R (region masking) + KD-C (rationales), sharded with a time budget | MedGemma (Kaggle Models) + `medgemma-lora.zip` |
| 2 | decision branch + KD-R + faithfulness term | KD-R cache |
| 3 | learned verifier (KD-C cache), generator S (flan-T5) | flan-t5-base dir |
| eval | Proposed-S inference → spec Tables A–C/F, VinDr Tables 4–6, faithfulness interventions | — |

Stages whose inputs are missing are skipped with a message; the template
generator (B9) and the rule verifier are always available. The whole
inference chain is exercised locally without torch by replaying the answer
key (`evidencekd.inference.oracle_decisions` → `pipeline --generator template`
→ `eval.vindr_tables`): VQA precision 1.0, 99.85 % of images at IoU ≥ 0.5,
Grounding@0.5 0.967 (spec Table B) on val.

## Running stage 0 on Kaggle

Everything below mirrors the sibling repo's workflow: bundles are built by a
script, notebooks are generated, inputs are located wherever Kaggle nests them,
internet-off is detected, and training checkpoints every epoch so it can be
resumed across sessions.

```bash
# once, locally (no torch needed for any of this)
python scripts/build_data.py                              # data/processed + reports/split_manifest.json
python scripts/fetch_weights.py                           # weights/fasterrcnn_resnet50_fpn_coco-*.pth (160 MB)
python scripts/make_kaggle_bundle.py --part code          # outputs/evidencekd-kaggle-code.zip    (~45 MB)
python scripts/make_kaggle_bundle.py --part weights       # outputs/evidencekd-kaggle-weights.zip (160 MB)
python scripts/make_kaggle_bundle.py --part images        # outputs/evidencekd-kaggle-images.zip  (4,394 PNG, ~360 MB)
python scripts/make_kaggle_bundle.py --part images --all-images   # all 15,000 PNG (~1.3 GB) — better detector
```

1. **kaggle.com/datasets → New Dataset** → upload `evidencekd-kaggle-code.zip`.
   Same for the weights zip, and for an images zip unless the `vpa-repo`
   dataset (same layout) is already on Kaggle.
2. **Code → New Notebook → File → Import Notebook** →
   `notebooks/kaggle_stage0_detector.ipynb`. Import before attaching anything.
3. *Add Input* → the code dataset, the images dataset, the weights dataset.
4. *Settings* → **Accelerator: GPU T4 x2** (or P100). Run all.

The notebook: locates inputs → checks the attached code is current → writes
`configs/kaggle.yaml` (paths, batch size by VRAM, `image_source: all` when the
15,000 PNGs are attached) → preflight → **bridge self-test** (answer key
through the EvidenceKD → VinDr scorer path must be perfect) → trains →
predicts val/test → scores mAP, Grounding@0.5 and VinDr's Tables 4–6 for the
detector-only baseline B7 next to the paper's numbers → overlays → zips
`detector.pt` + newest checkpoint + reports as `evidencekd-detector.zip`.

To continue a run in a later session, attach that zip as an input: the train
cell copies its checkpoints back and `--resume auto` picks the newest.
`notebooks/kaggle_stage0_predict.ipynb` runs a finished detector on val/test
and on any image of your own.

```bash
# the same steps, locally, on a machine with torch + a GPU
python -m evidencekd.preflight --config configs/detector.yaml
python -m evidencekd.eval.stage0 --config configs/detector.yaml --oracle --split val
python -m evidencekd.models.detector train --config configs/detector.yaml --resume auto
python -m evidencekd.models.detector predict --config configs/detector.yaml --checkpoint outputs/detector/detector.pt --split test
python -m evidencekd.eval.stage0 --config configs/detector.yaml --detections outputs/detections_test.json --split test
```

`vindr_cxr_vqa` (the sibling package: CSV readers, `<loc>` parsing, the
Tables 4–6 scorer) is imported from `../VinDR-CXR-VQA` — `pyproject.toml`
adds it to the pytest path, and the code bundle ships a copy. It is not
vendored into `src/`.

---

## The idea

A medical VLM is distilled into a student that **does not call a VLM at
inference**, yet still answers, localises, and explains in claims that cite
verified evidence regions.

- **C1 — Selective distillation.** KD-R (counterfactual region relevance) and
  KD-C (claim-level rationale filtering).
- **C2 — Answer-locked explanation**, mandatory citation, faithfulness
  measured by intervening on regions.

Two systems: **Proposed-S** (small decoder, no VLM at inference — the main
system) and **Proposed-L** (MedGemma + LoRA on the same decision branch — the
ceiling).

---

## Data

[VinDr-CXR-VQA](https://huggingface.co/datasets/Dangindev/VinDR-CXR-VQA)
([paper](https://arxiv.org/abs/2511.00504)) — **verified 2026-10-19**:

| | |
|---|---|
| Images | 4,394 |
| QA pairs | 17,597 (3–5 per image) |
| Fields | `question, answer, reason, type, difficulty, gt_finding, gt_location` |
| Question types | `Where, What, Is_there, How_many, Which, Yes_No` |
| Pathology classes | 14 |
| Balance | 41.7 % positive / 58.3 % negative |
| License | CC BY 4.0 |
| Splits shipped | **one (`train`) — val/test are ours to make** |

Full annotation (boxes, anatomy) comes from the underlying
[VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/), which is
**PhysioNet credentialed** (CITI + signed DUA, redistribution prohibited).
No image data is committed to this repository, and none may be uploaded
without checking that license first.

Boxes are embedded in text as `<loc_xmin_ymin_xmax_ymax>` tokens, so box
extraction is string parsing — see `data/vqa_schema.py`.

---

## Decisions that must not be reverted

Each of these is a correction someone is likely to "fix" back.

| Decision | Why |
|---|---|
| **The KD-R softmax has a null slot `r_∅`** | A plain `r_i / Σr_j` normalisation divides by zero when every region is irrelevant — which is exactly the diagnostic case the spec's stopping rule cares about. The null slot makes that case well-defined *and* gives the pointer somewhere to put `NO_SUPPORT` mass. Spec §4.4. |
| **Negative relevance is clipped to zero, not kept** | A region whose masking makes the teacher *more* confident is not evidence for the answer. Signed mass would make the softmax reward regions arguing against it. |
| **Annotation is used only at the KD-R gate** | `m_n` and teacher correctness decide *which samples* contribute. The target stays the teacher's full distribution, including mass on unannotated regions — spec §4.4: that off-target mass *is* the knowledge being transferred. Clamping the target to the annotated box would delete the point of KD-R. |
| **`INSUFFICIENT` is never promoted to `SUPPORTED`** | "We could not tell" is not "yes". Three-state verification collapses to useless if this slips. |
| **Absence requires *both* no box and a low whole-image score** | Spec §4.2. A missing box means nobody annotated it, not that the finding is absent. Asserting absence from box-emptiness is the exact inference the method forbids. |
| **CTR is computed on PA films only; unknown view → no CTR** | AP magnifies the heart, so an AP CTR systematically over-calls cardiomegaly. An unlabelled film is *not* assumed PA. |
| **Image-left is the patient's right** | On a frontal CXR the patient's right lung is displayed on the viewer's left. `facts.geometry.side_of` performs the inversion; `data.answer_norm.parse_side` reads sides out of *text*, where no inversion applies. The two are kept apart on purpose — merging them silently mirrors every side label. |
| **`LockedAnswer` is a guarded class, not `frozen=True`** | `frozen=True` raises a bare `FrozenInstanceError` that says nothing about why. The custom `__setattr__` names the rule being enforced. |
| **A generator disagreeing with the locked answer produces a *warning*, never a correction** | Spec §2: the decision branch owns the answer. Abstention, if wanted, is a separate branch with its own threshold — not the generator quietly concluding something else. |
| **Splitting is by image, never by question** | Each image carries 3–5 questions. A question-level split puts questions about the same radiograph in train and test. |
| **The split is VinDr-CXR-VQA's, byte-identical: 3,735 train / 659 val, test = a 300-image subset of val** — not the 10/10 seed-1337 default of `data/splits.py` | Tables 4–6 are then scored on exactly the images the paper scored. Because test ⊂ val, `ProcessedSplits.held_out` is all of val, and thresholds are chosen on `tune` = val ∖ test (359 images) so test is still never used to pick anything (spec §8). Manifest hash in `reports/split_manifest.json`. |
| **Region boxes are VinDr's `boxes_fused` at IoU 0.3, not `data/boxes.fuse_boxes` at 0.4** | `gt_location_fixed` — the supervision box — was chosen from exactly that fused set; re-fusing at another threshold would make the target box and the candidate regions disagree. `n_raters = votes` is kept. |
| **The detector never flips images horizontally** | Image-left is the patient's right; a flip silently mirrors every laterality fact. |
| **No box is fabricated when the detector is silent** | `vindr_bridge.compose` emits no `<loc>` for a question with no box above threshold. VinDr's scorer reports `with_loc_pct`; a fabricated centred box would hide a localisation failure as a bad one. |
| **`parse_count` trusts number words only for `How_many`** | "three times larger" is not a count. Digits are trusted anywhere; words are not. |
| **The core is torch-free** | See below. |

### Why the core is torch-free

The development machine has **no CUDA device** (Intel UHD 770) and its only
interpreter is an **MSYS2 MinGW build of Python 3.14**, for which PyTorch
publishes no wheels. Keeping schema parsing, box fusion, the KD-R arithmetic,
the claim grammar, Answer Lock and splitting in pure Python means the research
logic is **actually tested** rather than written and hoped for. Torch is in
the `train` extra and imported lazily by the two functions that need it
(`kd_r_loss`, `kd_c_loss`).

Training runs on Kaggle. Do not move a torch import to module level.

---

## Layout

```
src/evidencekd/
  labels.py                 VinDr class names; the 14 VQA classes + synonyms (VQA_CLASSES)
  data/
    vqa_schema.py           VinDr-CXR-VQA records, <loc_...> parsing
    vindr_processed.py      reader for ../VinDR-CXR-VQA's built data (JSONL + images.json)
    detection_dataset.py    detector samples from the VinDr CSVs (torch half lazy)
    answer_norm.py          THE answer normalisation rule (Yes/No, finding, side, count)
    boxes.py                IoU, multi-radiologist box fusion, Recall@K
    splits.py               split-by-image, leakage guards, split manifest hash
  facts/geometry.py         the geometric fact table (spec §4.2)
  claims/parser.py          claim grammar, citation whitelist, constrained decoding
  kd/
    kd_r.py                 counterfactual relevance, null-slot softmax, gated KL
    kd_c.py                 claim-level filtering (none / sample / claim), KD-C loss
  inference/answer_lock.py  Answer Lock, Evidence Pointer, NO_SUPPORT
  models/detector.py        stage 0: Faster R-CNN R50-FPN train / predict / resume (torch)
  eval/
    detection.py            class-aware mAP (pure Python)
    vindr_bridge.py         EvidenceKD output -> VinDr prediction string -> Tables 4-6
    stage0.py               CLI: mAP + Grounding@IoU + VinDr tables for a detector
  preflight.py              packages, GPU, data, weights, scorer — before any GPU time
  utils/config.py           YAML config with a PyYAML-free fallback
  teacher/                  not yet implemented
scripts/
  build_data.py             vindr build_dataset --disjoint-test -> data/processed
  fetch_weights.py          COCO Faster R-CNN weights for offline Kaggle
  make_kaggle_bundle.py     code / images / weights / all / repo zips
  make_notebooks.py         generates notebooks/*.ipynb from notebook_cells.py
notebooks/
  kaggle_stage0_detector.ipynb   train + score on Kaggle
  kaggle_stage0_predict.ipynb    run a trained detector, ask on one image
configs/
  detector.yaml             stage 0 hyper-parameters (all untuned)
tests/                      180 tests, no GPU required
docs/
  method_spec_v4.1.md       the method specification this implements
  superpowers/specs/        the design of this repo on VinDr-CXR-VQA (2026-10-26)
  implementation_audit.md   Phase 0 environment/data audit
  experiment_plan.md        milestones and entry conditions
```

`answer_norm.py` is the **single source of truth** for answer normalisation.
Spec §5 requires every baseline's text output to be parsed by the same rules;
nothing else may re-implement any of it.

---

## Running the tests

```bash
python -m pytest -q
```

No GPU, no torch, no dataset required. `--doctest-modules` is on, so the
examples in the docstrings are tests too.

---

## What is NOT done

Being explicit, because the gap is still large:

- **Only stage 0 has model code** (the Faster R-CNN detector). Text encoder,
  fusion, region/evidence encoders, answer heads, pointer, generator and
  verifier are unwritten — see the design doc §10 for the file plan.
- **No teacher integration.** The design names the sibling repo's MedGemma
  LoRA adapter as teacher and B4, but ``teacher/`` is empty and its
  answer-logprob access has not been exercised. KD-R is undefined without it.
- **No training run has happened.** The stage-0 notebook is generated and
  tested for syntax and freshness, not executed — this machine has no GPU.
- **No evaluation number exists.** The only scores on disk are the oracle
  bridge self-test (which must be perfect by construction).
- **No clinician annotation.** The spec calls for 800–1000 claims double-read
  with a reported kappa. Not started, not budgeted.

Known risks worth naming now:

- **The detector sees 224 px images** (the local VinDr copy is pre-resized);
  stage 0 upsamples to 512 px inside the model. Expect lower mAP than
  full-resolution VinDr-CXR results, and treat that as a measured limit.
- **Question-conditioned Deformable DETR** (the spec's detector) needs a
  custom CUDA kernel; building it offline on Kaggle is fragile. The design
  substitutes torchvision Faster R-CNN with a relevance head after fusion.
- **KD-R cost.** One teacher pass per region per question. This must be
  sharded, cached and resumable before any full run, and budgeted first.
- **Self-evaluation bias.** If one verifier both filters KD-C targets and
  scores the results, the claim metrics are not independent. Spec §4.3 calls
  for an independent check set; that is not optional.

---

## License

Code: see `LICENSE`. The datasets are **not** covered by it — VinDr-CXR-VQA is
CC BY 4.0, and the underlying VinDr-CXR is PhysioNet credentialed with
redistribution prohibited.
