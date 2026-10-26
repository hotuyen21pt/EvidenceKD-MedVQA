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

### Ví dụ bằng số: một câu hỏi đi hết pipeline

Bản ghi thật `06b2ff94bd0270f543deb7074c2a006a_1` của tập val. Mọi con số về
dữ liệu, hộp, nhãn, bảng dữ kiện và quy đổi tọa độ là **tính thật** bằng mã của
repo; các con số đầu ra của mạng nơ-ron (logit, xác suất, logprob của teacher)
là **ví dụ minh họa** đúng kích thước và đúng công thức, vì chưa có lần train nào.

**Đầu vào.** Ảnh gốc 2500×3012 px, PNG trên đĩa 224×224 → `fx = 224/2500 = 0,0896`,
`fy = 224/3012 = 0,0744`. Câu hỏi `Where is the Pulmonary fibrosis?` (`type = Where`,
`gt_finding = Pulmonary fibrosis`). Câu trả lời công bố:
`"The fibrotic changes … within the specified region <loc_2052_1893_2311_1950>."`
Hộp công bố = hộp sửa = `(2052, 1893, 2311, 1950)` (`repair_method = published`).
Ảnh này có 5 câu hỏi (How_many, Where, Is_there, Which, Yes_No), cùng một hộp.

**Hộp của bác sĩ trên ảnh (`boxes_raw`)** — ba người đọc R8, R9, R10:

| Lớp | Bác sĩ | Hộp (pixel gốc) |
|---|---|---|
| Pulmonary fibrosis | R10 | (2045, 1892, 2276, 1964) |
| Pulmonary fibrosis | R8 | (2052, 1893, 2312, 1950) ← chính là hộp công bố |
| Pulmonary fibrosis | R9 | (2062, 1874, 2289, 1970) |
| Pulmonary fibrosis | R8 | (472, 826, 598, 1088) |
| ILD | R10 | (316, 768, 733, 2285) |
| Aortic enlargement | R9 | (1293, 887, 1577, 1190) |

**Gộp (IoU ≥ 0,3, trung bình tọa độ) → `boxes_fused`**, sắp theo `votes` rồi diện tích,
thành vùng ứng viên `R01..R04` + giải phẫu (fallback: `A_thorax` = cả ảnh):

| Vùng | Lớp | votes | Hộp pixel gốc | Hộp 224 px (`× fx, × fy`) |
|---|---|---|---|---|
| R01 | Pulmonary fibrosis | 3 | (2053,0, 1886,5, 2292,2, 1961,4) | (184,0, 140,3, 205,4, 145,9) |
| R02 | ILD | 1 | (315,9, 768,1, 733,2, 2284,8) | (28,3, 57,1, 65,7, 169,9) |
| R03 | Aortic enlargement | 1 | (1292,6, 887,1, 1576,6, 1190,2) | (115,8, 66,0, 141,3, 88,5) |
| R04 | Pulmonary fibrosis | 1 | (472,0, 826,5, 597,6, 1088,1) | (42,3, 61,5, 53,5, 80,9) |
| A_thorax | — | — | (0, 0, 2500, 3012) | (0, 0, 224, 224) |

**Vùng đích của câu hỏi** = vùng cùng lớp có IoU ≥ 0,5 với hộp sửa:

```
IoU(R01, gt): giao x [2053,0 ; 2292,2] = 239,2   giao y [1893 ; 1950] = 57,0   → giao = 13 634
              |gt| = 259 × 57 = 14 763    |R01| = 239,2 × 74,9 = 17 916
              IoU = 13 634 / (14 763 + 17 916 − 13 634) = 13 634 / 19 045 = 0,716  ✓  (R04: 0,0)
→ target_regions = ("R01",)
```

**Nhãn huấn luyện (`labels_train.jsonl`)**, tính hình học từ hộp của lớp được hỏi:

```
question_type = 0 (Where)      yes_no = 1
findings      = [0,1,0,0,1,0,0,0,0,0,0,1,0,0]   # lớp 1 PF, 4 Aortic enlargement, 11 ILD có hộp
side          = 2 (bilateral)  # tâm R01 x = 2172 > đường giữa 1250 → "left" (ảnh-phải = bệnh nhân-trái);
                               # tâm R04 x = 535 < 1250 → "right"; hai bên → bilateral
count         = 2              # 2 hộp PF sau NMS
target_regions = ["R01"]       asked_class = 1
```

**Bảng dữ kiện (`facts/build`)** cho ảnh này, nếu nhánh quyết định báo có PF, AE, ILD
và điểm toàn ảnh của Pneumothorax = 0,02 < ngưỡng vắng mặt 0,3:

```
F1: presence Aortic enlargement [R03]      F2: side left [R03, A_thorax]        F3: count 1 [R03]
F4: presence ILD [R02]                     F5: side right [R02, A_thorax]       F6: count 1 [R02]
F7: presence Pulmonary fibrosis [R01, R04] F8: side bilateral [R01, R04, A_thorax]  F9: count 2 [R01, R04]
F10: co_occurrence Aortic enlargement with ILD [R03, R02]
F11: co_occurrence Aortic enlargement with Pulmonary fibrosis [R03, R01, R04]
F12: co_occurrence ILD with Pulmonary fibrosis [R02, R01, R04]
F13: absence Pneumothorax [A_thorax]
```
(Không có `zone`/`extent` vì giải phẫu fallback không có hộp phổi; không có `ctr` vì không biết tư thế chụp.)

**Decision branch — kích thước tensor** (batch 1):

```
ảnh          [3, 224, 224] ──transform──▶ [3, 512, 512]   (tỉ lệ 512/224 = 2,286)
FPN          '0' [256,128,128]  '1' [256,64,64]  '2' [256,32,32]  '3' [256,16,16]
hộp R01 ở 512 px = (420,5, 320,7, 469,5, 333,5); hộp mở rộng ×1,5 = (408,3, 317,5, 481,8, 336,7)
ROIAlign     mỗi vùng: 2 × [256, 7, 7] = 2 × 12 544 ──Linear──▶ 2 × 256 ──merge──▶ 256
token vùng   = roi + emb_loại(finding) + emb_lớp(PF) + Linear(tọa độ chuẩn hóa) → [256]
tokens       [1 + 5, 256]      (IMG, R01, R02, R03, R04, A_thorax)
câu hỏi      "where is the pulmonary fibrosis" → ids [CLS, where, is, the, pulmonary, fibrosis] → [6, 256]
fusion ×2    vùng ↔ câu hỏi (cross-attention 8 head), evidence encoder ×2 trên [6 + 6, 256]
q_pool       trung bình 6 token câu hỏi → [256];   regf = 4 vùng finding → [4, 256]
```

**Heads — ví dụ đầu ra và loss** (số minh họa):

```
relevance logits  [ 2,1  −1,3  −0,4   0,2 ]  → σ = [0,89  0,21  0,40  0,55]
   L_rel = BCE với nhãn [1,0,0,0] = (0,117 + 0,236 + 0,511 + 0,799) / 4 = 0,416

finding probs     PF 0,93   AE 0,72   ILD 0,61   còn lại < 0,10   (ngưỡng tune = 0,5 → có mặt: PF, AE, ILD)
yes/no            [0,06  0,94]      side [left 0,12  right 0,08  bilateral 0,80]      count [0,05 0,20 0,65 0,07 0,03]

pointer logits    [ 1,8  −0,9  −0,2   0,3 | null −1,0 ]
   exp            [6,05  0,41  0,82  1,35 | 0,37]   tổng 8,99
   π^S            [0,673 0,045 0,091 0,150 | 0,041]
   L_pointer = −log Σ_{G={R01}} π = −log 0,673 = 0,396

alignment cos(q, z_i) = [0,62 0,10 0,21 0,35], τ = 0,07 → /τ = [8,86 1,43 3,00 5,00]
   L_align = −log( e^8,86 / (e^8,86 + e^1,43 + e^3,00 + e^5,00) ) = −log(7040 / 7213) = 0,024

Evidence Pointer lúc suy luận: top_k = 3, min_p = 0,05, phải > null 0,041
   → region_ids = [R01 (0,673), R04 (0,150), R03 (0,091)], status SUPPORTED
```

**Teacher KD-R** (`a*` = câu trả lời công bố bỏ `<loc>`; logprob trung bình/token; số minh họa):

```
log p_T(a* | ảnh gốc)            = −0,42
log p_T(a* | che R01)            = −0,95   → r_1 = max(0, −0,42 − (−0,95)) = 0,53
log p_T(a* | che R02)            = −0,44   → r_2 = 0,02
log p_T(a* | che R03)            = −0,40   → r_3 = max(0, −0,02) = 0      (che làm teacher TỰ TIN HƠN → cắt về 0)
log p_T(a* | che R04)            = −0,50   → r_4 = 0,08
r_∅ = 0

τ_T = 1,0 (configs/kd_r.yaml):  π^T = softmax([0,53 0,02 0 0,08 | 0]) = [0,293 0,176 0,172 0,187 | 0,172]
   khối lượng trên vùng đích R01 = 0,293 < δ = 0,5  → cổng KHÔNG qua (gate_reason = mass_below_delta) → w = 0
τ_T = 0,2:                      π^T = softmax([2,65 0,10 0 0,40 | 0]) = [0,755 0,059 0,053 0,080 | 0,053]
   m = 0,755 ≥ 0,5; teacher đúng (câu sinh ra có "pulmonary fibrosis"); c = exp(−0,42) = 0,657 → w = 0,657
   KL(π^T ‖ π^S) = Σ p log(p/q)
     = 0,755·ln(0,755/0,673) + 0,059·ln(0,059/0,045) + 0,053·ln(0,053/0,091) + 0,080·ln(0,080/0,150) + 0,053·ln(0,053/0,041)
     = 0,087 + 0,016 − 0,029 − 0,050 + 0,014 = 0,037
   L_KD-R (một mẫu) = w·KL / (w + ε) = 0,037
```
Hai dòng τ_T cho thấy vì sao README đòi **báo cáo `r_null` và τ_T cùng mọi kết quả KD-R**:
chúng quyết định mẫu nào qua cổng.

**Faithfulness giai đoạn 2** (che *token* từng vùng trong evidence encoder, logit PF):

```
logit PF đầy đủ = 2,6  (log σ = −0,072)
bỏ R01 → 0,9 (log σ = −0,341) → sụt 0,270      bỏ R02 → 2,5 → 0,006
bỏ R03 → 2,55 → 0,003                           bỏ R04 → 2,2 (log σ = −0,105) → 0,034
π^cf = softmax([0,270 0,006 0,003 0,034 | 0]) = [0,245 0,188 0,187 0,193 | 0,187]
L_faith = KL(π^cf ‖ π^S) ≈ 0,49  → kéo pointer về phía vùng mà chính mạng dựa vào khi quyết định
```

**Suy luận và chuỗi cầu nối** cho câu hỏi này (generator mẫu câu B9, verifier luật):

```
decide():   type Where; finding được hỏi = "Pulmonary fibrosis" (có trong câu hỏi); có mặt (0,93 ≥ 0,5)
            side = bilateral (0,80); findings_present = [PF, AE, ILD]
locked answer:  "Pulmonary fibrosis is seen bilaterally."
dữ kiện chọn:   về PF → F7, F8, F9, F11, F12; trích vùng pointer R01/R04/R03 → thêm F1, F2, F3
mệnh đề:  <obs facts=F7> Pulmonary fibrosis is present. [R01] </obs>        → supported
          <obs facts=F8> The finding is bilateral. [R01] </obs>             → supported
          <obs facts=F9> There are 2 lesions. [R01] </obs>                  → supported
          <obs facts=F11> Aortic enlargement co-occurs with Pulmonary fibrosis. [R03] </obs> → supported
          Pulmonary fibrosis involves irreversible scarring of the lung tissue…  (background, từ `reason`)
hộp trả lời = vùng pointer đầu tiên cùng lớp = R01 = (2053,0, 1886,5, 2292,2, 1961,4)
norm1000:   x·999/2500 = ×0,3996 ; y·999/3012 = ×0,3317
            2053,0→820   1886,5→626   2292,2→916   1961,4→651   → <loc_820_626_916_651>
chuỗi:      "Pulmonary fibrosis is seen bilaterally <loc_820_626_916_651>. Reasoning: Pulmonary fibrosis is present.
             The finding is bilateral. There are 2 lesions. Aortic enlargement co-occurs with Pulmonary fibrosis."
```
(Hộp sửa của đề bài quy đổi là `<loc_820_628_923_647>`; hai chuỗi khác nhau vài đơn vị vì
R01 là hộp **đồng thuận** còn `gt_location` là hộp của một bác sĩ.)

**Scorer VinDr chấm chuỗi đó** (ảnh này, cùng 4 câu hỏi còn lại trả lời tương tự):

```
Bảng 4  (theo ảnh, 14 lớp): các câu trả lời nêu PF, AE, ILD; hộp thô của ảnh có đúng 3 lớp đó
        → TP 3, FP 0, FN 0, TN 11   (precision 1,0 · recall 1,0 cho ảnh này)
Bảng 5  (hộp dự đoán vs hộp thô, IoU ≥ 0,3, match "any"):
        IoU(R01, hộp R8) = 13 665 / (17 916 + 14 837 − 13 665) = 13 665 / 19 088 = 0,716 ≥ 0,3 → TP, mIoU_TP cộng 0,716
Bảng 6  (theo ảnh): IoU tốt nhất 0,716 ≥ 0,5 → ảnh được tính "định vị tốt"
Bảng B  (spec, theo câu hỏi): IoU(hộp trả lời, gt sửa) = 0,716 ≥ 0,5 → Grounding@.5 = 1; answer đúng → Answer&grounding = 1
Bảng C  (spec): 4 observation, 4 supported → supported-claim P = 1,0; 4 mệnh đề nêu "Pulmonary fibrosis"? 3/4 → relevance 0,75
```

**Can thiệp (Bảng F)**: tô đen R01 trên PNG 224 px (hình chữ nhật (184, 140)–(205, 146)) và
chạy lại nhánh quyết định; nếu xác suất PF rơi từ 0,93 xuống dưới 0,5 thì "đổi đáp án" = 1
cho đối chứng `cited`; làm tương tự với R03 (`random`/`uncited_top`), `A_thorax`, và ô R01
ghép từ ảnh khác (`other_image`). Hệ thống chỉ được gọi là faithful khi tỉ lệ đổi đáp án
của `cited` vượt cả bốn đối chứng.

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
