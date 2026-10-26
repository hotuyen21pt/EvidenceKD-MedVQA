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
