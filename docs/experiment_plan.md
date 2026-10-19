# Experiment Plan — EvidenceKD-MedThink v4.1

**Date:** 2026-10-19
**Companion to:** `implementation_audit.md` (read that first — it establishes
what is actually available).

**Scope note.** Decisions **D1** (where Q/A come from) and **D6** (which
Teacher) are unresolved. They determine the task definition, the answer space,
the losses and every metric. This plan is therefore written **concretely only
for the milestones that do not depend on them**, and marks the rest as
**BLOCKED** with its entry condition stated. Writing a detailed plan for
KD-R/KD-C now would be guesswork presented as a plan.

---

## 0. Milestone map

| ID | Milestone | Depends on | Status |
|---|---|---|---|
| **M0** | Phase 0 audit | — | ✅ **done** (`implementation_audit.md`) |
| **M1** | Local toolchain usable | D3 | ⬜ ready to start |
| **M2** | Data on disk + schema verified against real files | D2, D5 | ⬜ blocked on data access |
| **M3** | Data audit + splits + leakage check | M2, D4 | ⬜ |
| **M4** | Q/A construction + frozen task definition | **D1**, M3 | 🚫 **BLOCKED** |
| **M5** | Baseline VQA model + smoke test (CPU, stand-in config) | M4 | 🚫 blocked by M4 |
| **M6** | Baseline training on Kaggle | M5 | 🚫 |
| **M7** | Teacher interface + cache + correctness filter | **D6**, M4 | 🚫 **BLOCKED** |
| **M8** | KD-R target generation | M7 | 🚫 |
| **M9** | KD-C claim extraction + verifier + filtering | M7 | 🚫 |
| **M10** | Integrated student training | M8, M9 | 🚫 |
| **M11** | Inference pipeline (Evidence Pointer, Answer Lock) | M5 | 🚫 |
| **M12** | Evaluation + ablation matrix E0–E3 | M10, M11 | 🚫 |

**The critical path runs through D1.** Until it is answered, M4 onward cannot
begin, and M5–M12 inherit the block.

---

## M1 — Local toolchain usable

**Goal:** an interpreter on this machine that can run the data pipeline and
CPU smoke tests. Training stays on Kaggle regardless.

**Why it is needed:** the only local interpreter is MSYS2 Python 3.14 with no
torch and no numpy (`implementation_audit.md` §2.1). Without this, *nothing*
can be tested locally — not even a dataset reader.

**Tasks**

1. Install CPython **3.11** (not 3.14 — torch and pydicom both publish wheels
   for 3.11; 3.14 has none) into a venv at `new/.venv`.
2. Install a **CPU-only** dependency set: `torch` (CPU wheel), `torchvision`,
   `numpy`, `pandas`, `pydicom`, `pillow`, `pyyaml`, `pytest`.
3. Record exact resolved versions in `reports/env_local.json`.
4. Keep `.venv-build` (MSYS2) untouched — the sibling projects use it.

**Acceptance criteria**

- `python -c "import torch, torchvision, pydicom, pandas; print(torch.__version__)"` succeeds.
- `torch.cuda.is_available()` is `False` **and this is recorded, not worked around.**
- `pytest` collects and runs an empty suite without error.
- `reports/env_local.json` exists with pinned versions.

**Risk:** low. Independent of every open decision — safe to do now.

---

## M2 — Data on disk, schema verified against the real files

**Goal:** the dataset present locally, and its *actual* schema confirmed —
not the schema this audit read from PhysioNet.

**Entry condition:** **D2** (which copy, what resolution) and **D5**
(credentialing / what may be stored) answered. Requires a human step: no
Kaggle CLI and no credentials exist on this machine.

**Tasks**

1. Obtain the data (user action: Kaggle credentials, or a manual download, or
   PhysioNet directly).
2. Record for the real files: byte size, file count, image format, pixel
   dimensions, bit depth.
3. **Verify the CSV headers byte-for-byte against `implementation_audit.md`
   §3.1.** Note the audit found `rad_id` in `annotations_train.csv` but
   `rad_ID` in `image_labels_train.csv` — differing case. Confirm or correct.
4. Record a manifest hash of the dataset for reproducibility.

**Acceptance criteria**

- `reports/data_manifest.json` exists: file list, sizes, hashes, image
  dimensions, exact CSV headers.
- Any divergence from the audit's §3.1 schema is written back into the audit
  as a correction.
- **No original file is modified or deleted.** Derived files go to a separate
  `derived/` tree.

**Risk:** medium. Size may exceed free disk (audit §3.3); the mirror's format
is unverified.

---

## M3 — Data audit, splits, leakage check

**Goal:** know exactly what the data supports, and have a frozen split that is
defensible.

**Entry condition:** M2 complete; **D4** (box fusion rule) answered.

**Tasks**

1. `scripts/validate_data.py` → `reports/data_audit.json`, reporting per split:
   sample counts; images that decode successfully; missing/!valid annotations;
   **boxes outside image bounds**; duplicate images (pixel-hash); per-class box
   counts for all 22 local findings; per-class counts for the 6 global
   diagnoses; distribution of boxes per image.
2. **Inter-rater analysis** (this dataset's defining property): per finding,
   how often do 3/3, 2/3, 1/3 radiologists agree? Pairwise IoU distribution
   between radiologists on the same finding. This directly informs the
   `SUPPORTED / INSUFFICIENT` design in audit §3.5.
3. Apply the **D4** fusion rule; emit fused boxes to `derived/boxes_fused.parquet`
   with the rule and its parameters recorded alongside.
4. **Patient-level leakage check.** Probe DICOM headers for a patient
   identifier. Two outcomes, both acceptable, neither may be skipped:
   - identifier present → split by patient;
   - absent → **record that patient-level leakage could not be excluded**, and
     say so in the paper's limitations.
5. Freeze train/val/test with a recorded seed and a manifest hash. The official
   VinDr-CXR test split (3,000 images, 5-radiologist consensus) should be the
   test set; val is carved from train.

**Acceptance criteria**

- `reports/data_audit.json` + a readable Markdown summary exist.
- No image appears in two splits (verified by pixel-hash, not filename).
- The split manifest is hashed and committed.
- The leakage question has a recorded **answer**, even if that answer is
  "could not be determined".
- Filters that drop samples are **config-driven and counted** — no silent drops.

**Risk:** medium. The inter-rater analysis may show disagreement high enough
that some of the 22 classes are too noisy to use; that is a finding, not a
failure, and it should change the class list rather than be hidden.

---

## M4 — Q/A construction and frozen task definition 🚫 BLOCKED on D1

**Entry condition: D1 answered.**

This milestone defines what the model is actually asked to do. It cannot be
guessed at. Once D1 is answered, this milestone must produce, before any
model code:

- the **question set** and how each question is generated or sourced;
- the **answer space** — closed-set classification or free generation
  (this selects the loss in proposal §4.3, and the two must not be mixed);
- the **answer normalisation rule**, fixed once and used identically by every
  model and every metric (master prompt §10.1);
- which questions are **grounded** (have a box) and which are not;
- the **balance of yes/no answers**, since templated presence questions
  trivially skew to "no" on a dataset that is 70 % normal, and an unbalanced
  set makes accuracy meaningless;
- `docs/data_protocol.md` recording all of the above.

**Note a circularity that must be stated in the paper, not hidden:** if
questions and answers are both synthesised from the same radiologist labels
that also supply the boxes, then answer supervision and grounding supervision
are not independent. The work remains a valid testbed for evidence-grounded
reasoning, but it cannot be described as a natural clinical VQA benchmark.

---

## M5–M12 🚫 blocked

Each is blocked transitively on D1, and M7–M10 additionally on D6. Their
designs are specified in the master prompt (§§4–10); turning those into a
concrete task list requires knowing the answer space and the Teacher, so that
work is deferred rather than invented here.

Two commitments worth recording now, because they are decision-independent:

1. **The ablation matrix E0–E3** (master prompt §10.7) is the deliverable that
   makes or breaks the paper. E0 (baseline, no KD) must be run first and run
   honestly — a weak E0 would make KD-R and KD-C look effective for the wrong
   reason.
2. **No number enters any table before the run that produced it exists**
   (master prompt §10.7, §17).

---

## Standing rules for every milestone

- Nothing is reported as working without a command and its real output.
- A smoke test proves the *pipeline*, never the *method* (master prompt §12.3).
- Every experiment gets a row in `reports/experiment_registry.csv`: ID, config
  hash, split manifest hash, seed, start/end, status, checkpoint, log path.
- Original data is never modified or deleted; derived artefacts live in
  `derived/`.
- Teacher outputs are cached keyed on *(sample_id, model id, image hash,
  question, prompt version, decoding config)* — and a cache entry for the
  original image is never reused for a masked image (master prompt §5.2).
- Expected results are never written as achieved results.

---

## Immediate next actions

| # | Action | Blocked? |
|---|---|---|
| 1 | Answer **D1** and **D6** | user decision |
| 2 | Answer **D2**, **D4**, **D5** | user decision |
| 3 | **M1** — provision the local CPython 3.11 venv | **no — can start now** |
| 4 | M2 — obtain data | needs D2/D5 |
