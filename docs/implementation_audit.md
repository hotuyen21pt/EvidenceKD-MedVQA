# Implementation Audit — EvidenceKD-MedThink v4.1

> ## ⚠️ CORRECTION — 2026-10-19, after the method spec was supplied
>
> This audit was written before `method_spec_v4.1.md` was available. It
> assumed the dataset was raw **VinDr-CXR**. It is not. Three of its
> conclusions are **wrong** and are corrected here; the rest (environment,
> compute, licensing) stands.
>
> | §  | This audit said | Correct |
> |---|---|---|
> | 3.2 | No questions, answers or rationales exist. KD-C has no supervision source. | The project uses **[VinDr-CXR-VQA](https://huggingface.co/datasets/Dangindev/VinDR-CXR-VQA)** ([paper](https://arxiv.org/abs/2511.00504)): 4,394 images, 17,597 QA pairs, with `question`, `answer`, **`reason`** (150–250 words), `type`, `difficulty`, `gt_finding`, `gt_location`. The `reason` field is the rationale source this audit said was missing. Verified 2026-10-19. |
> | 3.1 | 22 local findings. | The VQA release covers **14 pathology classes**. The 22/6 breakdown still describes the underlying VinDr-CXR annotation used in `QA+Full` mode. |
> | 4 / D6 | No teacher is specified — hard blocker. | The spec names **MedGemma**, separately fine-tuned and frozen, with a **detector ensemble as the KD-R fallback** (spec §3, §8). |
>
> **Also note:** the HF release ships **one split (`train`) only**. Validation
> and test splits are ours to construct, by image — see
> `evidencekd.data.splits`. The VinDr-CXR test split referenced in §3.6 of
> this audit is *not* the evaluation set for this project.
>
> **Still open after the correction:** D2 (data size/resolution on disk),
> D3 (local interpreter — resolved as *not needed*; training runs on Kaggle),
> D4 (box fusion rule, for `QA+Full` mode), D5 (credentialing — **answered:
> yes**). D1 and D6 are **closed** by the spec.

**Date:** 2026-10-19
**Scope:** Phase 0 — repository and environment audit, before any code.
**Status of this document:** facts only. Every claim is tagged with how it was
established. Nothing here is a design decision; decisions are escalated in §7.

Evidence tags used throughout:

| Tag | Meaning |
|---|---|
| `MEASURED` | Observed by running a command on this machine. |
| `SOURCED` | Read from an authoritative external source (URL recorded). |
| `UNVERIFIED` | Assumed or stated in the proposal, **not yet checked**. Must not be relied on. |

---

## 1. Repository state

`MEASURED` — `ls -la`, `git status`, recursive `find`.

| Item | Finding |
|---|---|
| Project root | `C:\Users\User1\Desktop\work\paper\new` |
| Contents before this audit | **Empty.** No code, data, config, or docs. |
| Git | **Not a git repository.** `git status` → `fatal: not a git repository`. No history, nothing at risk. |
| Pre-existing EvidenceKD / MedThink artefacts | **None.** A recursive search of `C:\Users\User1\Desktop` (depth 5) for `evidence\|medthink\|vqa\|cxr\|medical` returned only unrelated hits in the `hasaki-*` business repos. |

**Conclusion:** greenfield. There is no prior implementation to preserve or
avoid breaking. The "do not delete existing work" constraint is trivially
satisfied — there is none.

### 1.1 Sibling projects set the house conventions

`MEASURED` — read `../s2it/README.md`, `../s2it/pyproject.toml`, `../tree/pyproject.toml`.

Two sibling reproduction projects exist at `../s2it` and `../tree`. They are
unrelated in subject (aspect sentiment quad prediction) but establish a
workflow this project should follow, because it is shaped by the same
hardware constraint:

- `src/<pkg>/` layout; `pyproject.toml` with `[tool.setuptools.packages.find] where = ["src"]`; `pytest` with `pythonpath = ["src"]`.
- Local machine does **data prep, tests and bundle building only** — never training.
- Training runs on **Kaggle**, offline, from a prebuilt `kaggle_offline_upload.zip` of wheels + cached artefacts.
- The Kaggle notebook is **generated** by `scripts/build_notebook.py`; the `.ipynb` is never hand-edited.
- Heavy optional dependencies are **imported lazily** to keep the offline wheelhouse small.
- Long runs **checkpoint every epoch and resume across sessions**, because one Kaggle session (~9 h) cannot hold a full run.
- `README.md` carries a "Decisions that must not be reverted" table recording *why* each non-obvious choice was made.

These are sound conventions and the rest of this plan assumes them.

---

## 2. Compute environment

`MEASURED` — `py -0p`, `Get-Command python`, `Get-CimInstance Win32_VideoController`,
`nvidia-smi`, `pip list`, `Get-PSDrive`, `Get-CimInstance Win32_ComputerSystem`.

| Item | Finding | Consequence |
|---|---|---|
| OS | Windows 11 Pro 10.0.26300 | — |
| Shells | PowerShell 5.1 + MSYS2 bash | PS 5.1 has no `&&` and no ternary; scripts must be POSIX-sh or PS-5.1-safe. |
| System RAM | 33,955,135,488 B ≈ **31.6 GiB** | Adequate for CPU-side data prep and DICOM decoding. |
| Free disk (C:) | **162.7 GiB** free / 313.3 GiB used | **Hard constraint — see §3.3.** |
| **GPU** | **Intel UHD Graphics 770** (integrated, shared memory). `nvidia-smi` not found. | **No CUDA device. No local training of any kind is possible.** |
| Python on PATH | `...\paper\.venv-build\bin\python.exe` | — |
| Python version | **3.14.3, MINGW GCC 15.2.0 64-bit (MSYS2 mingw64 build)** | See §2.1. |
| Installed packages | `colorama, iniconfig, langdetect, packaging, pip, pluggy, Pygments, pytest, six` | **No torch, numpy, pandas, pydicom, or transformers.** |
| `conda` | Not installed | — |
| `kaggle` CLI | **Not installed** | — |
| `~/.kaggle/kaggle.json` | **Does not exist** | **No Kaggle API credentials — the dataset cannot be fetched by an automated step.** |

### 2.1 The Python situation is a real constraint, not a detail

The only interpreter is an **MSYS2/MinGW build of Python 3.14.3**:

- **PyTorch publishes no wheels for Python 3.14**, and **none for MSYS2/MinGW on
  any version**. `pip install torch` on this interpreter will fail.
- The same applies to most of the scientific stack in binary-wheel form
  (`numpy`, `pandas`, `opencv`, `pydicom`'s C accelerators).
- `UNVERIFIED`: MSYS2 ships some of these as `mingw-w64-x86_64-*` pacman
  packages. Not tested.

This is survivable — it is exactly the sibling projects' situation, and they
never install torch locally. But it means:

> **Local code must run on a torch-free, numpy-optional interpreter, or we must
> provision a second interpreter.** See decision **D3**.

---

## 3. Dataset: VinDr-CXR

`SOURCED` — <https://physionet.org/content/vindr-cxr/1.0.0/> (canonical release).

The Kaggle mirror the user linked (`kaggle.com/datasets/tuktuai/vindr-cxr-physionet`)
renders client-side and could **not** be read programmatically. Its file list,
resolution and column names are therefore `UNVERIFIED` and **must be checked
against the real files once downloaded** (§3.4).

### 3.1 What VinDr-CXR actually contains

| Item | Finding |
|---|---|
| Images | **15,000 train + 3,000 test = 18,000** chest radiographs |
| Train balance | 10,606 normal / 4,394 abnormal |
| Test balance | 2,052 normal / 948 abnormal |
| Format | **DICOM** |
| Local findings (**with bounding boxes**) | **22 classes** — aortic enlargement, atelectasis, cardiomegaly, calcification, clavicle fracture, consolidation, edema, emphysema, enlarged PA, interstitial lung disease, infiltration, lung cavity, lung cyst, lung opacity, mediastinal shift, nodule/mass, pulmonary fibrosis, pneumothorax, pleural thickening, pleural effusion, rib fracture, other lesion |
| Global diagnoses (**no boxes**) | **6 classes** — lung tumor, pneumonia, tuberculosis, other diseases, COPD, no finding |
| `annotations_train.csv` columns | `image_id`, `rad_id`, `class_name`, `x_min`, `y_min`, `x_max`, `y_max` |
| `image_labels_train.csv` columns | `image_id`, `rad_ID`, + per-label binary columns |
| Annotation redundancy (train) | **Each training image read independently by 3 radiologists** |
| Annotation redundancy (test) | Consensus of 5 (3 annotators + 2 senior reviewers) |
| License | PhysioNet **Credentialed** Health Data License 1.5.0 — CITI training + signed DUA; redistribution prohibited |

### 3.2 The finding that determines the whole project

> **VinDr-CXR contains no questions, no answers, and no textual reports or
> rationales.** It is an object-detection and image-classification dataset.
> From the source: *"No textual reports, questions, answers, or rationales are
> included — only images and structured labels."*

Mapping the proposal's required inputs onto what the dataset supplies:

| Proposal needs | VinDr-CXR supplies | Gap |
|---|---|---|
| Question `Q` | — | **Entirely absent.** |
| Answer `A` | — | **Entirely absent.** Derivable from labels only under a declared Q/A construction scheme. |
| Evidence regions `B` | ✅ 22 classes of radiologist boxes, 3 annotators each | Present, but **3 disagreeing annotators** — no single ground-truth box. See §3.5. |
| Rationale / claims `C` | — | **Entirely absent.** No free text of any kind. |
| Teacher rationales (KD-C) | — | Must come from an external VLM (§4). |
| Teacher answer probabilities (KD-R) | — | Must come from an external model (§4). |

**Therefore** proposal §7 (KD-C), §9.3 (claim verification) and §10.3 (claim
metrics) have **no supervision source in this dataset as it stands**. KD-R (§6)
is mechanically feasible — it needs only regions plus a Teacher that scores
answers — but it still requires a question and a target answer to score, so it
inherits the same gap.

This is precisely the blocker the master prompt §2.2 anticipates:
*"Không được kết luận rằng dataset hỗ trợ grounding nếu chưa kiểm tra schema và
dữ liệu thực tế."* The schema has now been checked. It supports **grounding**.
It does **not** support **VQA** or **rationale** supervision without an explicit,
declared construction step.

### 3.3 Size vs. available disk

The PhysioNet release is full-resolution DICOM, on the order of **~190 GB**.
Free disk is **162.7 GiB** (`MEASURED`).

> **The full-resolution original does not fit on this machine.**

Mitigations, in order of preference:

1. The Kaggle mirror may already be downsampled to PNG/JPEG (common for CXR
   mirrors). `UNVERIFIED` — check first. If it is 8-bit PNG at ~1024 px, the
   whole set is likely 10–20 GB and fits comfortably.
2. Work on a declared subset.
3. Stream DICOM → resized PNG and delete DICOM as we go.

Resolving this is **decision D2**.

### 3.4 Access

- No `kaggle` CLI and no `kaggle.json` → the dataset **cannot currently be
  downloaded by an automated step from this machine.**
- PhysioNet additionally requires **CITI training + a signed DUA** under a
  *Credentialed* license.
- `UNVERIFIED`: whether the user holds PhysioNet credentialing. This is not a
  footnote — a paper on credentialed PhysioNet data is expected to state that
  the authors are credentialed, and the license forbids redistribution, which
  constrains what may go into a public repo or an uploaded Kaggle dataset.
  Flagged as **D5**.

### 3.5 Three radiologists disagree — grounding has no single ground truth

`SOURCED`. `annotations_train.csv` carries `rad_id`, and every training image
was read by 3 radiologists **independently, without consensus**.

Consequences the proposal does not currently address:

- "Ground-truth box" is undefined until a **fusion rule** is declared
  (e.g. Weighted Box Fusion, or IoU-linked majority vote at a stated threshold).
- The grounding metrics (§10.2 Box Recall@K, IoU) and the KD-R sample filter
  ("relevance mass on the target region", §6.7) both silently assume one box
  per finding. Both need redefining against the chosen fusion rule.
- Inter-rater disagreement is itself a **legitimate signal** for the
  `SUPPORTED / CONTRADICTED / INSUFFICIENT` trichotomy the proposal is built
  around: an image where 1 of 3 radiologists saw a finding is a genuine
  `INSUFFICIENT` case, not a labelling error. **This is an opportunity**, but it
  must be designed deliberately rather than averaged away.

Flagged as **D4**.

### 3.6 Patient-level leakage

`UNVERIFIED`. Proposal §3.1 requires patient-level split checking. VinDr-CXR
identifies images by an anonymised `image_id`; whether a patient identifier
survives in the DICOM headers, and whether any patient contributed more than
one study, has **not been checked** and cannot be until the data is on disk.

> Until then, **no claim that the split is leakage-free may be made.**

---

## 4. Teacher model — unspecified and on the critical path

The proposal requires a Teacher that can do two things:

1. **(KD-R, §6.1)** return a meaningful `p_T(a* | I, Q)`, and
   `p_T(a* | I∖Rᵢ, Q)` for every masked region. The master prompt is explicit
   that an arbitrary confidence score is not a substitute.
2. **(KD-C, §7)** generate free-text rationales that can be split into claims.

**No Teacher is named anywhere in the proposal or the repository.** This is
**D6**, and it gates both core contributions.

Constraints any candidate must satisfy:

- Runs **offline on Kaggle** (a Kaggle training session has no internet) within
  **16 GiB VRAM** on a T4/P100.
- Exposes **token-level logprobs** for the answer span, not just text.
- Licensed for research use, with the license recorded.
- `UNVERIFIED`: candidate medical VLMs (LLaVA-Med, MedGemma, CheXagent, MAIRA)
  have not been checked for Kaggle offline availability, VRAM fit, or logprob
  access.

### 4.1 KD-R cost is the single largest resource risk

KD-R needs **1 + N_regions** Teacher forward passes per (image, question) pair.
Order-of-magnitude, to be refined once a Teacher is chosen:

```
15,000 images × 3 questions/image × (1 + 8 regions) ≈ 405,000 VLM forward passes
```

At even 0.3 s/pass on a T4 that is **~34 GPU-hours** — before any student
training, against a ~9 h session cap. Feasible only with sharded, resumable,
cached generation across many sessions, which proposal §6.8 already
anticipates. **It must be budgeted explicitly before any KD-R run starts**
(master prompt §13).

---

## 5. Architecture feasibility — not yet verified

Every row here is `UNVERIFIED`. No component has been imported, instantiated,
or benchmarked (there is no local GPU to benchmark on).

| Component (proposal §1.2) | Known concern |
|---|---|
| ResNet-50 + FPN | Low risk; `torchvision` provides both. |
| CXR-BERT | `microsoft/BiomedVLP-CXR-BERT-specialized` exists on HF under an **MSR research license whose terms must be read and recorded** before use. Must be bundled as a Kaggle offline artefact. |
| Cross-attention fusion | Low risk — custom code. |
| **Question-conditioned Deformable DETR** | **Highest-risk component.** Standard Deformable DETR depends on a **custom CUDA kernel (`MultiScaleDeformableAttention`) that must be compiled**; offline Kaggle builds of custom CUDA extensions are fragile. A pure-PyTorch fallback exists in HF `transformers` but is substantially slower. *Question-conditioning* is a modification present in no released checkpoint, so it must be trained, not loaded. |
| Region Encoder | Low risk. |
| Geometric Fact Table | Low risk — deterministic code over box coordinates. Arguably the most defensible component, since its facts are computed rather than predicted. |
| Answer Head | Low risk. |
| Evidence Pointer | Low risk — custom. |
| Rationale Generator | Needs a decoder LM; VRAM must be budgeted **jointly** with the Teacher if both are resident. |
| Claim Verifier | Needs its own model or rules. Proposal §7.2 correctly warns that a VLM's output is not medical ground truth. **Note the self-evaluation bias flagged in §10.3: if one verifier both filters KD-C targets and scores results, the claim metrics are not independent.** |
| Answer Lock | Low risk — a hard constraint in the inference path, easy to test. |

**Recommendation:** build the pipeline against a **small stand-in
configuration** first (small CNN, small text encoder, grid regions instead of
Deformable DETR) so the *protocol* can be validated on CPU, then swap in the
target architecture once it is proven loadable on Kaggle. Master prompt §4.1
explicitly permits this, provided the stand-in is labelled as such.

---

## 6. What this means for the proposal, honestly

| Proposal component | Buildable from what exists today? |
|---|---|
| Grounding / region supervision | **Yes** — real radiologist boxes, 22 classes. The dataset's strength. |
| Answer prediction | **Only after a Q/A construction scheme is declared.** Answers would derive from the same labels as the boxes; the paper must say so plainly. |
| KD-R | **Mechanically yes**, once a Teacher with logprob access exists. Compute cost is the binding constraint, not correctness. |
| KD-C | **No supervision source exists.** Rationales would be entirely Teacher-generated and the verifier automatic. The resulting claim metrics are **model-vs-model, not model-vs-clinician**. |
| Clinician-validated claims | **No.** Not available, not budgeted, not planned. |

None of this makes the project unviable. It does mean the paper's claims must
be scoped to what the evidence supports, and that several framing decisions
have to be made **now** rather than discovered at write-up time.

---

## 7. Decisions requiring the user — blocking

Each of these changes what the research *is*, so per master prompt §0 they are
escalated rather than silently resolved.

| ID | Decision | Why it blocks | Suggested default |
|---|---|---|---|
| **D1** | **Where do questions and answers come from?** Synthesise templated Q/A from VinDr-CXR labels; or bring in a real medical VQA set (VQA-RAD / SLAKE / PathVQA) and accept weaker or absent boxes; or pair two datasets. | Determines the task, answer space, loss (classification vs generation), every metric, and whether grounding and answering share a label source. **Nothing downstream can be designed without it.** | *None — must be answered.* |
| **D2** | **Which copy of the data, at what resolution?** Full-res DICOM does not fit in 162.7 GiB. | Determines whether the data is usable on this machine at all. | Check the Kaggle mirror's real size/format first. |
| **D3** | **Provision a second Python interpreter?** The only local one is MSYS2 Python 3.14 with no torch and no numpy. | Determines whether anything beyond pure Python can be tested locally. | Install a CPython 3.11/3.12 venv for data prep + CPU smoke tests; keep training on Kaggle. |
| **D4** | **Box fusion rule across the 3 radiologists.** | Defines "ground-truth box", hence all grounding metrics and the KD-R region set. | WBF at a stated IoU; additionally use disagreement as a genuine `INSUFFICIENT` signal. |
| **D5** | **PhysioNet credentialing status**, and what may legally be committed or uploaded. | License compliance; constrains the Kaggle bundle and any public repo. | Assume credentialed; commit **no** image data; confirm before any upload. |
| **D6** | **Which Teacher model**, and does it expose answer logprobs? | KD-R is undefined without `p_T(a*\|·)`; KD-C is undefined without rationales. | *None — must be answered.* |

---

## 8. Audit conclusions

1. The repository is **empty**; a skeleton of directories has been created
   (no logic). Nothing is at risk.
2. The machine **cannot train anything** — no CUDA device. All training must
   run on Kaggle, following the sibling projects' offline-bundle pattern.
3. The dataset **is not downloaded**, **cannot be downloaded automatically**
   (no CLI, no credentials), and at full resolution **would not fit**.
4. VinDr-CXR provides **excellent grounding supervision** and **no VQA data at
   all**. The VQA layer must be constructed, and that construction is a
   research decision, not an implementation detail.
5. **KD-C has no supervision source** beyond what a Teacher invents and an
   automatic verifier approves. Reportable, but only if framed honestly.
6. **No Teacher has been chosen**, and the choice gates both core
   contributions.

> **Nothing has been implemented beyond empty directories. No training has run.
> No results exist. No metric in this project has a value yet.**

### Next step

Resolve **D1** and **D6** (hard blockers) and check **D2** by inspecting the
actual Kaggle mirror. `experiment_plan.md` is deliberately written only to the
point where those decisions bite.
