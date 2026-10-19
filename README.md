# EvidenceKD-MedThink v4.1

Evidence-grounded medical VQA on chest X-rays: a decision branch produces the
answer, findings and scores; a generator only *explains* them, under a
citation constraint; a verifier checks each claim.

Implementation of the method spec `docs/method_spec_v4.1.md` (CV2 —
EvidenceKD-MedThink v4.1).

> **Status: early implementation. No model has been trained. No result exists.**
> Every number in the spec's result tables is still blank, and nothing in this
> repository should be read as an experimental finding.

---

## Current state

| | |
|---|---|
| Research logic (schema, fusion, KD-R math, claim grammar, Answer Lock, splits) | **Implemented, 151 tests passing** |
| Model code (encoders, detector, heads, generator, verifier) | **Not written** |
| Teacher (MedGemma) integration | **Not written** |
| Training loops | **Not written** |
| Data on disk | **No** — dataset not yet downloaded |
| Any training run | **No** |
| Any reported metric | **No** |

```
$ python -m pytest -q
151 passed in 0.15s
```

Run on MSYS2 CPython 3.14.3, no GPU, no torch installed. That is deliberate —
see [Why the core is torch-free](#why-the-core-is-torch-free).

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
  labels.py                 VinDr class names, case-insensitive canonicalisation
  data/
    vqa_schema.py           VinDr-CXR-VQA records, <loc_...> parsing
    answer_norm.py          THE answer normalisation rule (Yes/No, finding, side, count)
    boxes.py                IoU, multi-radiologist box fusion, Recall@K
    splits.py               split-by-image, leakage guards, split manifest hash
  facts/geometry.py         the geometric fact table (spec §4.2)
  claims/parser.py          claim grammar, citation whitelist, constrained decoding
  kd/
    kd_r.py                 counterfactual relevance, null-slot softmax, gated KL
    kd_c.py                 claim-level filtering (none / sample / claim), KD-C loss
  inference/answer_lock.py  Answer Lock, Evidence Pointer, NO_SUPPORT
  models/ teacher/ eval/    not yet implemented
tests/                      151 tests, no GPU required
docs/
  method_spec_v4.1.md       the method specification this implements
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

Being explicit, because the gap is large:

- **No model code.** Vision encoder, text encoder, fusion, question-conditioned
  Deformable DETR, region/evidence encoders, answer heads, generator and
  verifier are all unwritten.
- **No teacher.** MedGemma is named in the spec but not integrated, and its
  answer-logprob access has not been verified. KD-R is undefined without it.
- **No training loops, no checkpointing, no resume.**
- **No data.** The dataset has not been downloaded; the schema above is from
  the published dataset card, not from files on disk.
- **No evaluation.** None of Tables A–F has a single value.
- **No clinician annotation.** The spec calls for 800–1000 claims double-read
  with a reported kappa. Not started, not budgeted.

Known risks worth naming now:

- **Question-conditioned Deformable DETR** needs a custom CUDA kernel;
  building it offline on Kaggle is fragile. A pure-PyTorch fallback exists but
  is slow.
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
