"""Supervision targets for the decision branch, built from the data — spec §2.

Every label the answer heads learn is derived here and nowhere else:

* question type — from the record's ``type``;
* finding multi-label — the classes with a fused radiologist box on the
  image (QA+Full) — i.e. what the image contains, not only what was asked;
* Yes/No — ``True`` for every released pair (DATA_NOTES §3), ``False`` for a
  synthetic negative (``negatives.py``);
* side / count — computed geometrically from the fused boxes of
  ``gt_finding`` (spec §2 "dựng lại nhãn đếm từ annotation đầy đủ"), with
  ``answer_norm`` as the QA-only fallback;
* relevance / pointer targets — the candidate regions overlapping the
  supervision box (``regions.target_region_ids``).

Pure Python.
"""
from __future__ import annotations

from dataclasses import dataclass

from evidencekd.data.answer_norm import normalize_answer
from evidencekd.data.regions import RegionSet, target_region_ids
from evidencekd.data.vindr_processed import ImageRecord, VQASample
from evidencekd.facts.geometry import Anatomy, count_of, laterality_of
from evidencekd.labels import VQA_CLASS_TO_ID, VQA_CLASSES

QUESTION_TYPES = ("Where", "What", "Is_there", "How_many", "Which", "Yes_No")
QTYPE_TO_ID = {t: i for i, t in enumerate(QUESTION_TYPES)}
SIDES = ("left", "right", "bilateral")
SIDE_TO_ID = {s: i for i, s in enumerate(SIDES)}
COUNT_CLASSES = 5   # 0, 1, 2, 3, 4+
IGNORE = -1


@dataclass(frozen=True)
class QALabels:
    sample_id: str
    question_type: int
    yes_no: int                       # 1 / 0 / IGNORE
    findings: tuple[int, ...]         # multi-hot over VQA_CLASSES
    side: int                         # SIDE_TO_ID or IGNORE
    count: int                        # 0..4 or IGNORE
    target_regions: tuple[str, ...]   # region ids answering the question (G_c)
    asked_class: int                  # VQA_CLASS_TO_ID[gt_finding]
    is_negative: bool = False

    def as_dict(self) -> dict:
        return {"sample_id": self.sample_id, "question_type": self.question_type, "yes_no": self.yes_no,
                "findings": list(self.findings), "side": self.side, "count": self.count,
                "target_regions": list(self.target_regions), "asked_class": self.asked_class,
                "is_negative": self.is_negative}


def finding_multi_hot(rec: ImageRecord) -> tuple[int, ...]:
    present = {f.box.class_name for f in rec.boxes_fused}
    return tuple(1 if c in present else 0 for c in VQA_CLASSES)


def geometric_side_count(rec: ImageRecord, finding: str, anatomy: Anatomy | None) -> tuple[int, int]:
    boxes = [f.box for f in rec.boxes_fused if f.box.class_name == finding]
    if not boxes:
        return IGNORE, IGNORE
    count = min(count_of(boxes, finding), COUNT_CLASSES - 1)
    side = IGNORE
    if anatomy is not None:
        s = laterality_of(boxes, finding, anatomy)
        if s is not None:
            side = SIDE_TO_ID[s]
    return side, count


def labels_for(sample: VQASample, rec: ImageRecord, regions: RegionSet, *,
               anatomy: Anatomy | None, mode: str = "qa_full",
               iou_threshold: float = 0.5) -> QALabels:
    """Targets for one QA pair.

    ``mode="qa_full"`` derives side/count geometrically; ``"qa_only"`` reads
    them from the released answer text through ``answer_norm`` and leaves
    the finding multi-label at the asked class only.
    """
    qt = QTYPE_TO_ID.get(sample.type, IGNORE)
    asked = VQA_CLASS_TO_ID.get(sample.gt_finding, IGNORE)
    if mode == "qa_full":
        findings = finding_multi_hot(rec)
        side, count = geometric_side_count(rec, sample.gt_finding, anatomy)
        if asked != IGNORE and findings[asked] == 0:
            # the question's finding has no fused box of its own (repair_method
            # 'unavailable'); still mark it present — the release says it is
            findings = tuple(1 if i == asked else v for i, v in enumerate(findings))
    else:
        findings = tuple(1 if i == asked else 0 for i in range(len(VQA_CLASSES)))
        na = normalize_answer(sample.answer, question_type=sample.type, gt_finding=sample.gt_finding)
        side = SIDE_TO_ID[na.side] if na.side in SIDE_TO_ID else IGNORE
        count = IGNORE if na.count is None else min(max(na.count, 0), COUNT_CLASSES - 1)
    targets = target_region_ids(regions, sample.gt_box, iou_threshold=iou_threshold)
    return QALabels(sample.sample_id, qt, 1, findings, side, count, targets, asked, False)


def negative_labels(sample_id: str, question_type: str, asked_class: str,
                    rec: ImageRecord | None = None) -> QALabels:
    """Targets for a synthetic negative: answer No, finding absent, no regions."""
    findings = finding_multi_hot(rec) if rec is not None else tuple(0 for _ in VQA_CLASSES)
    asked = VQA_CLASS_TO_ID[asked_class]
    findings = tuple(0 if i == asked else v for i, v in enumerate(findings))
    return QALabels(sample_id, QTYPE_TO_ID.get(question_type, IGNORE), 0, findings,
                    IGNORE, 0, (), asked, True)
