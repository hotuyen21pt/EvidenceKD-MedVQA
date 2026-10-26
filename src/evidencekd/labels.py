"""VinDr-CXR label space.

The canonical class names for VinDr-CXR v1.0.0: 22 *local* findings that carry
bounding boxes, and 6 *global* diagnoses that do not.

Source: https://physionet.org/content/vindr-cxr/1.0.0/

.. warning::
   The exact capitalisation used in ``annotations_train.csv`` has **not** been
   verified against the real file (the dataset is not on disk yet). Every
   lookup here is therefore case- and punctuation-insensitive via
   :func:`normalize_class_name`, and :mod:`evidencekd.data.validate` reports any
   ``class_name`` it cannot map rather than silently dropping it. Do not add a
   name here without seeing it in the actual CSV.
"""

from __future__ import annotations

from typing import Final

#: The 22 local findings annotated with bounding boxes.
LOCAL_FINDINGS: Final[tuple[str, ...]] = (
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged PA",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Lung cavity",
    "Lung cyst",
    "Mediastinal shift",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
)

#: The 6 global diagnoses. These have no bounding boxes.
GLOBAL_DIAGNOSES: Final[tuple[str, ...]] = (
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
)

#: The sentinel class VinDr uses for a radiograph with nothing to report.
NO_FINDING: Final[str] = "No finding"

#: Human-readable phrasing used when a class name appears inside a question or
#: a claim. Only entries that differ from a plain lower-casing are listed.
_READABLE: Final[dict[str, str]] = {
    "ILD": "interstitial lung disease",
    "Enlarged PA": "an enlarged pulmonary artery",
    "Nodule/Mass": "a nodule or mass",
    "COPD": "COPD",
    "Lung Opacity": "lung opacity",
}


def normalize_class_name(raw: str) -> str:
    """Return a canonical key for a VinDr class name.

    Lower-cases, collapses whitespace, and strips characters that vary between
    releases, so ``"Nodule/Mass"``, ``"nodule/mass"`` and ``"Nodule / Mass"``
    all agree.

    >>> normalize_class_name("Nodule / Mass") == normalize_class_name("nodule/mass")
    True
    """
    return "".join(ch for ch in raw.lower() if ch.isalnum())


#: Canonical key -> official class name, for every known class.
_BY_KEY: Final[dict[str, str]] = {
    normalize_class_name(name): name for name in LOCAL_FINDINGS + GLOBAL_DIAGNOSES
}


def canonical_class_name(raw: str) -> str | None:
    """Map an arbitrary spelling to its official class name.

    Returns ``None`` when the name is unknown. Callers must treat ``None`` as
    "report this and stop", never as "skip this row quietly" — an unmapped
    class means our label space disagrees with the data.
    """
    return _BY_KEY.get(normalize_class_name(raw))


def is_local_finding(raw: str) -> bool:
    """True when ``raw`` names one of the 22 box-annotated findings."""
    return canonical_class_name(raw) in set(LOCAL_FINDINGS)


def readable(raw: str) -> str:
    """Phrase a class name for use inside a question or a claim.

    >>> readable("ILD")
    'interstitial lung disease'
    >>> readable("Cardiomegaly")
    'cardiomegaly'
    """
    name = canonical_class_name(raw)
    if name is None:
        raise KeyError(f"unknown VinDr class name: {raw!r}")
    if name in _READABLE:
        return _READABLE[name]
    return name.lower()


# ---------------------------------------------------------------------------
# The 14 classes of the VinDr-CXR-VQA release (a subset of LOCAL_FINDINGS),
# ordered by prevalence in data_v1.json exactly as
# ``vindr_cxr_vqa.constants.PATHOLOGIES`` lists them. Kept here verbatim so the
# torch-free core does not depend on the sibling repository being importable;
# ``tests/test_vindr_processed.py`` asserts the two lists agree when it is.
# ---------------------------------------------------------------------------
VQA_CLASSES: Final[tuple[str, ...]] = (
    "Cardiomegaly",
    "Pulmonary fibrosis",
    "Lung Opacity",
    "Pleural effusion",
    "Aortic enlargement",
    "Calcification",
    "Other lesion",
    "Pleural thickening",
    "Infiltration",
    "Pneumothorax",
    "Nodule/Mass",
    "ILD",
    "Consolidation",
    "Atelectasis",
)

VQA_CLASS_TO_ID: Final[dict[str, int]] = {c: i for i, c in enumerate(VQA_CLASSES)}

#: Lower-case substrings that name each class in free text (question or
#: answer). Mirrors ``vindr_cxr_vqa.constants.PATHOLOGY_SYNONYMS``.
VQA_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "Cardiomegaly": ("cardiomegaly", "enlarged heart", "cardiac enlargement",
                     "enlargement of the cardiac silhouette"),
    "Pulmonary fibrosis": ("pulmonary fibrosis", "fibrosis", "fibrotic"),
    "Lung Opacity": ("lung opacity", "pulmonary opacity", "opacity", "opacities"),
    "Pleural effusion": ("pleural effusion", "effusion"),
    "Aortic enlargement": ("aortic enlargement", "enlarged aorta",
                           "aortic dilatation", "widened aorta"),
    "Calcification": ("calcification", "calcified", "calcific"),
    "Other lesion": ("other lesion",),
    "Pleural thickening": ("pleural thickening", "thickened pleura"),
    "Infiltration": ("infiltration", "infiltrate"),
    "Pneumothorax": ("pneumothorax",),
    "Nodule/Mass": ("nodule/mass", "nodule", "nodular", "mass"),
    "ILD": ("ild", "interstitial lung disease", "interstitial disease"),
    "Consolidation": ("consolidation", "consolidative"),
    "Atelectasis": ("atelectasis", "atelectatic", "collapse"),
}


def finding_in_text(text: str) -> str | None:
    """The VQA class a piece of text names, or ``None``.

    Longest synonym wins so that "pleural effusion" is not read as the bare
    "effusion" of another class, and "lung opacity" beats "opacity".

    >>> finding_in_text("Where is the Pleural effusion?")
    'Pleural effusion'
    >>> finding_in_text("How many findings are there?") is None
    True
    """
    low = text.lower()
    best: tuple[int, str] | None = None
    for cls, syns in VQA_SYNONYMS.items():
        for s in syns:
            if s in low and (best is None or len(s) > best[0]):
                best = (len(s), cls)
    return best[1] if best else None
