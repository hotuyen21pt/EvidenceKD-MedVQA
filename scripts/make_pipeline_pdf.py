"""Render docs/PIPELINE_VISUAL.pdf — one question traced through every stage.

The Vietnamese companion to README's "Luồng xử lý chi tiết": the same trace,
but with the chest X-ray embedded and every box drawn on it at true scale, so
"R01 is the consensus of three radiologists and it overlaps the supervision
box at IoU 0,72" becomes something you see rather than take on faith. The KD-R
section shows the masked images themselves — the whole method is "hide this
region and watch the teacher's confidence fall", which is a picture, not a
paragraph.

Nothing data-side is transcribed by hand: ``collect()`` recomputes the boxes,
the fusion, the target region, the labels, the fact table and the <loc> tokens
from this repo's own modules. It needs ``data/processed`` built
(``python scripts/build_data.py``) and the VinDr PNGs on disk at
``../VinDR-CXR-VQA/dataset``.

Network outputs (logits, probabilities, teacher log-probs) are ILLUSTRATIVE —
no training run has completed. They live in one block, ``ILLUSTRATIVE``, and
every figure derived from them is labelled in the document.

Layout is HTML printed by headless Chrome or Edge, which is what gives real
page breaks, Vietnamese typography and inline SVG without a LaTeX toolchain.

    python scripts/make_pipeline_pdf.py
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.normpath(os.path.join(REPO, "..", "VinDR-CXR-VQA")))

#: The question this document follows, start to finish.
SID = "06b2ff94bd0270f543deb7074c2a006a_1"
#: Where the 224 px PNGs live, relative to the repo.
IMAGES = os.path.normpath(os.path.join(REPO, "..", "VinDR-CXR-VQA", "dataset",
                                       "train_data", "train"))

BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
)

# ---------------------------------------------------------------------------
# Every number below is an ILLUSTRATION of the right shape and the right
# formula. None of it is measured: no training run has completed. The document
# marks every block that depends on this dict.
ILLUSTRATIVE = {
    "relevance_logits": [2.1, -1.3, -0.4, 0.2],
    "finding_probs": {"Pulmonary fibrosis": 0.93, "Aortic enlargement": 0.72, "ILD": 0.61},
    "yes_no": [0.06, 0.94],
    "side": {"left": 0.12, "right": 0.08, "bilateral": 0.80},
    "count": [0.05, 0.20, 0.65, 0.07, 0.03],
    "pointer_logits": [1.8, -0.9, -0.2, 0.3],
    "pointer_null": -1.0,
    "cos": [0.62, 0.10, 0.21, 0.35],
    "tau_align": 0.07,
    # teacher-forced mean log-prob of a* on the original and on each masked image
    "teacher_logp": {"orig": -0.42, "R01": -0.95, "R02": -0.44, "R03": -0.40, "R04": -0.50},
    # stage-2 faithfulness: logit of the asked finding with each region's token dropped
    "faith_logit_full": 2.6,
    "faith_logit_drop": {"R01": 0.90, "R02": 2.50, "R03": 2.55, "R04": 2.20},
    "pneumothorax_score": 0.02,
}


def find_browser() -> str:
    for cand in BROWSERS:
        if os.path.isabs(cand):
            if os.path.exists(cand):
                return cand
        elif shutil.which(cand):
            return shutil.which(cand)
    raise SystemExit("No headless Chrome or Edge found — this script prints the PDF "
                     "with one.\nInstall Chrome, or add your binary to BROWSERS.")


def e(s) -> str:
    return html.escape(str(s))


def vn(x, dec=None) -> str:
    """Vietnamese number: '.' groups thousands, ',' marks the decimal.

    ``dec=0`` rounds without introducing a decimal comma -- the English
    separators are swapped, not appended, or 2045.33 prints as "2.045,".
    """
    s = f"{int(round(x)):,}" if dec is None else f"{x:,.{dec}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def img64(image_id: str) -> str:
    path = os.path.join(IMAGES, image_id + ".png")
    if not os.path.isfile(path):
        raise SystemExit(f"[pdf] missing PNG: {path}\nBuild or attach the VinDr images first.")
    return base64.b64encode(open(path, "rb").read()).decode()


def softmax(xs, tau=1.0):
    ex = [math.exp(x / tau) for x in xs]
    s = sum(ex)
    return [v / s for v in ex]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ------------------------------------------------------------------- collect
def collect() -> dict:
    """Every data-side value the document shows, computed from the repo."""
    from evidencekd.data.anatomy import fallback_anatomy
    from evidencekd.data.boxes import iou
    from evidencekd.data.qa_labels import labels_for
    from evidencekd.data.regions import regions_from_annotation, target_region_ids
    from evidencekd.data.vindr_processed import load_images, load_samples
    from evidencekd.eval.vindr_bridge import loc_token, to_norm1000
    from evidencekd.facts.build import build_facts, fact_value_text
    from evidencekd.inference.templates import locked_answer_text

    pd_dir = os.path.join(REPO, "data", "processed")
    image_id = SID.rsplit("_", 1)[0]
    images = load_images(pd_dir)
    rec = images[image_id]

    sample = split = None
    for name in ("val", "train", "test"):
        for s in load_samples(pd_dir, name):
            if s.sample_id == SID:
                sample, split = s, name
                break
        if sample:
            break
    if sample is None:
        raise SystemExit(f"[pdf] {SID} not found in data/processed")

    anat = fallback_anatomy(rec.width, rec.height)
    regions = regions_from_annotation(rec, anatomy=anat)
    targets = target_region_ids(regions, sample.gt_box)
    labels = labels_for(sample, rec, regions, anatomy=anat)
    finding_regions = list(regions.finding_regions)

    facts = build_facts(
        regions, anat,
        present_classes=[r.label for r in finding_regions],
        whole_image_scores={"Pneumothorax": ILLUSTRATIVE["pneumothorax_score"]},
        absence_threshold=0.3)

    r01 = regions.by_id("R01")
    answer = locked_answer_text("Where", finding=sample.gt_finding, present=True,
                                side="bilateral", count=labels.count)

    # IoU(R01, supervision box) written out term by term, the way the doc shows it
    gt = sample.gt_box
    ix = max(0.0, min(r01.box.x_max, gt.x_max) - max(r01.box.x_min, gt.x_min))
    iy = max(0.0, min(r01.box.y_max, gt.y_max) - max(r01.box.y_min, gt.y_min))

    return {
        "sample": sample, "split": split, "rec": rec, "regions": regions,
        "finding_regions": finding_regions, "targets": targets, "labels": labels,
        "facts": facts, "fact_text": fact_value_text, "anat": anat,
        "answer": answer, "r01": r01,
        "iou_r01": iou(r01.box, gt), "inter_w": ix, "inter_h": iy, "inter": ix * iy,
        "gt_area": gt.area, "r01_area": r01.box.area,
        "loc_r01": loc_token(r01.box, rec.width, rec.height),
        "loc_gt": loc_token(gt, rec.width, rec.height),
        "norm_r01": to_norm1000(r01.box, rec.width, rec.height),
        "iou_fn": iou, "loc_fn": loc_token,
    }


# --------------------------------------------------------------- svg helpers
PAL = {
    "pf": "#b91c1c", "ild": "#0b6bcb", "ae": "#7c3aed", "fused": "#047857",
    "gt": "#ea580c", "anat": "#64748b", "mask": "#0f172a",
    "ink": "#1f2933", "mute": "#64748b", "rule": "#cbd5e1", "accent": "#0b6bcb",
}
CLASS_COLOR = {"Pulmonary fibrosis": PAL["pf"], "ILD": PAL["ild"],
               "Aortic enlargement": PAL["ae"]}


def xray(image_id, ow, oh, boxes, size=232, caption=None, legend=True, title=None):
    """The real PNG at `size` px with `boxes` (ORIGINAL pixels) drawn on it.

    The PNG is a 224x224 squash of a non-square original, so x and y scale
    independently — exactly what VQASample.fx / .fy do.
    """
    sx, sy = size / ow, size / oh
    p = [f'<svg class="xr" viewBox="0 0 {size + 2} {size + 2}" width="{size + 2}" height="{size + 2}">',
         f'<image x="1" y="1" width="{size}" height="{size}" '
         f'href="data:image/png;base64,{img64(image_id)}"/>']
    for b in boxes:
        x1, y1 = b["box"].x_min * sx + 1, b["box"].y_min * sy + 1
        x2, y2 = b["box"].x_max * sx + 1, b["box"].y_max * sy + 1
        if b.get("mask"):
            p.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" '
                     f'height="{y2 - y1:.1f}" fill="{PAL["mask"]}"/>')
            continue
        dash = f' stroke-dasharray="{b["dash"]}"' if b.get("dash") else ""
        p.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{max(x2 - x1, 1.2):.1f}" '
                 f'height="{max(y2 - y1, 1.2):.1f}" fill="none" stroke="{b["color"]}" '
                 f'stroke-width="{b.get("width", 1.5)}"{dash}/>')
        if b.get("tag"):
            tw = len(b["tag"]) * 4.4 + 6
            ty = y1 - 2.5 if y1 > 13 else y2 + 10
            p.append(f'<rect x="{x1:.1f}" y="{ty - 7.5:.1f}" width="{tw:.1f}" height="9.5" '
                     f'rx="2" fill="{b["color"]}"/>')
            p.append(f'<text x="{x1 + 3:.1f}" y="{ty - 0.5:.1f}" font-size="6.8" fill="#fff" '
                     f'font-weight="700">{e(b["tag"])}</text>')
    p.append(f'<rect x="1" y="1" width="{size}" height="{size}" fill="none" stroke="{PAL["rule"]}"/>')
    p.append("</svg>")
    leg = ""
    if legend:
        rows = "".join(
            f'<div class="lg"><span class="sw" style="border-color:{b["color"]};'
            f'{"border-style:dashed;" if b.get("dash") else ""}"></span>{e(b["label"])}</div>'
            for b in boxes if b.get("label"))
        leg = f'<div class="legend">{rows}</div>' if rows else ""
    head = f'<div class="pane-h">{e(title)}</div>' if title else ""
    cap = f'<div class="cap">{caption}</div>' if caption else ""
    return f'<div class="fig">{head}{"".join(p)}{leg}{cap}</div>'


def bars(values, labels, *, colors=None, width=430, rowh=15, vmax=None,
         fmt=lambda v: vn(v, 3), rule=None, rule_label=""):
    """Horizontal bar chart — used for every probability vector in the doc."""
    vmax = vmax or max(values + ([rule] if rule else [])) or 1.0
    lw, bw = 74, width - 74 - 46
    h = rowh * len(values) + 8
    p = [f'<svg class="bars" viewBox="0 0 {width} {h}" width="{width}" height="{h}">']
    for i, (v, lb) in enumerate(zip(values, labels)):
        y = i * rowh + 4
        col = (colors or [PAL["accent"]] * len(values))[i]
        p.append(f'<text x="{lw - 6}" y="{y + 8.5}" font-size="8" text-anchor="end" '
                 f'fill="{PAL["ink"]}" font-family="Consolas,monospace">{e(lb)}</text>')
        p.append(f'<rect x="{lw}" y="{y + 1.5}" width="{bw}" height="{rowh - 5}" fill="#eef2f7"/>')
        p.append(f'<rect x="{lw}" y="{y + 1.5}" width="{max(v / vmax * bw, 0.6):.1f}" '
                 f'height="{rowh - 5}" fill="{col}"/>')
        p.append(f'<text x="{lw + bw + 5}" y="{y + 8.5}" font-size="8" fill="{PAL["mute"]}" '
                 f'font-family="Consolas,monospace">{fmt(v)}</text>')
    if rule:
        x = lw + rule / vmax * bw
        p.append(f'<line x1="{x:.1f}" y1="2" x2="{x:.1f}" y2="{h - 6}" stroke="{PAL["gt"]}" '
                 f'stroke-width="1.2" stroke-dasharray="3 2"/>')
        p.append(f'<text x="{x + 3:.1f}" y="{h - 1}" font-size="7.2" fill="{PAL["gt"]}">'
                 f'{e(rule_label)}</text>')
    p.append("</svg>")
    return f'<div class="fig">{"".join(p)}</div>'


def flow_svg():
    """The ten steps, as boxes with the artefact each one writes."""
    rows = [
        ("1", "Dữ liệu vào", "data/vindr_processed", "data/processed/*.jsonl", "once"),
        ("2", "Detector (GĐ 0)", "models/detector", "detector.pt · detections_*.json", "train"),
        ("3", "Vùng dẫn xuất", "data/build_regions", "derived/regions_*.json · labels_train.jsonl", "once"),
        ("4", "Self-test oracle", "inference/oracle_decisions", "P = 1,0 · IoU≥0,5 = 99,85 %", "check"),
        ("5", "Decision branch GĐ 1", "models/decision_branch", "decision_stage1.pt · thresholds.json", "train"),
        ("6", "Teacher MedGemma", "teacher/kd_r_targets · kd_c_targets", "cache/kd_r · cache/kd_c", "teach"),
        ("7", "Decision branch GĐ 2", "decision_branch --stage 2", "decision_stage2.pt", "train"),
        ("8", "Verifier · generator", "models/verifier · generator_s", "verifier.pt · generator_s/final", "train"),
        ("9", "Suy luận và chấm", "inference/pipeline · eval/*", "Bảng A–C/F · Bảng 4–6", "infer"),
        ("10", "Phục vụ một ảnh", "serve.py", "HTTP, cùng hợp đồng UI VinDr", "infer"),
    ]
    kind = {"once": ("#64748b", "MỘT LẦN"), "train": ("#0b6bcb", "TRAIN"),
            "check": ("#047857", "KIỂM TRA"), "teach": ("#7c3aed", "TEACHER"),
            "infer": ("#b45309", "SUY LUẬN")}
    w, rh = 520, 25.5
    h = rh * len(rows) + 6
    p = [f'<svg class="flow" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    for i, (n, name, mod, art, k) in enumerate(rows):
        y = i * rh + 3
        col, tag = kind[k]
        if i:
            p.append(f'<line x1="18" y1="{y - 3.5}" x2="18" y2="{y + 1}" stroke="{PAL["rule"]}" stroke-width="1.2"/>')
        p.append(f'<circle cx="18" cy="{y + 10}" r="8.5" fill="{col}"/>')
        p.append(f'<text x="18" y="{y + 13}" font-size="8" fill="#fff" text-anchor="middle" '
                 f'font-weight="700">{n}</text>')
        p.append(f'<text x="34" y="{y + 9}" font-size="9" font-weight="600" fill="{PAL["ink"]}">{e(name)}</text>')
        p.append(f'<text x="34" y="{y + 19}" font-size="7.6" fill="{PAL["mute"]}" '
                 f'font-family="Consolas,monospace">{e(mod)}</text>')
        p.append(f'<rect x="170" y="{y + 2}" width="47" height="11" rx="5.5" fill="{col}" opacity=".13"/>')
        p.append(f'<text x="193.5" y="{y + 10}" font-size="6.6" fill="{col}" text-anchor="middle" '
                 f'font-weight="700" letter-spacing=".4">{e(tag)}</text>')
        p.append(f'<text x="228" y="{y + 14}" font-size="7.8" fill="{PAL["ink"]}" '
                 f'font-family="Consolas,monospace">{e(art)}</text>')
    p.append("</svg>")
    return f'<div class="fig">{"".join(p)}</div>'


def ruler(d):
    """The original-pixel → norm1000 conversion, drawn."""
    r, rec = d["r01"].box, d["rec"]
    xs = [("x_min", r.x_min, 2500), ("y_min", r.y_min, 3012),
          ("x_max", r.x_max, 2500), ("y_max", r.y_max, 3012)]
    w, rh = 470, 19
    p = [f'<svg viewBox="0 0 {w} {rh * 4 + 20}" width="{w}" height="{rh * 4 + 20}">']
    for i, (name, v, span) in enumerate(xs):
        y = i * rh + 12
        frac = v / span
        p.append(f'<text x="42" y="{y + 4}" font-size="8" text-anchor="end" '
                 f'font-family="Consolas,monospace" fill="{PAL["ink"]}">{e(name)}</text>')
        p.append(f'<rect x="48" y="{y - 3.5}" width="250" height="8" fill="#eef2f7"/>')
        p.append(f'<rect x="48" y="{y - 3.5}" width="{frac * 250:.1f}" height="8" fill="{PAL["accent"]}" opacity=".75"/>')
        p.append(f'<text x="306" y="{y + 4}" font-size="7.6" fill="{PAL["mute"]}" '
                 f'font-family="Consolas,monospace">{vn(v, 1)} / {vn(span)}</text>')
        p.append(f'<text x="398" y="{y + 4}" font-size="8" fill="{PAL["accent"]}" '
                 f'font-family="Consolas,monospace" font-weight="700">→ {int(frac * 999)}</text>')
    p.append(f'<text x="48" y="{rh * 4 + 16}" font-size="7.4" fill="{PAL["mute"]}">'
             f'norm1000: toạ độ · 999 / cạnh ảnh gốc, làm tròn xuống</text>')
    p.append("</svg>")
    return f'<div class="fig">{"".join(p)}</div>'


# ------------------------------------------------------------------- document
def build_body(d) -> str:
    rec, sample, regions = d["rec"], d["sample"], d["regions"]
    iid = rec.image_id
    fr = d["finding_regions"]
    I = ILLUSTRATIVE

    raw_boxes = [{"box": b, "color": CLASS_COLOR.get(b.class_name, PAL["mute"]),
                  "label": f"{b.class_name} · {b.rad_id}", "width": 1.3}
                 for b in rec.boxes_raw]
    fused_boxes = [{"box": r.box, "color": PAL["fused"] if r.id == "R01" else CLASS_COLOR.get(r.label, PAL["mute"]),
                    "tag": r.id, "label": f"{r.id} · {r.label} · {r.n_raters} bác sĩ",
                    "width": 1.7 if r.id == "R01" else 1.3} for r in fr]
    # f-string expressions cannot carry brace literals on every supported Python,
    # so every box list the document draws is named here first.
    target_boxes = [{"box": d["r01"].box, "color": PAL["fused"], "tag": "R01",
                     "label": "R01 — vùng đồng thuận", "width": 2.0},
                    {"box": sample.gt_box, "color": PAL["gt"], "dash": "4 2",
                     "label": "hộp giám sát (bbox_fixed)", "width": 1.8}]
    mask_r01 = [{"box": d["r01"].box, "mask": True}]
    mask_r03 = [{"box": fr[2].box, "mask": True}]
    answer_boxes = [{"box": d["r01"].box, "color": PAL["fused"], "tag": "hộp trả lời",
                     "label": "R01 — vùng pointer đầu tiên cùng lớp", "width": 2.0}]

    # --- derived illustrative quantities, computed here so the arithmetic is right
    rel = [sigmoid(x) for x in I["relevance_logits"]]
    pl = I["pointer_logits"] + [I["pointer_null"]]
    pi_s = softmax(pl)
    logp = I["teacher_logp"]
    r_vec = [max(0.0, logp["orig"] - logp[f"R0{i}"]) for i in (1, 2, 3, 4)] + [0.0]
    pi_t1 = softmax(r_vec, 1.0)
    pi_t2 = softmax(r_vec, 0.2)
    conf = math.exp(logp["orig"])
    kl = sum(p * math.log(p / q) for p, q in zip(pi_t2, pi_s))
    drops = [max(0.0, math.log(sigmoid(I["faith_logit_full"])) - math.log(sigmoid(I["faith_logit_drop"][f"R0{i}"])))
             for i in (1, 2, 3, 4)] + [0.0]
    pi_cf = softmax(drops)
    kl_f = sum(p * math.log(p / q) for p, q in zip(pi_cf, pi_s))
    rids = [r.id for r in fr] + ["r∅"]

    facts_rows = "".join(
        f'<tr{" class=hint" if f.id in ("F7", "F8", "F9") else ""}>'
        f'<td class="mono">{f.id}</td><td>{e(f.attr)}</td>'
        f'<td>{e(d["fact_text"](f))}</td>'
        f'<td class="mono">{e(", ".join(f.evidence_ids))}</td></tr>'
        for f in d["facts"].facts)

    raw_rows = "".join(
        f'<tr><td>{e(b.class_name)}</td><td class="mono">{e(b.rad_id)}</td>'
        f'<td class="mono num">({vn(b.x_min, 0)}, {vn(b.y_min, 0)}, {vn(b.x_max, 0)}, {vn(b.y_max, 0)})</td></tr>'
        for b in rec.boxes_raw)

    fused_rows = "".join(
        f'<tr{" class=ok" if r.id in d["targets"] else ""}><td class="mono">{r.id}</td>'
        f'<td>{e(r.label)}</td><td class="num">{r.n_raters}</td>'
        f'<td class="mono num">({vn(r.box.x_min, 1)}, {vn(r.box.y_min, 1)}, {vn(r.box.x_max, 1)}, {vn(r.box.y_max, 1)})</td>'
        f'<td class="mono num">({vn(r.box.x_min * rec.fx, 1)}, {vn(r.box.y_min * rec.fy, 1)}, '
        f'{vn(r.box.x_max * rec.fx, 1)}, {vn(r.box.y_max * rec.fy, 1)})</td></tr>'
        for r in fr)

    return f"""
<div class="title-block">
  <div class="kicker">EvidenceKD-MedThink v4.1 · VinDr-CXR-VQA</div>
  <h1>Luồng pipeline, một câu hỏi xuyên suốt</h1>
  <p class="sub">Từ hộp của ba bác sĩ đến chuỗi nộp cho scorer — mỗi bước kèm ảnh thật,
  số thật, và tên module chạy ra nó.</p>
  <table class="meta">
    <tr><td>Bản ghi</td><td class="mono">{e(SID)}</td></tr>
    <tr><td>Câu hỏi</td><td><strong>{e(sample.question)}</strong> &nbsp;
        <span class="mono">type = {e(sample.type)}</span></td></tr>
    <tr><td>Tập</td><td class="mono">{e(d["split"])} &nbsp;·&nbsp; ảnh {vn(rec.width)} × {vn(rec.height)} px
        → PNG 224 × 224</td></tr>
    <tr><td>Dựng bằng</td><td class="mono">python scripts/make_pipeline_pdf.py</td></tr>
  </table>
</div>

<div class="why">
  <strong>Hai loại số trong tài liệu này.</strong> Mọi thứ thuộc <em>dữ liệu</em> — hộp bác sĩ,
  kết quả gộp, IoU, nhãn, bảng dữ kiện, quy đổi toạ độ, chuỗi <span class="mono">&lt;loc&gt;</span> —
  do chính code của repo tính ra khi dựng file này; sửa code thì số ở đây đổi theo. Mọi thứ thuộc
  <em>đầu ra mạng nơ-ron</em> — logit, xác suất, log-prob của teacher — là
  <strong>minh hoạ</strong>: đúng kích thước, đúng công thức, nhưng chưa có lần huấn luyện nào hoàn tất,
  nên không con số nào trong các khối đó được trích dẫn như kết quả. Các khối ấy đều mang nhãn
  <span class="mark">MINH HOẠ</span>.
</div>

<h2>1. Mười bước, và thứ gì rơi ra sau mỗi bước</h2>
<p>Thiết kế có một nguyên tắc: <strong>nhánh quyết định sinh ra câu trả lời, generator chỉ giải
thích nó</strong>. Generator không sửa được đáp án (Answer Lock), mỗi mệnh đề nó viết phải trích
một mã vùng có thật, và một verifier riêng chấm từng mệnh đề — nên lời giải thích không thể trôi
khỏi bằng chứng.</p>
{flow_svg()}
<p class="note">Thứ tự trên đúng bằng thứ tự các cell trong
<span class="mono">notebooks/kaggle_full_pipeline.ipynb</span>. Giai đoạn nào thiếu đầu vào thì bị bỏ
qua kèm thông báo — trừ KD: nếu toàn bộ shard KD-R hỏng thì phiên dừng hẳn, vì nếu không stage 2 sẽ
âm thầm tụt về checkpoint stage 1 và cả phiên vẫn kết thúc exit 0 như thể đã chạy.</p>

<h2>2. Bước 1 — ảnh và hộp của ba bác sĩ</h2>
<div class="two">
  <div>
    {xray(iid, rec.width, rec.height, raw_boxes,
          caption="Sáu hộp thô: ba bác sĩ R8, R9, R10 đọc độc lập. Ba hộp Pulmonary fibrosis ở "
                  "đỉnh phổi gần trùng nhau — đó là thứ sẽ gộp thành R01.")}
  </div>
  <div>
    <table class="tbl"><tr><th>Lớp</th><th>Bác sĩ</th><th>Hộp (pixel gốc)</th></tr>{raw_rows}</table>
    <p class="note">Hộp công bố của câu hỏi là
    <span class="mono">({vn(sample.gt_box.x_min, 0)}, {vn(sample.gt_box.y_min, 0)},
    {vn(sample.gt_box.x_max, 0)}, {vn(sample.gt_box.y_max, 0)})</span> — chính là hộp của R8.
    Toạ độ nội bộ <strong>luôn là pixel DICOM gốc</strong>; chỉ quy đổi ở ranh giới
    (224 px cho model, norm1000 cho chuỗi <span class="mono">&lt;loc&gt;</span>).</p>
  </div>
</div>

<h2>3. Bước 3 — gộp hộp thành vùng ứng viên</h2>
<p><span class="mono">data/boxes.fuse_boxes</span> gộp ở IoU ≥ 0,3 (đúng tham số VinDR dùng để sửa
hộp), lấy trung bình toạ độ, rồi <span class="mono">regions_from_annotation</span> xếp theo
(số bác sĩ ↓, diện tích ↓) và cắt ở 8 vùng. Thứ tự đó là lý do <strong>R01 luôn là tổn thương được
đồng thuận nhất</strong> — cũng chính là vùng KD-R che trước tiên.</p>
<div class="two">
  <div>{xray(iid, rec.width, rec.height, fused_boxes,
             caption="Bốn vùng ứng viên sau khi gộp. R01 (đậm) gộp từ ba hộp của R8/R9/R10; "
                     "R02–R04 mỗi vùng chỉ một bác sĩ thấy.")}</div>
  <div>
    <table class="tbl">
      <tr><th>Vùng</th><th>Lớp</th><th class="num">bác sĩ</th><th>Hộp pixel gốc</th><th>Hộp 224 px</th></tr>
      {fused_rows}
      <tr><td class="mono">A_thorax</td><td>giải phẫu</td><td class="num">—</td>
          <td class="mono num">(0, 0, {vn(rec.width)}, {vn(rec.height)})</td>
          <td class="mono num">(0, 0, 224, 224)</td></tr>
    </table>
    <p class="note"><span class="mono">fx = 224/{vn(rec.width)} = {vn(rec.fx, 4)}</span> ·
    <span class="mono">fy = 224/{vn(rec.height)} = {vn(rec.fy, 4)}</span> — ảnh gốc không vuông nên
    hai trục co khác nhau. <span class="mono">A_thorax</span> là giải phẫu fallback (cả ảnh) vì
    torchxrayvision không có mặt; có nó thì sẽ là hộp phổi và hộp tim.</p>
  </div>
</div>

<h2>4. Vùng đích — câu hỏi này được giám sát bởi vùng nào</h2>
<p><span class="mono">target_region_ids</span> giữ vùng <em>cùng lớp</em> có IoU ≥ 0,5 với hộp
giám sát. Đây là tập <span class="mono">G</span> của pointer loss và là tập dương của relevance head.</p>
<div class="two">
  <div>{xray(iid, rec.width, rec.height, target_boxes,
             caption="Hai hộp gần trùng — nhưng không trùng hẳn: R01 là trung bình của ba bác sĩ, "
                     "hộp giám sát là hộp của một người.")}</div>
  <div>
<pre class="code">IoU(R01, hộp giám sát)
   giao x = {vn(d["inter_w"], 1)}        giao y = {vn(d["inter_h"], 1)}
   giao   = {vn(d["inter"])}
   |gt|   = {vn(d["gt_area"])}       |R01| = {vn(d["r01_area"])}
   IoU    = {vn(d["inter"])} / ({vn(d["gt_area"])} + {vn(d["r01_area"])} − {vn(d["inter"])})
          = {vn(d["iou_r01"], 4)}   ≥ 0,5  ✓

   R04 (cùng lớp, thuỳ dưới): IoU = 0  ✗

→ target_regions = {e(tuple(d["targets"]))}</pre>
    <p class="note">R04 cũng là Pulmonary fibrosis nhưng ở vị trí khác, nên không vào tập đích —
    câu hỏi hỏi <em>chỗ này</em>, không phải <em>mọi chỗ có xơ phổi</em>.</p>
  </div>
</div>

<h2>5. Nhãn huấn luyện — tính bằng hình học, không đọc từ câu chữ</h2>
<p><span class="mono">data/qa_labels.labels_for</span> ở chế độ <span class="mono">qa_full</span>
suy ra bên và số lượng từ <em>hộp</em>, không parse câu trả lời. Đó là điều kiện để head bên/đếm
học được thứ có thật trên ảnh.</p>
<pre class="code">labels_train.jsonl   {e(SID)}

   question_type  = {d["labels"].question_type} (Where)            yes_no = {d["labels"].yes_no}
   findings       = {e(list(d["labels"].findings))}
                    lớp 1 Pulmonary fibrosis · 4 Aortic enlargement · 11 ILD đều có hộp
   side           = {d["labels"].side} (bilateral)
                    tâm R01 x = {vn((d["r01"].box.x_min + d["r01"].box.x_max) / 2)} &gt; đường giữa {vn(rec.width / 2)} → "left"   (ảnh-phải = bệnh nhân-trái)
                    tâm R04 x = {vn((fr[3].box.x_min + fr[3].box.x_max) / 2)} &lt; {vn(rec.width / 2)}            → "right"     ⇒ hai bên
   count          = {d["labels"].count}                       asked_class = {d["labels"].asked_class}
   target_regions = {e(list(d["labels"].target_regions))}</pre>

<h2>6. Bảng dữ kiện — thứ duy nhất generator được phép nói</h2>
<p><span class="mono">facts/build.build_facts</span> dựng bảng từ vùng + giải phẫu + đầu ra head.
Mỗi mệnh đề về sau phải trích <em>một fact trong bảng này</em> và <em>một mã vùng của fact đó</em>;
verifier chấm đúng theo đó.</p>
<table class="tbl">
<tr><th>Fact</th><th>Thuộc tính</th><th>Giá trị</th><th>Vùng dẫn chứng</th></tr>
{facts_rows}
</table>
<p class="note">Không có <span class="mono">zone</span> / <span class="mono">extent</span> vì giải
phẫu fallback không có hộp phổi; không có <span class="mono">ctr</span> vì không biết tư thế chụp —
tính CTR mà không biết AP hay PA là bịa. <span class="mono">F13 absence</span> xuất hiện vì điểm
toàn ảnh của Pneumothorax ({vn(I["pneumothorax_score"], 2)}) dưới ngưỡng vắng mặt 0,3.</p>

<div class="pb"></div>
<h2>7. Bước 5 — decision branch <span class="mark">MINH HOẠ từ đây</span></h2>
<p>Một câu hỏi vào model dưới dạng: ảnh 224 px, danh sách hộp vùng ở 224 px, mã lớp/loại vùng,
token câu hỏi. Backbone của detector GĐ 0 <strong>đóng băng</strong>; chỉ phần phía trên học.</p>
<pre class="code nowrap">ảnh        [3, 224, 224] ──transform──▶ [3, 512, 512]        (tỉ lệ 512/224 = 2,286)
FPN        '0' [256,128,128]   '1' [256,64,64]   '2' [256,32,32]   '3' [256,16,16]
hộp R01 ở 512 px = ({vn(d["r01"].box.x_min * rec.fx * 512 / 224, 1)}  {vn(d["r01"].box.y_min * rec.fy * 512 / 224, 1)}  {vn(d["r01"].box.x_max * rec.fx * 512 / 224, 1)}  {vn(d["r01"].box.y_max * rec.fy * 512 / 224, 1)})
ROIAlign   mỗi vùng: hộp gốc + hộp mở rộng ×1,5 → 2 × [256, 7, 7] ──Linear──▶ 2 × 256 ──merge──▶ 256
token vùng = roi + emb_loại(finding) + emb_lớp(PF) + Linear(toạ độ chuẩn hoá)   → [256]
tokens     [1 + {len(regions.regions)}, 256]        (IMG, {e(", ".join(r.id for r in regions.regions))})
câu hỏi    "{e(sample.question.lower().rstrip("?"))}" → [CLS, where, is, the, pulmonary, fibrosis] → [6, 256]
fusion ×2  vùng ↔ câu hỏi (cross-attention 8 head);   evidence encoder ×2 trên [6 + 6, 256]</pre>

<h3>Relevance — vùng nào trả lời câu hỏi</h3>
{bars(rel, [r.id for r in fr],
      colors=[PAL["fused"] if r.id in d["targets"] else PAL["accent"] for r in fr],
      fmt=lambda v: vn(v, 2))}
<pre class="code">L_rel = BCE(σ(logits), [1, 0, 0, 0]) = {vn(sum(-math.log(rel[0]) if i == 0 else -math.log(1 - rel[i]) for i in range(4)) / 4, 3)}</pre>

<h3>Pointer — vùng nào được trích làm bằng chứng</h3>
{bars(pi_s, rids, colors=[PAL["fused"]] + [PAL["accent"]] * 3 + [PAL["mute"]],
      fmt=lambda v: vn(v, 3))}
<pre class="code">π^S = softmax([{", ".join(vn(x, 1) for x in pl)}])
L_pointer = −log Σ_(G={{R01}}) π = −log {vn(pi_s[0], 3)} = {vn(-math.log(pi_s[0]), 3)}

Evidence Pointer lúc suy luận: top_k = 3, min_p = 0,05, phải &gt; slot null {vn(pi_s[-1], 3)}
   → region_ids = [R01 ({vn(pi_s[0], 3)}), R04 ({vn(pi_s[3], 3)}), R03 ({vn(pi_s[2], 3)})]   status = SUPPORTED</pre>

<h2>8. Bước 6 — KD-R: che vùng và nhìn teacher mất tự tin</h2>
<p>Đây là toàn bộ ý tưởng của KD-R, và nó là một bức ảnh chứ không phải một đoạn văn. Teacher
(MedGemma + adapter VinDr, đóng băng) được hỏi cùng một câu trên ảnh gốc và trên từng ảnh đã
<strong>tô đen một vùng</strong>. Vùng nào bị che mà làm teacher mất tự tin nhiều nhất thì vùng đó
mang bằng chứng.</p>
<div class="panes">
  <div class="pane">{xray(iid, rec.width, rec.height, [], size=150, legend=False, title="ảnh gốc")}
    <div class="pane-n">log p_T(a*) = {vn(logp["orig"], 2)}</div></div>
  <div class="pane">{xray(iid, rec.width, rec.height,
      mask_r01, size=150, legend=False, title="che R01")}
    <div class="pane-n">{vn(logp["R01"], 2)} → r₁ = {vn(r_vec[0], 2)}</div>
    <div class="pane-d">sụt nhiều nhất</div></div>
  <div class="pane">{xray(iid, rec.width, rec.height,
      mask_r03, size=150, legend=False, title="che R03")}
    <div class="pane-n">{vn(logp["R03"], 2)} → r₃ = {vn(r_vec[2], 2)}</div>
    <div class="pane-d">che xong teacher còn tự tin hơn → cắt về 0</div></div>
</div>
<p class="cap" style="max-width:none">Vùng tô đen đúng bằng phép
<span class="mono">constant_fill</span> của <span class="mono">teacher/medgemma.mask_region</span>:
hình chữ nhật đen trên PNG 224 px, <em>trước</em> processor, để nó sống sót qua bước resize.
<span class="mark">MINH HOẠ</span> — các log-prob là ví dụ.</p>

<h3>Từ sụt log-prob thành phân phối đích</h3>
<div class="two">
  <div>
    <div class="pane-h">τ_T = 1,0 — mặc định của configs/kd_r.yaml</div>
    {bars(pi_t1, rids, colors=[PAL["fused"]] + [PAL["accent"]] * 3 + [PAL["mute"]],
          rule=0.5, rule_label="δ = 0,5", fmt=lambda v: vn(v, 3))}
    <pre class="code">khối lượng trên R01 = {vn(pi_t1[0], 3)} &lt; δ = 0,5
→ CỔNG KHÔNG QUA  (gate_reason = mass_below_delta)
→ w = 0 — mẫu này KHÔNG vào loss</pre>
  </div>
  <div>
    <div class="pane-h">τ_T = 0,2 — cùng dữ liệu, chỉ đổi nhiệt độ</div>
    {bars(pi_t2, rids, colors=[PAL["fused"]] + [PAL["accent"]] * 3 + [PAL["mute"]],
          rule=0.5, rule_label="δ = 0,5", fmt=lambda v: vn(v, 3))}
    <pre class="code">m = {vn(pi_t2[0], 3)} ≥ 0,5 ✓   teacher đúng ✓   c = exp({vn(logp["orig"], 2)}) = {vn(conf, 3)}
→ w = 1·1·c = {vn(conf, 3)}
   KL(π^T ‖ π^S) = {vn(kl, 3)}
   L_KD-R = w·KL / (w + ε) = {vn(kl, 3)}</pre>
  </div>
</div>
<div class="why"><strong>Vì sao r_∅ và τ_T phải đi kèm mọi kết quả KD-R.</strong> Hai cột trên khác
nhau <em>duy nhất</em> ở τ_T. Một bên loại mẫu khỏi loss, một bên giữ với trọng số {vn(conf, 3)}.
Hai siêu tham số này quyết định mẫu nào qua cổng, nên một con số KD-R không kèm chúng là không diễn
giải được.</div>

<h2>9. Bước 7 — faithfulness: che <em>token</em> thay vì che pixel</h2>
<p>KD-R hỏi teacher. Số hạng faithfulness hỏi <em>chính student</em>: bỏ token của từng vùng trong
evidence encoder, đo logit của finding được hỏi sụt bao nhiêu, rồi kéo pointer về phía vùng mà mạng
thật sự dựa vào — chứ không phải vùng nó <em>nên</em> chỉ.</p>
{bars(pi_cf, rids, colors=[PAL["fused"]] + [PAL["accent"]] * 3 + [PAL["mute"]], fmt=lambda v: vn(v, 3))}
<pre class="code">logit đầy đủ = {vn(I["faith_logit_full"], 1)}   bỏ R01 → {vn(I["faith_logit_drop"]["R01"], 2)} (sụt {vn(drops[0], 3)})   bỏ R02 → {vn(I["faith_logit_drop"]["R02"], 2)} ({vn(drops[1], 3)})
                        bỏ R03 → {vn(I["faith_logit_drop"]["R03"], 2)} ({vn(drops[2], 3)})   bỏ R04 → {vn(I["faith_logit_drop"]["R04"], 2)} ({vn(drops[3], 3)})
L_faith = KL(π^cf ‖ π^S) = {vn(kl_f, 3)}</pre>

<div class="pb"></div>
<h2>10. Bước 9 — suy luận: từ head ra chuỗi nộp cho scorer</h2>
<pre class="code">decide()        type Where; finding được hỏi = "{e(sample.gt_finding)}" (nêu thẳng trong câu hỏi)
                có mặt ({vn(I["finding_probs"]["Pulmonary fibrosis"], 2)} ≥ ngưỡng 0,5);  side = bilateral;  findings_present = [PF, AE, ILD]
locked answer   "{e(d["answer"])}"
                ← Answer Lock: generator KHÔNG sửa được chuỗi này
dữ kiện chọn    về PF → F7, F8, F9, F11, F12      trích vùng pointer R01/R04/R03 → thêm F1, F2, F3

mệnh đề   &lt;obs facts=F7&gt;  Pulmonary fibrosis is present. [R01] &lt;/obs&gt;                        → supported
          &lt;obs facts=F8&gt;  The finding is bilateral. [R01] &lt;/obs&gt;                             → supported
          &lt;obs facts=F9&gt;  There are 2 lesions. [R01] &lt;/obs&gt;                                  → supported
          &lt;obs facts=F11&gt; Aortic enlargement co-occurs with Pulmonary fibrosis. [R03] &lt;/obs&gt;  → supported</pre>

<h3>Hộp trả lời và quy đổi norm1000</h3>
<div class="two">
  <div>{xray(iid, rec.width, rec.height,
             answer_boxes,
             caption="Hộp trả lời = vùng pointer đầu tiên cùng lớp. Không có thì lấy vùng relevance "
                     "cao nhất cùng lớp; không có nữa thì KHÔNG phát &lt;loc&gt; — không bao giờ bịa hộp.")}</div>
  <div>{ruler(d)}
<pre class="code">chuỗi nộp:
"{e(d["answer"])} {e(d["loc_r01"])}.
 Reasoning: Pulmonary fibrosis is present. The finding is
 bilateral. There are 2 lesions. Aortic enlargement
 co-occurs with Pulmonary fibrosis."

hộp giám sát quy đổi: {e(d["loc_gt"])}
lệch vài đơn vị vì R01 là hộp đồng thuận ba bác sĩ,
còn gt_location là hộp của một người.</pre>
  </div>
</div>

<h2>11. Scorer chấm chuỗi đó</h2>
<table class="tbl">
<tr><th>Bảng</th><th>Cách chấm</th><th>Cho ảnh / câu này</th></tr>
<tr class="ok"><td class="mono">4</td><td>theo ảnh, 14 lớp: câu trả lời nêu lớp nào</td>
    <td>nêu PF, AE, ILD; hộp thô có đúng 3 lớp đó → TP 3 · FP 0 · FN 0 · TN 11</td></tr>
<tr class="ok"><td class="mono">5</td><td>hộp dự đoán vs hộp thô, IoU ≥ 0,3, match "any"</td>
    <td>IoU(R01, hộp R8) = {vn(d["iou_r01"], 3)} ≥ 0,3 → TP</td></tr>
<tr class="ok"><td class="mono">6</td><td>theo ảnh: IoU tốt nhất ≥ 0,5</td>
    <td>{vn(d["iou_r01"], 3)} ≥ 0,5 → ảnh được tính "định vị tốt"</td></tr>
<tr class="ok"><td class="mono">B</td><td>spec, theo câu hỏi: Grounding@0,5</td>
    <td>IoU(hộp trả lời, gt sửa) = {vn(d["iou_r01"], 3)} → Grounding = 1 · Answer&amp;grounding = 1</td></tr>
<tr class="hint"><td class="mono">C</td><td>spec: mệnh đề được fact đỡ</td>
    <td>4 observation, 4 supported → supported-claim P = 1,0; 3/4 nêu PF → relevance 0,75</td></tr>
</table>

<h3>Self-test oracle — toàn tập val, chạy thật, không GPU</h3>
<p>Thay head bằng annotation rồi chạy đúng đường ống trên cho cả 659 ảnh val. Nếu kết quả không gần
hoàn hảo thì <strong>cầu nối hỏng</strong>, chứ không phải model.</p>
<table class="tbl">
<tr><th>Nguồn</th><th>Chỉ số</th><th class="num">Giá trị</th></tr>
<tr><td class="mono">eval_preds_val_oracle.json</td><td>VQA precision / recall / F1 (Bảng 4)</td><td class="num mono">1,0 / 0,9943 / 0,9972</td></tr>
<tr><td class="mono">eval_preds_val_oracle.json</td><td>ảnh có IoU ≥ 0,5 (Bảng 6)</td><td class="num mono">99,85 %</td></tr>
<tr><td class="mono">spec_pipeline_val_oracle.json</td><td>Grounding@0,5 / mIoU (Bảng B)</td><td class="num mono">0,9674 / 0,9049</td></tr>
<tr><td class="mono">spec_pipeline_val_oracle.json</td><td>supported-claim P / citation P (Bảng C)</td><td class="num mono">1,0 / 1,0</td></tr>
</table>

<h2>12. Huấn luyện và suy luận khác nhau ở đâu</h2>
<table class="tbl">
<tr><th></th><th>Huấn luyện</th><th>Suy luận</th></tr>
<tr><td>Nguồn vùng</td><td class="mono">regions_train.json</td><td class="mono">regions_{{val,test}}.json</td></tr>
<tr><td></td><td>từ <span class="mono">boxes_fused</span> — hộp bác sĩ</td><td>từ <strong>detector</strong></td></tr>
<tr><td>Hộp giám sát</td><td class="mono">bbox_fixed_orig</td><td>không dùng</td></tr>
<tr><td>Nhãn</td><td class="mono">labels_train.jsonl + negatives.jsonl</td><td>không có</td></tr>
<tr><td>Teacher</td><td>đọc cache KD-R/KD-C từ đĩa (không nạp MedGemma lúc train)</td><td>không dùng</td></tr>
<tr><td>Ngưỡng</td><td>chọn trên <span class="mono">tune</span> = val ∖ test (359 ảnh)</td><td>đọc <span class="mono">thresholds.json</span></td></tr>
</table>

<div class="two">
<div class="src"><div class="src-h">Huấn luyện</div>
<pre class="code">python -m evidencekd.models.detector train \\
   --config configs/detector.yaml --resume auto
python -m evidencekd.data.build_regions \\
   --config configs/detector.yaml
python -m evidencekd.models.decision_branch train \\
   --config configs/decision.yaml --stage 1
python -m evidencekd.teacher.kd_r_targets \\
   --config configs/teacher.yaml --shard 0 --n-shards 30
python -m evidencekd.teacher.kd_r_targets \\
   --config configs/teacher.yaml --gate
python -m evidencekd.models.decision_branch train \\
   --config configs/decision.yaml --stage 2 \\
   --init decision_stage1.pt
python -m evidencekd.models.verifier train \\
   --config configs/decision.yaml
python -m evidencekd.models.generator_s train \\
   --config configs/generator.yaml</pre></div>
<div class="src"><div class="src-h">Suy luận và chấm</div>
<pre class="code">python -m evidencekd.models.detector predict \\
   --config configs/detector.yaml --split test \\
   --checkpoint outputs/detector/detector.pt
python -m evidencekd.models.decision_branch predict \\
   --config configs/decision.yaml --split test \\
   --checkpoint outputs/decision/decision_stage2.pt
python -m evidencekd.inference.pipeline \\
   --config configs/decision.yaml --tune
python -m evidencekd.inference.pipeline \\
   --config configs/decision.yaml --split test --generator s
python -m evidencekd.eval.tables_spec \\
   --config configs/decision.yaml --split test
python -m evidencekd.eval.vindr_tables \\
   --config configs/decision.yaml --split test
python -m evidencekd.eval.faithfulness \\
   --config configs/decision.yaml --split test</pre></div>
</div>

<h2>13. Chạy trên Kaggle</h2>
<table class="tbl">
<tr><th>Giai đoạn</th><th>Chạy gì</th><th>Input phải attach</th></tr>
<tr><td class="mono">0</td><td>detector → detections → B7 Bảng 4–6</td><td>code zip, ảnh, trọng số COCO</td></tr>
<tr><td class="mono">derived</td><td>vùng, giải phẫu, mẫu âm, nhãn; self-test oracle</td><td>—</td></tr>
<tr><td class="mono">1</td><td>decision branch → ngưỡng trên val ∖ test</td><td>—</td></tr>
<tr class="warn"><td class="mono">teacher</td><td>KD-R + KD-C, chia shard, có ngân sách thời gian</td>
    <td><strong>MedGemma</strong> + <span class="mono">medgemma-lora.zip</span></td></tr>
<tr><td class="mono">2</td><td>decision branch + KD-R + faithfulness</td><td>cache KD-R</td></tr>
<tr><td class="mono">3</td><td>verifier học (cache KD-C), generator S</td><td>flan-T5</td></tr>
<tr><td class="mono">eval</td><td>Proposed-S → Bảng A–C/F, Bảng 4–6, can thiệp faithfulness</td><td>—</td></tr>
</table>
<div class="why"><strong>MedGemma phải là checkpoint vision-language.</strong> Teacher nạp qua
<span class="mono">AutoModelForImageTextToText</span>. Notebook chỉ nhận thư mục có
<span class="mono">vision_config</span> trong <span class="mono">config.json</span>; một flan-T5
(attach cho generator S) bị từ chối ngay ở cell setup thay vì làm hỏng cả 30 shard KD-R ở cell 9.</div>

<p class="note">Dựng lại tài liệu này: <span class="mono">python scripts/make_pipeline_pdf.py</span>.
Mọi số thuộc dữ liệu được tính lại từ <span class="mono">data/processed</span> và
<span class="mono">src/evidencekd/</span> ở thời điểm chạy; ảnh X-quang nhúng base64 từ
<span class="mono">../VinDR-CXR-VQA/dataset</span>. Chưa có lần huấn luyện nào hoàn tất — mọi khối
mang nhãn MINH HOẠ không phải kết quả đo.</p>
"""


CSS = """
@page { size: A4; margin: 15mm 14mm 14mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI","Noto Sans",system-ui,sans-serif; color: #1f2933;
       font-size: 9.8pt; line-height: 1.5; margin: 0; -webkit-print-color-adjust: exact; }
h1 { font-size: 21pt; margin: 6px 0 4px; letter-spacing: -.4px; }
h2 { font-size: 13.5pt; margin: 19px 0 8px; padding-bottom: 5px;
     border-bottom: 2px solid #0b6bcb; color: #0b6bcb; }
h3 { font-size: 11pt; margin: 14px 0 6px; color: #334155; }
p { margin: 7px 0; }
.pb { break-before: page; }
h2, h3 { break-after: avoid; }
table, .fig, pre, .why, .panes, .two { break-inside: avoid; }
.kicker { font-size: 8.5pt; letter-spacing: 1.4px; text-transform: uppercase;
          color: #0b6bcb; font-weight: 700; }
.sub { color: #475569; font-size: 10.6pt; margin: 2px 0 12px; }
.title-block { border-bottom: 3px solid #0b6bcb; padding-bottom: 11px; margin-bottom: 14px; }
.meta { border-collapse: collapse; margin: 8px 0 0; font-size: 9.2pt; }
.meta td { padding: 2.5px 14px 2.5px 0; vertical-align: top; }
.meta td:first-child { color: #64748b; white-space: nowrap; }
.note { background: #f1f5f9; border-left: 3px solid #94a3b8; padding: 7px 10px;
        font-size: 8.8pt; color: #475569; margin: 9px 0; }
.why { background: #fff7ed; border-left: 3px solid #ea580c; padding: 8px 11px;
       font-size: 9.3pt; margin: 10px 0; }
.mark { background: #fde68a; color: #92400e; font-size: 7.4pt; font-weight: 700;
        padding: 1.5px 5px; border-radius: 3px; letter-spacing: .4px; vertical-align: 2px; }
.mono { font-family: ui-monospace,"Cascadia Mono",Consolas,monospace; font-size: .93em; }
.num { text-align: right; }
pre.code { font-family: ui-monospace,"Cascadia Mono",Consolas,monospace;
           font-size: 8.1pt; line-height: 1.45; background: #f8fafc;
           border: 1px solid #e2e8f0; border-radius: 4px; padding: 8px 10px;
           white-space: pre-wrap; margin: 7px 0; overflow-wrap: anywhere; }
pre.code.nowrap { white-space: pre; font-size: 7.4pt; }
.tbl { border-collapse: collapse; width: 100%; margin: 9px 0; font-size: 8.8pt; }
.tbl th { background: #0b6bcb; color: #fff; text-align: left; padding: 5px 7px;
          font-weight: 600; font-size: 8.4pt; }
.tbl td { padding: 3.5px 7px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
.tbl tr.ok td { background: #ecfdf5; }
.tbl tr.warn td { background: #fffbeb; }
.tbl tr.hint td { background: #eff6ff; }
.fig { margin: 10px 0; text-align: center; }
.fig svg { max-width: 100%; }
.cap { font-size: 8.4pt; color: #64748b; margin-top: 6px; text-align: left;
       max-width: 430px; margin-left: auto; margin-right: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 3px 12px; justify-content: center;
          margin-top: 6px; font-size: 8pt; color: #475569; }
.lg { display: flex; align-items: center; gap: 5px; }
.sw { width: 15px; height: 8px; border: 2px solid; display: inline-block; }
.two { display: flex; gap: 13px; margin: 10px 0; align-items: flex-start; }
.two > * { flex: 1; min-width: 0; }
.panes { display: flex; gap: 14px; justify-content: center; align-items: flex-start; margin: 10px 0; }
.pane { text-align: center; }
.pane-h { font-size: 8.8pt; font-weight: 700; margin-bottom: 4px; color: #334155; }
.pane-d { font-size: 7.6pt; color: #94a3b8; margin-top: 3px; max-width: 160px; }
.pane-n { font-family: ui-monospace,Consolas,monospace; font-size: 7.8pt;
          color: #0b6bcb; margin-top: 3px; }
.src { border: 1px solid #e2e8f0; border-radius: 4px; overflow: hidden; }
.src-h { background: #f1f5f9; padding: 5px 9px; font-size: 8.6pt; font-weight: 700;
         color: #334155; border-bottom: 1px solid #e2e8f0; }
.src pre.code { border: 0; border-radius: 0; margin: 0; background: #fff; font-size: 7.5pt; }
"""


def build_html(d) -> str:
    return (f'<!doctype html><html lang="vi"><head><meta charset="utf-8">'
            f'<title>EvidenceKD-MedVQA — luồng pipeline, một câu hỏi xuyên suốt</title>'
            f"<style>{CSS}</style></head><body>{build_body(d)}</body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "PIPELINE_VISUAL.pdf"))
    ap.add_argument("--keep-html", action="store_true",
                    help="leave the intermediate .html next to the PDF")
    args = ap.parse_args()

    os.chdir(REPO)
    data = collect()
    print(f"[pdf] record {SID} · IoU(R01, gt) = {data['iou_r01']:.4f} · "
          f"{len(data['facts'].facts)} facts · {data['loc_r01']}")

    doc = build_html(data)
    html_path = os.path.splitext(args.out)[0] + ".html"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(html_path, "w", encoding="utf-8").write(doc)
    print(f"[pdf] html {len(doc) / 1024:.0f} KB (ảnh X-quang nhúng base64)")

    browser = find_browser()
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
                    f"--print-to-pdf={args.out}",
                    "file:///" + html_path.replace("\\", "/")],
                   check=False, capture_output=True)
    if not args.keep_html and os.path.exists(html_path):
        os.remove(html_path)
    if not os.path.exists(args.out):
        raise SystemExit(f"[pdf] {browser} produced no file")
    print(f"[pdf] wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
