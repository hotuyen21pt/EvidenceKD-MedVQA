"""HTTP backend for the one-image UI — the same contract as VinDr's ``serve.py``
plus an ``evidence`` field (regions, facts, pointer, claims with status).

    python -m evidencekd.serve --mock                                  # no torch, canned answers
    python -m evidencekd.serve --detector outputs/detector/detector.pt \
        --decision outputs/decision/decision_stage2.pt [--generator-s outputs/generator_s/final] [--verifier outputs/verifier/verifier.pt]

Routes: GET /api/health · /api/meta · /api/samples · /api/image · POST /api/predict · /api/load.
Serves ``ui/dist`` when built (the sibling repo's React page works unchanged;
the ``evidence`` field is extra). CORS *, no auth — LAN only.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from evidencekd.data.boxes import Box
from evidencekd.data.vindr_processed import load_images, load_samples
from evidencekd.eval.vindr_bridge import to_norm1000
from evidencekd.labels import VQA_CLASSES
from evidencekd.utils.config import load_config

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rel(box, w, h):
    return [round(box[0] / w, 5), round(box[1] / h, 5), round(box[2] / w, 5), round(box[3] / h, 5)]


class Engine:
    def __init__(self, cfg, *, mock: bool, detector: str | None, decision: str | None, generator_s: str | None,
                 verifier: str | None, thresholds: str | None):
        self.cfg = cfg; self.mock = mock
        self.images = load_images(cfg.paths.processed_dir)
        self.samples = {sp: load_samples(cfg.paths.processed_dir, sp) for sp in ("val", "test", "train")}
        self.by_id = {s.sample_id: s for sp in self.samples.values() for s in sp}
        self.paths = {"detector": detector, "decision": decision, "generator_s": generator_s, "verifier": verifier, "thresholds": thresholds}
        self.runner = None; self.state = "mock" if mock else "cold"; self.error = None; self.lock = threading.Lock()

    def ensure(self):
        if self.mock or self.runner is not None:
            return
        with self.lock:
            if self.runner is not None:
                return
            self.state = "loading"
            try:
                from evidencekd.inference.pipeline import Runner, load_thresholds
                derived = self.cfg.paths.derived_dir
                self.runner = Runner(self.cfg, detector_ckpt=self.paths["detector"], decision_ckpt=self.paths["decision"],
                                     thresholds=load_thresholds(self.paths["thresholds"]),
                                     anatomy_cache=os.path.join(derived, "anatomy.json"),
                                     background_path=os.path.join(derived, "background.json"),
                                     generator="s" if self.paths["generator_s"] else "template",
                                     generator_dir=self.paths["generator_s"], verifier_path=self.paths["verifier"])
                self.state = "ready"
            except Exception as e:  # noqa: BLE001
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
                raise

    def health(self):
        return {"state": self.state, "mock": self.mock, "adapter": self.paths["decision"], "mode": "evidencekd",
                "model": self.paths["detector"], "error": self.error, "images_indexed": len(self.images),
                "splits": {k: len(v) for k, v in self.samples.items()}, "dataset_dir": self.cfg.paths.dataset_dir,
                "dataset_dir_exists": os.path.isdir(self.cfg.paths.dataset_dir)}

    def meta(self):
        return {"pathologies": list(VQA_CLASSES), "question_types": ["Where", "What", "Is_there", "How_many", "Which", "Yes_No"],
                "loc_format": "norm1000", "pred_space": "norm1000", "box_source": "fixed", "iou_threshold": 0.3,
                "good_iou": 0.5, "max_new_tokens": 160}

    def predict(self, body: dict) -> dict:
        from PIL import Image
        t0 = time.time()
        rec = self.by_id.get(body.get("sample_id") or "")
        question = body.get("question") or (rec.question if rec else "")
        if rec is not None:
            img = Image.open(os.path.join(self.cfg.paths.dataset_dir, rec.image_path)).convert("RGB")
            W, H, iid = rec.image_width, rec.image_height, rec.image_id
        elif body.get("image_b64"):
            raw = body["image_b64"].split(",", 1)[-1]
            img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
            if max(img.size) > 224:
                img = img.resize((224, 224))
            W, H, iid = int(body.get("width") or img.width), int(body.get("height") or img.height), "upload"
        else:
            raise ValueError("sample_id or image_b64 required")
        if self.mock:
            out = self._mock(rec, question, W, H)
        else:
            self.ensure()
            out = self.runner.run(img, question, image_id=iid, orig_size=(W, H), qtype=rec.type if rec else None)
        gt = []
        if rec is not None:
            for f in self.images[rec.image_id].boxes_fused:
                gt.append({"class_name": f.box.class_name, "pixel": [f.box.x_min, f.box.y_min, f.box.x_max, f.box.y_max],
                           "rel": _rel([f.box.x_min, f.box.y_min, f.box.x_max, f.box.y_max], W, H)})
        pred_boxes = []
        if out.get("answer_box"):
            b = out["answer_box"]
            best = max((self._iou(b, g["pixel"]) for g in gt), default=None)
            pred_boxes.append({"raw": list(to_norm1000(Box(*b, class_name=""), W, H)), "pixel": b, "rel": _rel(b, W, H),
                               "best_iou": None if best is None else round(best, 4), "matched_gt": None, "matched_iou": None})
        return {"mock": self.mock, "prediction": out["bridge"], "answer": out["answer"],
                "reason": " ".join(c["text"] for c in out["claims"]), "findings": out["findings_present"],
                "pred_space": "norm1000", "pred_boxes": pred_boxes, "gt_boxes": gt,
                "best_iou": pred_boxes[0]["best_iou"] if pred_boxes else None, "latency_s": round(time.time() - t0, 3),
                "image_width": W, "image_height": H, "image_url": f"/api/image?image_id={iid}" if rec else None,
                "record": None if rec is None else {"sample_id": rec.sample_id, "gt_finding": rec.gt_finding,
                                                     "gt_answer": rec.answer, "gt_reason": rec.reason, "type": rec.type},
                "evidence": {k: out[k] for k in ("regions", "facts", "pointer", "claims", "scores", "warnings", "question_type")}}

    @staticmethod
    def _iou(a, b):
        return Box(*a, class_name="").area and __import__("evidencekd.data.boxes", fromlist=["iou"]).iou(Box(*a, class_name=""), Box(*b, class_name=""))

    def _mock(self, rec, question, W, H):
        from evidencekd.data.anatomy import fallback_anatomy
        from evidencekd.data.regions import regions_from_annotation
        from evidencekd.inference.pipeline import build_output
        if rec is None:
            return {"bridge": "MOCK: no record.", "answer": "MOCK", "claims": [], "findings_present": [], "answer_box": None,
                    "regions": [], "facts": [], "pointer": {"region_ids": [], "null_probability": 1.0, "status": "NO_SUPPORT"},
                    "scores": {}, "warnings": [], "question_type": "Is_there"}
        img = self.images[rec.image_id]
        rs = regions_from_annotation(img, anatomy=fallback_anatomy(W, H))
        dec = {"question_type": rec.type, "finding_probs": {c: (0.9 if c in img.vqa_classes_present else 0.05) for c in VQA_CLASSES},
               "yes_prob": 0.9, "side_probs": {"left": 0.4, "right": 0.4, "bilateral": 0.2}, "count_probs": [0.1, 0.6, 0.2, 0.05, 0.05],
               "relevance": [1.0 if r.label == rec.gt_finding else 0.2 for r in rs.finding_regions],
               "pointer": [0.8 if r.label == rec.gt_finding else 0.02 for r in rs.finding_regions] + [0.05]}
        return build_output(rec, question, rec.type, rs, fallback_anatomy(W, H), dec, {"finding": 0.5, "yes": 0.5, "absence": 0.3},
                            generator="template", background={})


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None  # type: ignore[assignment]
    ui_dir: str = ""

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type"); self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query); e = self.engine
        if u.path == "/api/health":
            return self._json(200, e.health())
        if u.path == "/api/meta":
            return self._json(200, e.meta())
        if u.path == "/api/samples":
            split = q.get("split", ["val"])[0]; text = q.get("q", [""])[0].lower(); typ = q.get("type", [""])[0]
            items = [s for s in e.samples.get(split, []) if (not text or text in s.question.lower() or text in s.image_id or text in s.gt_finding.lower())
                     and (not typ or s.type == typ)]
            off, lim = int(q.get("offset", ["0"])[0]), min(int(q.get("limit", ["60"])[0]), 500)
            return self._json(200, {"total": len(items), "offset": off, "items": [
                {"sample_id": s.sample_id, "image_id": s.image_id, "question": s.question, "type": s.type, "difficulty": s.difficulty,
                 "gt_finding": s.gt_finding, "box_label_consistent": s.box_label_consistent} for s in items[off:off + lim]]})
        if u.path == "/api/image":
            iid = q.get("image_id", [""])[0]; rec = e.images.get(iid)
            if rec is None:
                return self._json(404, {"error": "unknown image"})
            p = os.path.join(e.cfg.paths.dataset_dir, rec.image_path)
            with open(p, "rb") as f:
                data = f.read()
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Cache-Control", "max-age=3600")
            self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        # static UI
        path = u.path if u.path != "/" else "/index.html"
        fp = os.path.join(self.ui_dir, path.lstrip("/"))
        if self.ui_dir and os.path.isfile(fp):
            data = open(fp, "rb").read(); ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        if self.ui_dir and os.path.isfile(os.path.join(self.ui_dir, "index.html")):
            data = open(os.path.join(self.ui_dir, "index.html"), "rb").read()
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        self._json(404, {"error": "UI not built: cd ui && npm install && npm run build"})

    def do_POST(self):
        u = urlparse(self.path); e = self.engine
        n = int(self.headers.get("Content-Length", "0") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        try:
            if u.path == "/api/load":
                e.ensure(); return self._json(200, e.health())
            if u.path == "/api/predict":
                return self._json(200, e.predict(body))
        except Exception as ex:  # noqa: BLE001
            return self._json(500, {"error": f"{type(ex).__name__}: {ex}"})
        self._json(404, {"error": "unknown route"})

    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/decision.yaml")
    ap.add_argument("--detector", default=os.environ.get("EKD_DETECTOR"))
    ap.add_argument("--decision", default=os.environ.get("EKD_DECISION"))
    ap.add_argument("--generator-s", default=os.environ.get("EKD_GENERATOR_S"))
    ap.add_argument("--verifier", default=os.environ.get("EKD_VERIFIER"))
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--ui-dir", default=os.path.join(REPO, "ui", "dist"))
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--preload", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if not args.mock and not (args.detector and args.decision):
        ap.error("--detector and --decision are required unless --mock")
    Handler.engine = Engine(cfg, mock=args.mock, detector=args.detector, decision=args.decision, generator_s=args.generator_s,
                            verifier=args.verifier, thresholds=args.thresholds)
    Handler.ui_dir = args.ui_dir if os.path.isdir(args.ui_dir) else ""
    if args.preload:
        Handler.engine.ensure()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}  (mock={args.mock}, ui={'yes' if Handler.ui_dir else 'no'})")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
