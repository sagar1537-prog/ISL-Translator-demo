#!/usr/bin/env python3
"""ISL Counter - web app for the ISL sign translator.

How it works
    browser   webcam -> MediaPipe hand tracking (runs in the page, nothing is
              uploaded) -> 21 hand landmarks per frame
    server    landmarks -> the same 63 features as the desktop app -> trained
              classifier (NumPy only) -> word + confidence
    browser   voting over recent frames (a word must win 10 of the last 15
              predictions at >= 70% confidence) -> display + speech

Run:     python app.py            then open http://127.0.0.1:5000
Deploy:  gunicorn app:app        (see README.md)

Settings (environment variables, all optional)
    ISL_MODEL       path to the model file          (default: model/isl_model.npz)
    ISL_THRESHOLD   minimum confidence per frame    (default: 0.70)
    ISL_BUFFER      predictions in the voting window (default: 15)
    ISL_VOTES       votes needed to accept a word   (default: 10)
    PORT            port to listen on               (default: 5000)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

from engine import load_npz
from features import NONE_LABEL, NUM_LANDMARKS, canonical_features, legacy_features

HERE = Path(__file__).resolve().parent

MODEL_PATH = Path(os.environ.get("ISL_MODEL", HERE / "model" / "isl_model.npz"))
THRESHOLD = float(os.environ.get("ISL_THRESHOLD", "0.70"))
VOTE_BUFFER = int(os.environ.get("ISL_BUFFER", "15"))
VOTES_NEEDED = int(os.environ.get("ISL_VOTES", "10"))

# How each label is shown and spoken. Labels not listed are shown as-is.
DISPLAY_NAMES = {
    "thankyou": "Thank you",
    "trainticket": "Train ticket",
    "trainstation": "Train station",
    "storeorshop": "Store / shop",
}

model = load_npz(MODEL_PATH)
WINDOW = model.window  # None for a single-frame model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024  # a request is a few KB; refuse anything big
app.json.sort_keys = False


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label.replace("_", " ").capitalize())


class BadRequest(ValueError):
    pass


def _points(value) -> np.ndarray:
    try:
        pts = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        raise BadRequest("points must be numbers") from None
    if pts.shape != (NUM_LANDMARKS, 3) or not np.isfinite(pts).all():
        raise BadRequest(f"points must be {NUM_LANDMARKS} finite [x, y, z] triples")
    return pts


def _features(pts: np.ndarray, is_left: bool, size: tuple[int, int]) -> np.ndarray:
    cfg = model.feature
    if int(cfg.get("version", 1)) < 2:
        return legacy_features(pts)
    return canonical_features(pts, image_size=size if cfg.get("aspect_correct", True) else None,
                              is_left=is_left, mirror_left=cfg.get("mirror_left", True),
                              scale=cfg.get("scale", True))


@app.get("/")
def index():
    words = [display_name(l) for l in model.labels if l != NONE_LABEL]
    return render_template("index.html", words=words)


@app.get("/api/config")
def config():
    return jsonify({
        "labels": model.labels,
        "display": {l: display_name(l) for l in model.labels},
        "none_label": NONE_LABEL,
        "threshold": THRESHOLD,
        "buffer": VOTE_BUFFER,
        "votes": VOTES_NEEDED,
        "window": WINDOW,
        "model": {"type": model.arch["type"],
                  "accuracy": model.metrics.get("take_accuracy")},
    })


@app.post("/api/predict")
def predict():
    """Body: {"points": [[x, y, z] x 21], "is_left": bool, "width": int, "height": int}
    For a window (GRU) model send "sequence": [points, points, ...] instead of "points".
    Coordinates are MediaPipe's normalized values for the un-mirrored camera frame."""
    t0 = time.perf_counter()
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON object."), 400
    try:
        width, height = int(body.get("width", 0)), int(body.get("height", 0))
        size = (width, height) if width > 0 and height > 0 else (1, 1)
        is_left = bool(body.get("is_left", False))
        if WINDOW:
            seq = body.get("sequence")
            if not isinstance(seq, list) or len(seq) != WINDOW:
                raise BadRequest(f"this model needs a 'sequence' of {WINDOW} frames")
            x = np.stack([_features(_points(p), is_left, size) for p in seq])[None]
        else:
            x = _features(_points(body.get("points")), is_left, size)[None]
    except (BadRequest, TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400

    probs = model.probs(x)[0]
    order = np.argsort(-probs)[:3]
    best = int(order[0])
    return jsonify({
        "label": model.labels[best],
        "confidence": round(float(probs[best]), 4),
        "accepted": bool(probs[best] >= THRESHOLD and model.labels[best] != NONE_LABEL),
        "top": [{"label": model.labels[i], "confidence": round(float(probs[i]), 4)} for i in order],
        "server_ms": round((time.perf_counter() - t0) * 1000, 2),
    })


@app.get("/healthz")
def health():
    return jsonify(ok=True, words=len(model.labels))


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    if request.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"Model: {MODEL_PATH.name}  |  {len(model.labels)} words  |  threshold {THRESHOLD:.2f}")
    print(f"Open http://127.0.0.1:{port} in Chrome or Edge")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, threaded=True)
