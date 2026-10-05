#!/usr/bin/env python3
"""ISL Translator Mark III - web app (sign -> English, English -> sign, Teach mode).

Everything per frame (hand + body tracking, recognition, sentence building) runs
in the browser. The server only serves the page and the model, and optionally
asks a local LLM (Ollama) to polish sentences and the summary.

    python app.py          then open http://127.0.0.1:8000

Optional settings (environment variables)
    PORT / HOST            where to listen (default 8000 / 127.0.0.1)
    ISL_THRESHOLD          confidence needed for a word (default 0.70)
    ISL_LLM_MODEL          an Ollama model name, e.g. llama3.2:3b, to polish the English
    OLLAMA_URL             default http://127.0.0.1:11434
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template, request

HERE = Path(__file__).resolve().parent
MODEL_JSON = HERE / "static" / "model" / "model.json"
POSE_TASK = HERE / "static" / "models" / "pose_landmarker_lite.task"
PERSONAL = HERE / "personal"
FEATURE_DIM = 143
LLM_MODEL = os.environ.get("ISL_LLM_MODEL", "").strip()
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


def model_info() -> dict:
    if not MODEL_JSON.is_file():
        return {"ready": False}
    try:
        with open(MODEL_JSON, encoding="utf-8") as f:
            spec = json.load(f)
        return {"ready": True, "words": len(spec["labels"]) - 1, "metrics": spec.get("metrics", {})}
    except (OSError, ValueError, KeyError):
        return {"ready": False}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/config")
def config():
    return jsonify({
        "model": model_info(),
        "pose": POSE_TASK.is_file(),
        "threshold": float(os.environ.get("ISL_THRESHOLD", "0.75")),
        "margin": float(os.environ.get("ISL_MARGIN", "0.25")),
        "personal": len(list(PERSONAL.glob("*.json"))) if PERSONAL.is_dir() else 0,
        "llm": bool(LLM_MODEL),
    })


@app.post("/api/polish")
def polish():
    """Optional: rewrite the rule-based English and summary with a local LLM."""
    if not LLM_MODEL:
        return jsonify(error="no local LLM configured"), 404
    body = request.get_json(silent=True) or {}
    gloss = [str(g)[:40] for g in body.get("gloss", [])][:40]
    draft = str(body.get("english", ""))[:600]
    history = [str(h)[:300] for h in body.get("history", [])][-12:]
    prompt = (
        "You turn Indian Sign Language glosses into natural English for a deaf signer talking to a "
        "counter clerk. Keep the meaning, do not invent facts. Reply as JSON with keys "
        '"sentence" (one or two short sentences, first person as the signer) and "summary" '
        "(one or two sentences, third person, what the person is telling the clerk overall).\n"
        f"Signs in order: {' '.join(gloss)}\nRule-based draft: {draft}\n"
        f"Earlier in the conversation: {' | '.join(history) or 'nothing'}\n")
    req = urllib.request.Request(f"{OLLAMA}/api/generate", method="POST",
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": LLM_MODEL, "prompt": prompt, "stream": False,
                                                  "format": "json", "options": {"temperature": 0.2}}).encode())
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            out = json.loads(json.loads(r.read().decode())["response"])
        return jsonify(sentence=str(out.get("sentence", ""))[:400], summary=str(out.get("summary", ""))[:600])
    except Exception as e:  # LLM not running / bad output: the page keeps the rule-based text
        return jsonify(error=f"local LLM unavailable: {e}"), 503


# --------------------------------------------------------------------------- Teach mode recordings
def _clean_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())[:40]


def _finite_rows(rows, width=None, max_rows=400) -> bool:
    if not isinstance(rows, list) or not (1 <= len(rows) <= max_rows):
        return False
    for r in rows:
        if not isinstance(r, list) or (width and len(r) != width):
            return False
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in r):
            return False
    return True


@app.get("/api/personal")
def personal_list():
    items = []
    for f in sorted(PERSONAL.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            items.append({"id": f.stem, "label": d["label"], "display": d.get("display", d["label"]),
                          "frames": len(d.get("features", [])), "created": d.get("created", 0)})
        except Exception:
            continue
    return jsonify(items=items)


@app.post("/api/personal")
def personal_save():
    d = request.get_json(silent=True) or {}
    label = _clean_label(d.get("label", ""))
    display = str(d.get("display") or d.get("label") or "").strip()[:40]
    if not label or label == "none":
        return jsonify(error="Give the sign a name (letters and numbers)."), 400
    if not _finite_rows(d.get("features"), FEATURE_DIM):
        return jsonify(error="The recording is empty or damaged. Record it again."), 400
    raw = d.get("raw") or {}
    frames = raw.get("frames") if isinstance(raw, dict) else None
    if not isinstance(frames, list) or len(frames) > 400:
        frames = []
    PERSONAL.mkdir(exist_ok=True)
    rid = f"{label}__{int(time.time() * 1000)}"
    (PERSONAL / f"{rid}.json").write_text(json.dumps({
        "label": label, "display": display or label, "category": "your signs", "created": int(time.time()),
        "fps": float(d.get("fps") or 30), "features": d["features"],
        "raw": {"fps": float(raw.get("fps") or 30), "frames": frames}}), encoding="utf-8")
    return jsonify(ok=True, id=rid)


@app.delete("/api/personal/<rid>")
def personal_delete(rid: str):
    if not re.fullmatch(r"[a-z0-9]+__\d+", rid):
        return jsonify(error="bad id"), 400
    f = PERSONAL / f"{rid}.json"
    if f.is_file():
        f.unlink()
    return jsonify(ok=True)


@app.get("/api/personal/sign/<label>")
def personal_sign(label: str):
    """A recorded take of a word, in the avatar's format (for words not trained into the library yet)."""
    label = _clean_label(label)
    for f in sorted(PERSONAL.glob(f"{label}__*.json"), reverse=True):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("raw", {}).get("frames"):
                return jsonify(label=label, display=d.get("display", label), fps=d["raw"].get("fps", 30), frames=d["raw"]["frames"])
        except Exception:
            continue
    return jsonify(error="not found"), 404


@app.get("/healthz")
def health():
    return jsonify(ok=True, model=model_info().get("ready", False))


_gz: dict = {}
_lock = threading.Lock()


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    p = request.path
    if p.startswith("/static/"):
        stable = p.startswith(("/static/vendor/", "/static/fonts/", "/static/models/"))
        resp.headers["Cache-Control"] = "public, max-age=604800, immutable" if stable else "no-cache"
        if (resp.status_code == 200 and p.endswith((".js", ".mjs", ".css", ".wasm", ".json", ".task"))
                and "gzip" in request.headers.get("Accept-Encoding", "")):
            key = (p, resp.headers.get("ETag", ""), resp.headers.get("Last-Modified", ""))
            with _lock:
                data = _gz.get(key)
                if data is None:
                    resp.direct_passthrough = False
                    data = gzip.compress(resp.get_data(), 6)
                    _gz[key] = data
            resp.direct_passthrough = False
            resp.set_data(data)
            resp.headers["Content-Encoding"] = "gzip"
            resp.headers["Vary"] = "Accept-Encoding"
            resp.headers["Content-Length"] = str(len(data))
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    info = model_info()
    print("ISL Translator Mark III")
    print(f"  model: {'%d words' % info['words'] if info.get('ready') else 'NOT TRAINED YET - run train.bat first'}")
    print(f"  body tracking: {'on' if POSE_TASK.is_file() else 'off (run setup.bat to download it)'}")
    print(f"  local LLM: {LLM_MODEL or 'off (optional)'}")
    print(f"  open http://127.0.0.1:{port}")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, threaded=True)
