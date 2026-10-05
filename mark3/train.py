#!/usr/bin/env python3
"""train.py - Mark III: train on the whole Kaggle ISL keypoint dataset (+ your own recorded signs).

    python train.py --data "C:\\Users\\VSJ\\Downloads\\archive\\keypoints"      full training
    python train.py --data "..." --finetune                                   quick update after
                                                                              recording your own signs

Reads keypoints/<category>/<word>/*.parquet (MediaPipe Holistic: frame, type, landmark_index,
x, y, z): both hands, plus nose, shoulders, elbows and wrists. Your own recordings from the
app's Teach mode (personal/*.json) are added and weighted up, and may add new words.

Writes
    static/model/model.json     the model for the browser (calibrated confidence)
    static/signs/<word>.json    one real signer's movement per word, for the signing avatar
    model/isl_mk3.pt            PyTorch checkpoint (used by --finetune)
    model/report.txt            accuracy, weakest words, most-confused pairs
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from features3 import (BODY_BLOCK, FEATURE_DIM, FEATURE_VERSION, HAND_BLOCK, L0, LSH, P0, POSE_IDS, R0, RSH,
                       clip_features, display_name, mirror, resample)

HERE = Path(__file__).resolve().parent
NONE = "none"
T = 24                                    # frames per window seen by the model
PERSONAL_WEIGHT = 8                       # your own takes count this many times more than a dataset video


def label_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# --------------------------------------------------------------------------- dataset
def read_video(path: Path):
    """-> right (n,21,3), left (n,21,3), pose (n,7,3); NaN where missing."""
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df = df[df["type"].isin(("right_hand", "left_hand", "pose"))]
    frames, inv = np.unique(df["frame"].to_numpy(), return_inverse=True)
    n = len(frames)
    hands = {t: np.full((n, 21, 3), np.nan, np.float32) for t in ("right_hand", "left_hand")}
    pose = np.full((n, len(POSE_IDS), 3), np.nan, np.float32)
    typ = df["type"].to_numpy()
    lm = df["landmark_index"].to_numpy().astype(int)
    xyz = df[["x", "y", "z"]].to_numpy(np.float32)
    for t in hands:
        m = (typ == t) & (lm >= 0) & (lm < 21)
        hands[t][inv[m], lm[m]] = xyz[m]
        bad = ~np.isfinite(hands[t]).all(axis=(1, 2))
        hands[t][bad] = np.nan
    for k, pid in enumerate(POSE_IDS):
        m = (typ == "pose") & (lm == pid)
        pose[inv[m], k] = xyz[m]
    return hands["right_hand"], hands["left_hand"], pose


def load_dataset(root: Path, cache: Path, workers: int):
    folders = sorted(f for f in root.glob("*/*") if f.is_dir())
    if not folders:
        sys.exit(f"Error: no <category>/<word> folders under {root}")
    items = [(f.parent.name, f.name, p) for f in folders for p in sorted(f.glob("*.parquet"))]
    store = {}
    if cache.is_file():
        try:
            store = pickle.load(open(cache, "rb"))
        except Exception:
            store = {}
    todo = [it for it in items if str(it[2].relative_to(root)) not in store]
    print(f"{len(items)} videos in {len(folders)} words ({len(items) - len(todo)} cached, {len(todo)} to read)")
    if todo:
        t0 = time.time()

        def work(it):
            try:
                return read_video(it[2])
            except Exception as e:
                return e
        with ThreadPoolExecutor(workers) as pool:
            for i, (it, res) in enumerate(zip(todo, pool.map(work, todo)), 1):
                key = str(it[2].relative_to(root))
                store[key] = None if isinstance(res, Exception) else tuple(a.astype(np.float16) for a in res)
                if isinstance(res, Exception):
                    print(f"  skip {key}: {res}")
                if i % 200 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  read {i}/{len(todo)}  ({el:.0f}s, ~{el / i * (len(todo) - i):.0f}s left)", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(store, open(cache, "wb"), protocol=4)
    return [(cat, word, store[str(p.relative_to(root))], str(p.relative_to(root))) for cat, word, p in items
            if store.get(str(p.relative_to(root))) is not None]


def load_personal(folder: Path):
    """Takes recorded in the app's Teach mode: personal/*.json."""
    out = []
    for f in sorted(folder.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            feats = np.asarray(d["features"], np.float32)
            if feats.ndim == 2 and feats.shape[1] == FEATURE_DIM and len(feats) >= 6:
                out.append((label_key(d["label"]), d.get("category", "your signs"), feats, f.name, d.get("raw"), d.get("fps", 30)))
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    return out


# --------------------------------------------------------------------------- sampling
class Clip:
    __slots__ = ("label", "cat", "feats", "a0", "a1", "key", "weight", "raw")

    def __init__(self, label, cat, feats, key, weight=1, raw=None):
        self.label, self.cat, self.feats, self.key, self.weight, self.raw = label, cat, feats, key, weight, raw
        active = np.where((feats[:, R0] > 0.5) | (feats[:, L0] > 0.5))[0]
        self.a0, self.a1 = (int(active[0]), int(active[-1])) if len(active) else (-1, -1)


def time_warp(rng: random.Random) -> np.ndarray:
    """Random increasing curve 0..1: the sign speeds up and slows down inside the window."""
    steps = np.array([rng.uniform(0.5, 1.5) for _ in range(T - 1)])
    u = np.concatenate([[0.0], np.cumsum(steps)])
    return u / u[-1]


def positive_window(c: Clip, rng: random.Random) -> np.ndarray:
    n_act = c.a1 - c.a0 + 1
    cover = rng.uniform(0.6, 1.0) * n_act
    start = c.a0 + rng.uniform(0, n_act - cover)
    margin = rng.uniform(-0.05, 0.25) * n_act
    t0 = start - rng.uniform(0, 1) * margin
    t1 = start + cover + rng.uniform(0, 1) * margin
    return resample(c.feats, t0, max(t1, t0 + 2), T, time_warp(rng) if rng.random() < 0.6 else None)


def none_window(clips, rng: random.Random) -> np.ndarray:
    kind = rng.random()
    a, b = rng.choice(clips), rng.choice(clips)
    if kind < 0.45:                                   # transition between two signs
        ta, tb = a.feats[a.a0:a.a1 + 1], b.feats[b.a0:b.a1 + 1]
        ca, cb = int(len(ta) * rng.uniform(0.45, 0.8)), int(len(tb) * rng.uniform(0.2, 0.55))
        seq = np.concatenate([ta[ca:], tb[:cb]]) if len(ta[ca:]) and len(tb[:cb]) else ta
        return resample(seq, 0, len(seq) - 1, T)
    if kind < 0.75:                                   # resting / hands down
        rest = np.concatenate([a.feats[:max(a.a0, 0)], a.feats[a.a1 + 1:]]) if a.a0 > 0 else a.feats[:0]
        if len(rest) >= 6:
            s = rng.randint(0, len(rest) - 6)
            return resample(rest, s, min(len(rest) - 1, s + rng.randint(6, 30)), T)
        w = np.zeros((T, FEATURE_DIM), np.float32)
        w[:, P0:] = a.feats[a.a0, P0:]
        return w
    n_act = a.a1 - a.a0 + 1                           # a sliver: too little of a sign to tell
    t0, t1 = ((a.a0 - 0.3 * n_act, a.a0 + 0.2 * n_act) if rng.random() < 0.5
              else (a.a1 - 0.15 * n_act, a.a1 + 0.4 * n_act))
    return resample(a.feats, t0, t1, T)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    B = len(x)
    flip = rng.random(B) < 0.5
    x[flip] = mirror(x[flip])
    th = rng.uniform(-0.2, 0.2, B).astype(np.float32)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    sc = rng.uniform(0.85, 1.15, B).astype(np.float32)[:, None]
    shift = rng.uniform(-0.15, 0.15, (B, 2)).astype(np.float32)
    # positions: hands and arm points share the same body-relative transform
    def tf(ix, iy, on):
        px, py = x[:, :, ix].copy(), x[:, :, iy].copy()
        x[:, :, ix] = np.where(on, (c * px - s * py) * sc + shift[:, :1], 0)
        x[:, :, iy] = np.where(on, (s * px + c * py) * sc + shift[:, 1:], 0)
    body_on = x[:, :, P0] > 0.5
    for k in range(5):
        tf(P0 + 1 + 2 * k, P0 + 2 + 2 * k, body_on)
    for b0 in (R0, L0):
        on = x[:, :, b0] > 0.5
        tf(b0 + 1, b0 + 2, on)
        shp = x[:, :, b0 + 3:b0 + HAND_BLOCK].reshape(B, T, 21, 3)
        hx, hy = shp[..., 0].copy(), shp[..., 1].copy()
        shp[..., 0] = c[..., None] * hx - s[..., None] * hy
        shp[..., 1] = s[..., None] * hx + c[..., None] * hy
        shp += rng.normal(0, 0.03, shp.shape).astype(np.float32) * on[..., None, None]
        x[:, :, b0 + 3:b0 + HAND_BLOCK] = shp.reshape(B, T, 63)
        drop = (rng.random((B, T)) < 0.06) & on               # tracker dropouts (fast motion)
        x[:, :, b0:b0 + HAND_BLOCK][drop] = 0
    nobody = rng.random(B) < 0.12                              # body tracker missing
    x[nobody, :, P0:] = 0
    for b0 in (R0, L0):
        x[nobody, :, b0 + 1:b0 + 3] = 0
    return x


# --------------------------------------------------------------------------- model
class TCN(nn.Module):
    """Per-frame projection -> 3 residual temporal convolutions (dilation 1, 2, 4) -> mean+max pool."""

    def __init__(self, n_classes: int, feat: int = FEATURE_DIM, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Linear(2 * feat, hidden)
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d) for d in (1, 2, 4)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        v = torch.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        h = torch.relu(self.inp(torch.cat([x, v], -1))).transpose(1, 2)
        for conv in self.convs:
            h = torch.relu(conv(h)) + h
        return self.out(self.drop(torch.cat([h.mean(2), h.amax(2)], 1)))


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def export_json(model, labels, cats, temperature, metrics, path: Path) -> None:
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    w = {"inp_w": b64(sd["inp.weight"].T), "inp_b": b64(sd["inp.bias"]),
         "out_w": b64(sd["out.weight"].T), "out_b": b64(sd["out.bias"]),
         "convs": [{"w": b64(sd[f"convs.{i}.weight"]), "b": b64(sd[f"convs.{i}.bias"]), "dilation": d}
                   for i, d in enumerate((1, 2, 4))]}
    spec = {"format": "isl-mk3", "feature_version": FEATURE_VERSION, "feature_dim": FEATURE_DIM, "T": T,
            "hidden": int(sd["inp.weight"].shape[0]), "temperature": float(temperature), "labels": labels,
            "display": {l: display_name(l) for l in labels}, "category": cats, "metrics": metrics, "w": w}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(spec), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- sign library (for the avatar)
def library_frames(r, l, p, aspect, a0, a1, max_frames=90):
    """Real signer movement, normalized: origin = middle of the shoulders, unit = shoulder width."""
    n = len(r)
    s0, s1 = max(0, a0 - 4), min(n - 1, a1 + 4)
    idx = np.arange(s0, s1 + 1)
    if len(idx) > max_frames:
        idx = np.rint(np.linspace(s0, s1, max_frames)).astype(int)
    pose = p.astype(np.float32).copy()
    pose[..., 0] *= aspect
    sh = pose[:, [LSH, RSH], :2]
    ok = np.isfinite(sh).all(axis=(1, 2))
    if not ok.any():
        return None, 1.0
    c = np.nanmedian((sh[ok, 0] + sh[ok, 1]) / 2, axis=0)
    s = float(np.nanmedian(np.linalg.norm(sh[ok, 0] - sh[ok, 1], axis=1)))
    if s < 1e-4:
        return None, 1.0
    out = []
    q = lambda v: None if not np.isfinite(v) else round(float(v), 3)
    for i in idx:
        fr = {}
        for key, hand in (("r", r[i]), ("l", l[i])):
            h = hand.astype(np.float32)
            if np.isfinite(h).all():
                hx = (h[:, 0] * aspect - c[0]) / s
                hy = (h[:, 1] - c[1]) / s
                hz = h[:, 2] * aspect / s
                fr[key] = [round(float(v), 3) for v in np.stack([hx, hy, hz], 1).reshape(-1)]
            else:
                fr[key] = None
        fr["p"] = [q((pose[i, k, 0] - c[0]) / s) if dim == 0 else q((pose[i, k, 1] - c[1]) / s)
                   for k in range(len(POSE_IDS)) for dim in (0, 1)]
        out.append(fr)
    return out, len(idx) / max(1, (s1 - s0 + 1))


def write_library(best_clips: dict, raw_by_key: dict, aspect: float, fps: float, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for label, clip in best_clips.items():
        if clip.raw is not None:                      # your own recording (already normalized by the app)
            frames, f = clip.raw.get("frames"), clip.raw.get("fps", 30)
        else:
            r, l, p = raw_by_key[clip.key]
            frames, ratio = library_frames(r.astype(np.float32), l.astype(np.float32), p.astype(np.float32),
                                           aspect, clip.a0, clip.a1)
            f = fps * ratio
        if not frames:
            continue
        (out_dir / f"{label}.json").write_text(json.dumps(
            {"label": label, "display": display_name(label), "fps": round(f, 2), "frames": frames}, separators=(",", ":")),
            encoding="utf-8")
        count += 1
    return count


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Train ISL Translator Mark III")
    ap.add_argument("--data", type=Path, required=True, help="the Kaggle 'keypoints' folder")
    ap.add_argument("--aspect", type=float, default=16 / 9, help="dataset video width/height (default 16:9)")
    ap.add_argument("--fps", type=float, default=25.0, help="dataset video frame rate (for avatar playback)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--finetune", action="store_true", help="quick update of the last model (after Teach mode)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--min-videos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    args = ap.parse_args()
    if args.finetune:
        args.epochs = min(args.epochs, 15)
        args.lr = min(args.lr, 5e-4)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rng, nrng = random.Random(args.seed), np.random.default_rng(args.seed)

    raw = load_dataset(args.data, HERE / "model" / "dataset_cache3.pkl", args.workers)
    raw_by_key = {key: arrays for _, _, arrays, key in raw}
    print("Computing features ...")
    clips, cats = [], {}
    for cat, word, (r, l, p), key in raw:
        lab = label_key(word)
        c = Clip(lab, cat.replace("_", " "), clip_features(r.astype(np.float32), l.astype(np.float32), p.astype(np.float32), args.aspect), key)
        if c.a0 >= 0 and c.a1 - c.a0 >= 3:
            clips.append(c)
            cats[lab] = c.cat
    personal = load_personal(HERE / "personal")
    for lab, cat, feats, name, rawp, fps in personal:
        c = Clip(lab, cat, feats, f"personal:{name}", PERSONAL_WEIGHT, rawp)
        if c.a0 >= 0 and c.a1 - c.a0 >= 3:
            clips.append(c)
            cats.setdefault(lab, cat)
    if personal:
        print(f"Your own recordings: {len(personal)} takes of {len({p[0] for p in personal})} words")

    per = Counter(c.label for c in clips if c.weight == 1)
    per_personal = Counter(c.label for c in clips if c.weight > 1)
    keep = sorted(l for l in set(per) | set(per_personal) if per[l] >= args.min_videos or per_personal[l] >= 2)
    dropped = sorted(set(per) - set(keep))
    clips = [c for c in clips if c.label in keep]
    labels = keep + [NONE]
    idx = {l: i for i, l in enumerate(labels)}
    print(f"{len(clips)} usable recordings, {len(keep)} words" + (f" (too few videos: {', '.join(dropped)})" if dropped else ""))

    by = defaultdict(list)
    for c in clips:
        by[c.label].append(c)
    train, val = [], []
    for l, cs in by.items():
        rng.shuffle(cs)
        ds = [c for c in cs if c.weight == 1]
        k = max(1, round(len(ds) * args.val)) if len(ds) >= 3 else 0
        val += ds[:k]
        train += [c for c in cs if c not in ds[:k]]
    print(f"train {len(train)}, check {len(val)} (held-back dataset videos, never trained on)")

    model = TCN(len(labels))
    ckpt_path = HERE / "model" / "isl_mk3.pt"
    if args.finetune and ckpt_path.is_file():
        old = torch.load(ckpt_path, weights_only=False)
        sd = model.state_dict()
        for k, v in old["state_dict"].items():
            if k.startswith("out."):
                continue
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v
        for i, l in enumerate(old["labels"]):                    # keep what the old model knew per word
            if l in idx:
                sd["out.weight"][idx[l]] = old["state_dict"]["out.weight"][i]
                sd["out.bias"][idx[l]] = old["state_dict"]["out.bias"][i]
        model.load_state_dict(sd)
        print(f"Fine-tuning the existing model ({len(old['labels']) - 1} -> {len(keep)} words)")
    elif args.finetune:
        print("No earlier model found: doing a full training instead.")

    ema = copy.deepcopy(model)
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    steps = args.epochs * (sum(c.weight for c in train) * args.windows * 7 // 6 // args.batch + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps + 10)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    vx = np.stack([resample(c.feats, c.a0, c.a1, T) for c in val]) if val else np.zeros((0, T, FEATURE_DIM), np.float32)
    vy = np.array([idx[c.label] for c in val], dtype=int)
    vn = np.stack([none_window(val or train, rng) for _ in range(max(60, len(val) // 5))])

    def evaluate(m):
        m.eval()
        with torch.no_grad():
            lv = m(torch.from_numpy(vx)).numpy() if len(vx) else np.zeros((0, len(labels)))
            ln = m(torch.from_numpy(vn)).numpy()
        return lv, ln

    best, best_state, best_metrics = -1.0, None, {}
    step_no = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        xs, ys = [], []
        for c in train:
            for _ in range(args.windows * c.weight):
                xs.append(positive_window(c, rng)); ys.append(idx[c.label])
        for _ in range(len(xs) // 6):
            xs.append(none_window(train, rng)); ys.append(idx[NONE])
        X = augment(np.stack(xs), nrng); Y = np.array(ys)
        perm = nrng.permutation(len(X))
        model.train(); tot = 0.0
        for i in range(0, len(X), args.batch):
            j = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(torch.from_numpy(X[j])), torch.from_numpy(Y[j]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            step_no += 1
            decay = min(0.995, (1 + step_no) / (10 + step_no))     # EMA warm-up: follows quickly at first
            with torch.no_grad():                                   # exponential moving average of the weights
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm, alpha=1 - decay)
            tot += loss.item() * len(j)
        msg = f"epoch {ep:3d}/{args.epochs}  loss {tot / len(X):.3f}"
        if len(vx):
            lv, ln = evaluate(ema)
            top1 = float((lv.argmax(1) == vy).mean())
            top5 = float((np.argsort(-lv, 1)[:, :5] == vy[:, None]).any(1).mean())
            none_ok = float((ln.argmax(1) == idx[NONE]).mean())
            msg += f"  | check: top-1 {top1:.3f}  top-5 {top5:.3f}  none {none_ok:.2f}"
            score = top1 + 0.2 * none_ok
            if score > best:
                best, best_state = score, copy.deepcopy(ema.state_dict())
                best_metrics = {"top1": round(top1, 4), "top5": round(top5, 4), "none": round(none_ok, 3), "epoch": ep}
                msg += "  *"
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best_state if best_state is not None else ema.state_dict())
    model.eval()

    # ---- confidence calibration: make "90%" mean right about 90% of the time
    temperature = 1.0
    if len(vx):
        lv, ln = evaluate(model)
        L = np.concatenate([lv, ln]); Yc = np.concatenate([vy, np.full(len(ln), idx[NONE])])
        def nll(t):
            z = L / t; z = z - z.max(1, keepdims=True)
            return float(-(z[np.arange(len(Yc)), Yc] - np.log(np.exp(z).sum(1))).mean())
        temperature = min(np.arange(0.5, 3.01, 0.05), key=nll)
        p = np.exp((lv / temperature) - (lv / temperature).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        conf, pred = p.max(1), p.argmax(1)
        for thr in (0.6, 0.7, 0.8, 0.9):
            sel = conf >= thr
            if sel.any():
                best_metrics[f"acc_at_{int(thr * 100)}"] = round(float((pred[sel] == vy[sel]).mean()), 4)
                best_metrics[f"shown_at_{int(thr * 100)}"] = round(float(sel.mean()), 4)
    best_metrics.update({"words": len(keep), "recordings": len(clips), "temperature": round(float(temperature), 3)})

    # ---- report
    lines = [f"ISL Translator Mk III - training report", json.dumps(best_metrics, indent=1)]
    if len(vx):
        lv, _ = evaluate(model)
        pred = lv.argmax(1)
        confused = Counter((labels[a], labels[b]) for a, b in zip(vy, pred) if a != b)
        acc = defaultdict(list)
        for a, b in zip(vy, pred):
            acc[labels[a]].append(a == b)
        weak = sorted(((np.mean(v), k, len(v)) for k, v in acc.items()))[:25]
        lines += ["", "Weakest words (accuracy on held-back videos):"] + [f"  {k:<18} {a:.2f}  ({n} videos)" for a, k, n in weak]
        lines += ["", "Most confused pairs (true -> predicted):"] + [f"  {a} -> {b}: {n}" for (a, b), n in confused.most_common(25)]
    (HERE / "model").mkdir(exist_ok=True)
    (HERE / "model" / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    cat_of = {l: cats.get(l, "") for l in labels}
    export_json(model, labels, cat_of, temperature, best_metrics, HERE / "static" / "model" / "model.json")
    torch.save({"state_dict": model.state_dict(), "labels": labels, "category": cat_of, "T": T,
                "feature_version": FEATURE_VERSION, "aspect": args.aspect, "temperature": temperature,
                "metrics": best_metrics}, ckpt_path)

    # ---- sign library: per word, the recording the model is most sure about (your own take first)
    with torch.no_grad():
        best_clip = {}
        for c in clips:
            if c.label == NONE:
                continue
            w = resample(c.feats, c.a0, c.a1, T)[None]
            pr = torch.softmax(model(torch.from_numpy(w)) / temperature, 1)[0, idx[c.label]].item()
            act = c.feats[c.a0:c.a1 + 1]
            coverage = float(((act[:, R0] > 0.5) | (act[:, L0] > 0.5)).mean() * (act[:, P0] > 0.5).mean())
            score = pr * coverage + (10 if c.weight > 1 and c.raw else 0)
            if score > best_clip.get(c.label, (-1, None))[0]:
                best_clip[c.label] = (score, c)
    n_lib = write_library({k: v[1] for k, v in best_clip.items()}, raw_by_key, args.aspect, args.fps, HERE / "static" / "signs")

    print(f"\nSaved the model, {n_lib} signs for the avatar, and model/report.txt")
    print(f"Check on held-back videos: {best_metrics}")
    print("Open model/report.txt to see the weakest words and the pairs it confuses.")


if __name__ == "__main__":
    main()
