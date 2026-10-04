#!/usr/bin/env python3
"""train.py - Mark II: train on the whole Kaggle ISL keypoint dataset.

    python train.py --data "C:\\Users\\VSJ\\Downloads\\archive\\keypoints"

Reads keypoints/<category>/<word>/*.parquet (MediaPipe Holistic: frame, type,
landmark_index, x, y, z), keeps both hands and the upper body, and trains a
small temporal convolution network that recognises a sign from a short window
of frames. Windows of different lengths are used, so fast and slow signing both
work, plus a "none" class (hands down, between-sign transitions) so continuous
signing can be split into words.

Output (used by the web app straight away):
    static/model/model.json    weights + labels for the browser (no server work per frame)
    model/isl_mk2.pt           PyTorch checkpoint (for further training)

The first run parses every video once and caches it (model/dataset_cache.pkl);
later runs start in seconds.
"""
from __future__ import annotations

import argparse
import base64
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

from features2 import (FEATURE_DIM, FEATURE_VERSION, HAND_BLOCK, L0, P0, POSE_IDS, R0,
                       clip_features, display_name, mirror, resample)

HERE = Path(__file__).resolve().parent
NONE = "none"
T = 24                                   # frames per window seen by the model


# --------------------------------------------------------------------------- data
def label_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_video(path: Path):
    """-> right (n,21,3), left (n,21,3), pose (n,3,3); NaN where missing."""
    df = pd.read_parquet(path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df = df[df["type"].isin(("right_hand", "left_hand", "pose"))]
    frames, inv = np.unique(df["frame"].to_numpy(), return_inverse=True)
    n = len(frames)
    out = {"right_hand": np.full((n, 21, 3), np.nan, np.float32),
           "left_hand": np.full((n, 21, 3), np.nan, np.float32),
           "pose": np.full((n, 3, 3), np.nan, np.float32)}
    typ = df["type"].to_numpy()
    lm = df["landmark_index"].to_numpy().astype(int)
    xyz = df[["x", "y", "z"]].to_numpy(np.float32)
    for t in ("right_hand", "left_hand"):
        m = (typ == t) & (lm >= 0) & (lm < 21)
        out[t][inv[m], lm[m]] = xyz[m]
    for k, pid in enumerate(POSE_IDS):
        m = (typ == "pose") & (lm == pid)
        out["pose"][inv[m], k] = xyz[m]
    # a hand counts only if all 21 points are there
    for t in ("right_hand", "left_hand"):
        bad = ~np.isfinite(out[t]).all(axis=(1, 2))
        out[t][bad] = np.nan
    return out["right_hand"], out["left_hand"], out["pose"]


def load_dataset(root: Path, cache: Path, workers: int):
    folders = sorted(f for f in root.glob("*/*") if f.is_dir())
    if not folders:
        sys.exit(f"Error: no <category>/<word> folders under {root}")
    items = [(f.parent.name, f.name, p) for f in folders for p in sorted(f.glob("*.parquet"))]
    if not items:
        sys.exit(f"Error: no .parquet files under {root}")
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
            except Exception as e:  # corrupt file
                return e
        with ThreadPoolExecutor(workers) as pool:
            for i, (it, res) in enumerate(zip(todo, pool.map(work, todo)), 1):
                key = str(it[2].relative_to(root))
                if isinstance(res, Exception):
                    print(f"  skip {key}: {res}")
                    res = None
                else:
                    res = tuple(a.astype(np.float16) for a in res)
                store[key] = res
                if i % 200 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  read {i}/{len(todo)}  ({el:.0f}s, ~{el / i * (len(todo) - i):.0f}s left)", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(store, open(cache, "wb"), protocol=4)
    return [(cat, word, store[str(p.relative_to(root))], str(p.relative_to(root))) for cat, word, p in items
            if store.get(str(p.relative_to(root))) is not None]


# --------------------------------------------------------------------------- sampling
class Clip:
    __slots__ = ("label", "cat", "feats", "a0", "a1", "key")

    def __init__(self, label, cat, feats, key):
        self.label, self.cat, self.feats, self.key = label, cat, feats, key
        active = np.where((feats[:, R0] > 0.5) | (feats[:, L0] > 0.5))[0]
        self.a0, self.a1 = (int(active[0]), int(active[-1])) if len(active) else (-1, -1)


def positive_window(c: Clip, rng: random.Random) -> np.ndarray:
    n_act = c.a1 - c.a0 + 1
    cover = rng.uniform(0.6, 1.0) * n_act                 # part of the sign that is visible
    start = c.a0 + rng.uniform(0, n_act - cover)
    margin = rng.uniform(-0.05, 0.25) * n_act             # some context before/after
    t0 = start - rng.uniform(0, 1) * margin
    t1 = start + cover + rng.uniform(0, 1) * margin
    return resample(c.feats, t0, max(t1, t0 + 2), T)


def none_window(clips: list[Clip], rng: random.Random) -> np.ndarray:
    kind = rng.random()
    a, b = rng.choice(clips), rng.choice(clips)
    if kind < 0.45:                                       # transition between two signs
        ta = a.feats[a.a0:a.a1 + 1]
        tb = b.feats[b.a0:b.a1 + 1]
        cut_a = int(len(ta) * rng.uniform(0.45, 0.8))
        cut_b = int(len(tb) * rng.uniform(0.2, 0.55))
        seq = np.concatenate([ta[cut_a:], tb[:cut_b]]) if len(ta[cut_a:]) and len(tb[:cut_b]) else ta
        return resample(seq, 0, len(seq) - 1, T)
    if kind < 0.75:                                       # resting: before/after the sign, or hands down
        rest = np.concatenate([a.feats[:max(a.a0, 0)], a.feats[a.a1 + 1:]]) if a.a0 > 0 else a.feats[:0]
        if len(rest) >= 6:
            s = rng.randint(0, len(rest) - 6)
            return resample(rest, s, min(len(rest) - 1, s + rng.randint(6, 30)), T)
        w = np.zeros((T, FEATURE_DIM), np.float32)
        w[:, P0:P0 + 3] = a.feats[a.a0, P0:P0 + 3]
        return w
    # a sliver of a sign (too little to tell): the start or the very end
    n_act = a.a1 - a.a0 + 1
    if rng.random() < 0.5:
        t0, t1 = a.a0 - 0.3 * n_act, a.a0 + 0.2 * n_act
    else:
        t0, t1 = a.a1 - 0.15 * n_act, a.a1 + 0.4 * n_act
    return resample(a.feats, t0, t1, T)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """x (B, T, F) in place-ish augmentation."""
    B = len(x)
    flip = rng.random(B) < 0.5
    x[flip] = mirror(x[flip])
    th = rng.uniform(-0.2, 0.2, B).astype(np.float32)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    sc = rng.uniform(0.85, 1.15, B).astype(np.float32)[:, None]
    shift = rng.uniform(-0.15, 0.15, (B, 2)).astype(np.float32)
    for b0 in (R0, L0):
        on = x[:, :, b0] > 0.5
        px, py = x[:, :, b0 + 1].copy(), x[:, :, b0 + 2].copy()
        x[:, :, b0 + 1] = np.where(on, (c * px - s * py) * sc + shift[:, :1], 0)
        x[:, :, b0 + 2] = np.where(on, (s * px + c * py) * sc + shift[:, 1:], 0)
        shp = x[:, :, b0 + 3:b0 + HAND_BLOCK].reshape(B, T, 21, 3)
        hx, hy = shp[..., 0].copy(), shp[..., 1].copy()
        shp[..., 0] = c[..., None] * hx - s[..., None] * hy
        shp[..., 1] = s[..., None] * hx + c[..., None] * hy
        shp += rng.normal(0, 0.03, shp.shape).astype(np.float32) * on[..., None, None]
        x[:, :, b0 + 3:b0 + HAND_BLOCK] = shp.reshape(B, T, 63)
        drop = (rng.random((B, T)) < 0.06) & on               # tracker dropouts
        x[:, :, b0:b0 + HAND_BLOCK][drop] = 0
    nopose = rng.random(B) < 0.15                              # pose model missing/failed
    x[nopose, :, P0:P0 + 3] = 0
    for b0 in (R0, L0):
        x[nopose, :, b0 + 1:b0 + 3] = 0
    return x


# --------------------------------------------------------------------------- model
class TCN(nn.Module):
    """Per-frame projection -> two residual temporal convolutions -> mean+max pool -> classes.
    Input gets frame-to-frame velocity appended, so movement is explicit."""

    def __init__(self, n_classes: int, feat: int = FEATURE_DIM, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Linear(2 * feat, hidden)
        self.c1 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.c2 = nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):                                # (B, T, F)
        v = torch.zeros_like(x)
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        h = torch.relu(self.inp(torch.cat([x, v], -1))).transpose(1, 2)   # (B, H, T)
        h = torch.relu(self.c1(h)) + h
        h = torch.relu(self.c2(h)) + h
        pooled = torch.cat([h.mean(2), h.amax(2)], 1)
        return self.out(self.drop(pooled))


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def export_json(model: TCN, labels, cats, path: Path, metrics: dict) -> None:
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    spec = {
        "format": "isl-mk2", "feature_version": FEATURE_VERSION, "feature_dim": FEATURE_DIM, "T": T,
        "hidden": int(sd["inp.weight"].shape[0]), "labels": labels,
        "display": {l: display_name(l) for l in labels}, "category": cats, "metrics": metrics,
        "w": {
            "inp_w": b64(sd["inp.weight"].T), "inp_b": b64(sd["inp.bias"]),           # (2F, H)
            "c1_w": b64(sd["c1.weight"]), "c1_b": b64(sd["c1.bias"]),               # (H, H, 5)
            "c2_w": b64(sd["c2.weight"]), "c2_b": b64(sd["c2.bias"]),
            "out_w": b64(sd["out.weight"].T), "out_b": b64(sd["out.bias"]),         # (2H, C)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(spec), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Train ISL Translator Mark II on the Kaggle keypoint dataset")
    ap.add_argument("--data", type=Path, required=True, help="the 'keypoints' folder")
    ap.add_argument("--aspect", type=float, default=16 / 9, help="video width/height of the dataset (default 16:9)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--windows", type=int, default=6, help="random windows per video per epoch")
    ap.add_argument("--val", type=float, default=0.15, help="share of videos held out for checking")
    ap.add_argument("--min-videos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rng, nrng = random.Random(args.seed), np.random.default_rng(args.seed)

    raw = load_dataset(args.data, HERE / "model" / "dataset_cache.pkl", args.workers)
    print("Computing features ...")
    clips, cats = [], {}
    for cat, word, (r, l, p), key in raw:
        lab = label_key(word)
        f = clip_features(r.astype(np.float32), l.astype(np.float32), p.astype(np.float32), args.aspect)
        c = Clip(lab, cat, f, key)
        if c.a0 >= 0 and c.a1 - c.a0 >= 3:
            clips.append(c)
            cats[lab] = cat.replace("_", " ")
    per = Counter(c.label for c in clips)
    keep = sorted(l for l, n in per.items() if n >= args.min_videos)
    dropped = sorted(set(per) - set(keep))
    clips = [c for c in clips if c.label in keep]
    labels = keep + [NONE]
    idx = {l: i for i, l in enumerate(labels)}
    print(f"{len(clips)} usable videos, {len(keep)} words" + (f" (dropped, too few videos: {', '.join(dropped)})" if dropped else ""))

    by = defaultdict(list)
    for c in clips:
        by[c.label].append(c)
    train, val = [], []
    for l, cs in by.items():
        rng.shuffle(cs)
        k = max(1, round(len(cs) * args.val)) if len(cs) >= 3 else 0
        val += cs[:k]; train += cs[k:]
    print(f"train {len(train)} videos, check {len(val)} videos")

    model = TCN(len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs *
                                                ((len(train) * args.windows * 7 // 6) // args.batch + 1))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    # fixed check set: the whole active part of each held-out video + some none windows
    vx = np.stack([resample(c.feats, c.a0, c.a1, T) for c in val]) if val else np.zeros((0, T, FEATURE_DIM), np.float32)
    vy = np.array([idx[c.label] for c in val])
    vn = np.stack([none_window(val or train, rng) for _ in range(max(50, len(val) // 5))])

    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        xs, ys = [], []
        for c in train:
            for _ in range(args.windows):
                xs.append(positive_window(c, rng)); ys.append(idx[c.label])
        for _ in range(len(xs) // 6):
            xs.append(none_window(train, rng)); ys.append(idx[NONE])
        X = augment(np.stack(xs), nrng); Y = np.array(ys)
        perm = nrng.permutation(len(X))
        model.train(); tot = 0.0
        for i in range(0, len(X), args.batch):
            j = perm[i:i + args.batch]
            xb, yb = torch.from_numpy(X[j]), torch.from_numpy(Y[j])
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(j)
        msg = f"epoch {ep:3d}/{args.epochs}  loss {tot / len(X):.3f}"
        if len(vx):
            model.eval()
            with torch.no_grad():
                pv = torch.softmax(model(torch.from_numpy(vx)), 1).numpy()
                pn = torch.softmax(model(torch.from_numpy(vn)), 1).numpy()
            top1 = float((pv.argmax(1) == vy).mean())
            top5 = float((np.argsort(-pv, 1)[:, :5] == vy[:, None]).any(1).mean())
            none_ok = float((pn.argmax(1) == idx[NONE]).mean())
            msg += f"  | check: top-1 {top1:.3f}  top-5 {top5:.3f}  none {none_ok:.2f}"
            score = top1 + 0.2 * none_ok
            if score > best:
                best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
                best_metrics = {"top1": round(top1, 4), "top5": round(top5, 4), "none": round(none_ok, 3),
                                "epoch": ep, "words": len(keep), "videos": len(clips)}
                msg += "  *"
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_metrics = {"words": len(keep), "videos": len(clips)}
    model.eval()
    cat_of = {l: cats.get(l, "") for l in labels}
    export_json(model, labels, cat_of, HERE / "static" / "model" / "model.json", best_metrics)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "category": cat_of, "T": T,
                "feature_version": FEATURE_VERSION, "aspect": args.aspect, "metrics": best_metrics},
               HERE / "model" / "isl_mk2.pt")
    print(f"\nSaved static/model/model.json and model/isl_mk2.pt  ->  {best_metrics}")


if __name__ == "__main__":
    main()
