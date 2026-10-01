"""engine.py - torch-free sign recognition (NumPy only).

Everything needed to turn hand landmarks into words at run time:

    load_model(path)      isl_model.npz (or .pth, converted once) -> NumpyModel
    Predictor             per-frame wrapper for frame (MLP) and window (GRU) models
    Smoother              temporal voting, so one odd frame never shows a word
    SignEngine            Predictor + Smoother behind one call; usable on a server
                          that only receives landmark coordinates

PyTorch is needed only for training. At run time this keeps memory use and
install size small (no ~800 MB torch install, ~40 MB less RAM, instant start),
which is what makes free-tier hosting possible. The maths is identical to the
torch models (verified to within 1e-5 in the tests).
"""
from __future__ import annotations

import json
import re
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from features import FEATURE_DIM, NONE_LABEL, featurize

NPZ_META_KEY = "__meta__"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _natural(key: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key)]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically safe


class NumpyModel:
    """Frame MLP or window GRU with the same weights and maths as models.py."""

    def __init__(self, arch: dict, labels: list[str], feature: dict, weights: dict[str, np.ndarray],
                 legacy: bool = False, metrics: dict | None = None) -> None:
        self.arch, self.labels, self.feature = dict(arch), list(labels), dict(feature)
        self.legacy, self.metrics = legacy, metrics or {}
        self.weights = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        if self.arch["type"] == "mlp":
            prefixes = sorted({k.rsplit(".", 1)[0] for k, v in self.weights.items()
                               if k.endswith(".weight") and v.ndim == 2}, key=_natural)
            self._layers = [(self.weights[f"{p}.weight"].T.copy(), self.weights[f"{p}.bias"]) for p in prefixes]
            in_dim = self._layers[0][0].shape[0]
            n_out = self._layers[-1][0].shape[1]
        elif self.arch["type"] == "gru":
            n_layers = int(self.arch.get("layers", 1))
            self._gru = []
            for k in range(n_layers):
                self._gru.append((self.weights[f"gru.weight_ih_l{k}"].T.copy(),
                                  self.weights[f"gru.weight_hh_l{k}"].T.copy(),
                                  self.weights[f"gru.bias_ih_l{k}"], self.weights[f"gru.bias_hh_l{k}"]))
            head = sorted({k.rsplit(".", 1)[0] for k in self.weights
                           if k.startswith("head.") and k.endswith(".weight")}, key=_natural)[-1]
            self._head = (self.weights[f"{head}.weight"].T.copy(), self.weights[f"{head}.bias"])
            in_dim = self._gru[0][0].shape[0] // 2
            n_out = self._head[0].shape[1]
        else:
            raise ValueError(f"Unknown model type: {self.arch['type']!r}")
        if in_dim != FEATURE_DIM:
            raise ValueError(f"Model expects {in_dim} inputs; this project produces {FEATURE_DIM}.")
        if n_out != len(self.labels):
            raise ValueError(f"Model has {n_out} outputs but {len(self.labels)} labels.")

    @property
    def window(self) -> int | None:
        return self.arch.get("window") if self.arch["type"] == "gru" else None

    @property
    def has_none_class(self) -> bool:
        return NONE_LABEL in self.labels

    def logits(self, x: np.ndarray) -> np.ndarray:
        """x: (B, 63) for an MLP, (B, T, 63) for a GRU -> (B, classes)."""
        x = np.asarray(x, dtype=np.float32)
        if self.arch["type"] == "mlp":
            for i, (w, b) in enumerate(self._layers):
                x = x @ w + b
                if i < len(self._layers) - 1:
                    np.maximum(x, 0.0, out=x)
            return x
        vel = np.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        seq = np.concatenate([x, vel], axis=-1)
        h = None
        for w_ih, w_hh, b_ih, b_hh in self._gru:
            hid = w_hh.shape[0]
            gx = seq @ w_ih + b_ih                       # (B, T, 3H), gates r, z, n
            h = np.zeros((seq.shape[0], hid), np.float32)
            outs = []
            for t in range(seq.shape[1]):
                gh = h @ w_hh + b_hh
                r = _sigmoid(gx[:, t, :hid] + gh[:, :hid])
                z = _sigmoid(gx[:, t, hid:2 * hid] + gh[:, hid:2 * hid])
                n = np.tanh(gx[:, t, 2 * hid:] + r * gh[:, 2 * hid:])
                h = (1.0 - z) * n + z * h
                outs.append(h)
            seq = np.stack(outs, axis=1)
        return h @ self._head[0] + self._head[1]

    def probs(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def save_npz(path: Path | str, arch: dict, labels: list[str], feature: dict,
             weights: dict[str, np.ndarray], metrics: dict | None = None, legacy: bool = False) -> Path:
    """Write a self-describing, torch-free model file (no pickle inside)."""
    path = Path(path)
    meta = {"format": "isl-translator-npz", "version": 1, "arch": arch, "labels": list(labels),
            "feature": dict(feature), "metrics": metrics or {}, "legacy": legacy}
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
    arrays[NPZ_META_KEY] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


def load_npz(path: Path | str) -> NumpyModel:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(bytes(data[NPZ_META_KEY]).decode("utf-8"))
        weights = {k: data[k] for k in data.files if k != NPZ_META_KEY}
    return NumpyModel(meta["arch"], meta["labels"], meta["feature"], weights,
                      legacy=meta.get("legacy", False), metrics=meta.get("metrics", {}))


def load_model(path: Path | str, classes: Path | str | None = None, verbose: bool = True) -> NumpyModel:
    """Load isl_model.npz, or a .pth checkpoint.

    For a .pth, an up-to-date .npz next to it is used if present (no torch
    needed). Otherwise the .pth is converted once with torch and the .npz is
    written, so later starts don't need torch at all.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return load_npz(path)
    npz = path.with_suffix(".npz")
    if npz.is_file() and (not path.is_file() or npz.stat().st_mtime >= path.stat().st_mtime) and classes is None:
        return load_npz(npz)
    if not path.is_file():
        raise FileNotFoundError(f"model not found: {path}")
    try:
        import torch  # noqa: F401
        from models import load_checkpoint
    except ImportError:
        raise SystemExit(f"Error: {path.name} is a PyTorch file and torch is not installed.\n"
                         f"Convert it once on a machine with torch (python export_model.py {path.name}) "
                         f"and use the .npz file.") from None
    loaded = load_checkpoint(path, classes, "cpu")
    weights = {k: v.detach().cpu().numpy() for k, v in loaded.model.state_dict().items()}
    model = NumpyModel(loaded.arch, loaded.labels, loaded.feature, weights, loaded.legacy, loaded.metrics)
    try:
        save_npz(npz, loaded.arch, loaded.labels, loaded.feature, weights, loaded.metrics, loaded.legacy)
        if verbose:
            print(f"Converted {path.name} -> {npz.name} (torch-free; used automatically from now on)")
    except OSError:
        pass
    return model


# --------------------------------------------------------------------------- #
# Per-frame prediction and temporal smoothing
# --------------------------------------------------------------------------- #
class Predictor:
    """Wraps a frame model or a window model behind one per-frame call."""

    def __init__(self, model: NumpyModel, min_fill: float = 0.8) -> None:
        self.m = model
        self.window = model.window
        self.min_valid = int(np.ceil((self.window or 1) * min_fill))
        self.buffer: deque[np.ndarray | None] = deque(maxlen=self.window or 1)

    def reset(self) -> None:
        self.buffer.clear()

    def update(self, hand) -> tuple[str, float] | None:
        """Call once per frame (hand may be None). Returns the raw (label, confidence)."""
        vec = featurize(hand, self.m.feature) if hand is not None else None
        if self.window is None:
            if vec is None:
                return None
            probs = self.m.probs(vec[None])[0]
        else:
            self.buffer.append(vec)
            valid = [v for v in self.buffer if v is not None]
            if len(self.buffer) < self.window or len(valid) < self.min_valid or vec is None:
                return None
            seq, last = [], valid[0]  # fill short gaps with the nearest earlier frame
            for v in self.buffer:
                last = v if v is not None else last
                seq.append(last)
            probs = self.m.probs(np.stack(seq)[None])[0]
        i = int(probs.argmax())
        return self.m.labels[i], float(probs[i])


class Smoother:
    """Accept a word only if it wins >= min_votes of the last `size` frames."""

    def __init__(self, size: int, min_votes: int, threshold: float) -> None:
        self.buf: deque[tuple[str | None, float]] = deque(maxlen=size)
        self.min_votes, self.threshold = min(min_votes, size), threshold

    def reset(self) -> None:
        self.buf.clear()

    def update(self, raw: tuple[str, float] | None) -> None:
        if raw is not None and raw[0] != NONE_LABEL and raw[1] >= self.threshold:
            self.buf.append(raw)
        else:  # no hand, "none" sign, or low confidence -> empty vote
            self.buf.append((None, 0.0))

    def result(self) -> tuple[str, float] | None:
        votes = Counter(l for l, _ in self.buf if l is not None)
        if not votes:
            return None
        label, n = votes.most_common(1)[0]
        if n < self.min_votes:
            return None
        confs = [c for l, c in self.buf if l == label]
        return label, sum(confs) / len(confs)


class SignEngine:
    """Landmarks in, words out. One instance per user/stream.

    Desktop:  engine.update(hand)                       (a common.Hand)
    Server:   engine.update_landmarks(points, is_left, (w, h))
              points = 21 x [x, y, z] in MediaPipe normalized image coordinates
              of the RAW (un-mirrored) frame, e.g. from MediaPipe in the browser.
    Returns (raw, stable): raw = this frame's (label, conf) or None,
    stable = the voted word (label, conf) or None.
    """

    def __init__(self, model: NumpyModel, threshold: float = 0.65, buffer: int = 15, votes: int = 10) -> None:
        self.model = model
        self.predictor = Predictor(model)
        self.smoother = Smoother(buffer, votes, threshold)

    def reset(self) -> None:
        self.predictor.reset()
        self.smoother.reset()

    def update(self, hand):
        raw = self.predictor.update(hand)
        self.smoother.update(raw)
        return raw, self.smoother.result()

    def update_landmarks(self, points, is_left: bool = False, image_size: tuple[int, int] | None = None):
        if points is None:
            return self.update(None)
        pts = np.asarray(points, dtype=np.float32).reshape(21, 3)
        hand = SimpleNamespace(points=pts, is_left=bool(is_left), image_size=image_size or (1, 1))
        if image_size is None:  # unknown frame shape: treat coordinates as square
            hand.image_size = (1, 1)
        return self.update(hand)
