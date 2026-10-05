"""Mark III features: both hands + the arms + where everything is relative to the body.

The same maths runs in the browser (static/js/engine3.js); keep the two in step.

Per frame (FEATURE_DIM = 143):
    [R_present, R_pos(2), R_shape(63),            signer's right hand   (66)
     L_present, L_pos(2), L_shape(63),            signer's left hand    (66)
     body_present, nose(2), l_elbow(2), r_elbow(2), l_wrist(2), r_wrist(2)]   (11)

    shape   21 hand landmarks, wrist-relative, divided by palm size (distance-independent)
    pos     hand wrist relative to the middle of the shoulders, in shoulder widths
    arms    elbows and wrists from the body tracker, same units. They keep fast movements
            visible even in frames where motion blur makes the hand itself hard to see.
    All x (and hand z) values are first multiplied by the frame's width/height.

Mirroring (left-handed signers) swaps right/left blocks and negates every x value.
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
HAND_BLOCK = 1 + 2 + NUM_LANDMARKS * 3              # 66
BODY_BLOCK = 11
FEATURE_DIM = 2 * HAND_BLOCK + BODY_BLOCK          # 143
R0, L0, P0 = 0, HAND_BLOCK, 2 * HAND_BLOCK
# MediaPipe Pose ids: nose, left/right shoulder, left/right elbow, left/right wrist (the person's own sides)
POSE_IDS = (0, 11, 12, 13, 14, 15, 16)
NOSE, LSH, RSH, LEL, REL, LWR, RWR = range(7)
FEATURE_VERSION = 3


def _hand_block(hand, center, scale, out: np.ndarray, a: float) -> None:
    if hand is None or not np.isfinite(hand).all():
        out[:] = 0.0
        return
    p = hand.astype(np.float32).copy()
    p[:, 0] *= a
    p[:, 2] *= a
    wrist = p[0].copy()
    p -= wrist
    palm = float(np.linalg.norm(p[[5, 9, 17]], axis=1).mean())
    if palm > 1e-6:
        p /= palm
    out[0] = 1.0
    if center is not None:
        out[1] = (wrist[0] - center[0]) / scale
        out[2] = (wrist[1] - center[1]) / scale
    else:
        out[1:3] = 0.0
    out[3:] = p.reshape(-1)


def frame_features(right, left, pose, aspect: float) -> np.ndarray:
    """right/left: (21, 3) or None; pose: (7, 2+) in POSE_IDS order (NaN where unseen) or None."""
    out = np.zeros(FEATURE_DIM, np.float32)
    center = scale = None
    if pose is not None and np.isfinite(pose[[LSH, RSH], :2]).all():
        q = pose[:, :2].astype(np.float32).copy()
        q[:, 0] *= aspect
        c = (q[LSH] + q[RSH]) / 2
        s = float(np.linalg.norm(q[LSH] - q[RSH]))
        if s > 1e-4:
            center, scale = c, s
            out[P0] = 1.0
            for k, idx in enumerate((NOSE, LEL, REL, LWR, RWR)):
                if np.isfinite(q[idx]).all():
                    out[P0 + 1 + 2 * k: P0 + 3 + 2 * k] = (q[idx] - c) / s
    _hand_block(right, center, scale, out[R0:R0 + HAND_BLOCK], aspect)
    _hand_block(left, center, scale, out[L0:L0 + HAND_BLOCK], aspect)
    return out


def clip_features(right, left, pose, aspect: float) -> np.ndarray:
    n = len(right)
    out = np.zeros((n, FEATURE_DIM), np.float32)
    for i in range(n):
        out[i] = frame_features(right[i], left[i], pose[i], aspect)
    return out


def _x_mask() -> np.ndarray:
    m = np.zeros(FEATURE_DIM, bool)
    for b in (R0, L0):
        m[b + 1] = True
        m[b + 3: b + HAND_BLOCK: 3] = True
    m[P0 + 1: P0 + BODY_BLOCK: 2] = True               # x of nose, elbows, wrists
    return m


X_MASK = _x_mask()


def mirror(x: np.ndarray) -> np.ndarray:
    """Mirror features (..., FEATURE_DIM): left-handed <-> right-handed."""
    y = x.copy()
    y[..., R0:R0 + HAND_BLOCK], y[..., L0:L0 + HAND_BLOCK] = x[..., L0:L0 + HAND_BLOCK], x[..., R0:R0 + HAND_BLOCK]
    for a, b in ((3, 5), (7, 9)):                      # (left elbow, right elbow), (left wrist, right wrist)
        y[..., P0 + a:P0 + a + 2], y[..., P0 + b:P0 + b + 2] = x[..., P0 + b:P0 + b + 2], x[..., P0 + a:P0 + a + 2]
    y[..., X_MASK] *= -1.0
    return y


def resample(seq: np.ndarray, t0: float, t1: float, T: int, warp: np.ndarray | None = None) -> np.ndarray:
    """Nearest-frame resampling of seq (n, F) between fractional indices t0..t1 to T frames.
    warp: optional increasing (T,) curve in [0, 1] for speed changes within the window."""
    u = np.linspace(0, 1, T) if warp is None else warp
    idx = np.rint(np.clip(t0 + (t1 - t0) * u, 0, len(seq) - 1)).astype(int)
    return seq[idx]


DISPLAY = {
    "trainticket": "train ticket", "trainstation": "train station", "storeorshop": "store",
    "youplural": "you all", "howareyou": "how are you", "goodmorning": "good morning",
    "goodafternoon": "good afternoon", "goodevening": "good evening", "goodnight": "good night",
    "thankyou": "thank you", "biglarge": "big", "smalllittle": "small", "streetorroad": "street",
    "raceethnicity": "race", "exmonsoon": "monsoon", "tshirt": "T-shirt", "pant": "pants",
    "cellphone": "cell phone", "i": "I", "india": "India", "monday": "Monday", "tuesday": "Tuesday",
    "wednesday": "Wednesday", "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
    "sunday": "Sunday", "god": "God", "none": "",
}


def display_name(label: str) -> str:
    return DISPLAY.get(label, label.replace("_", " "))
