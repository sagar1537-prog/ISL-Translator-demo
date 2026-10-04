"""Mark II features: both hands + where they are relative to the body.

The same maths runs in the browser (static/js/engine2.js); keep the two in step.

Per frame (FEATURE_DIM = 135):
    [R_present, R_pos(2), R_shape(63),  L_present, L_pos(2), L_shape(63),  pose_present, nose(2)]

    R / L     the signer's own right / left hand
    shape     21 landmarks, wrist-relative, divided by palm size (distance-independent)
    pos       wrist position relative to the middle of the shoulders, in shoulder widths
    nose      head position relative to the shoulders, in shoulder widths
    All x (and hand z) values are first multiplied by the frame's width/height,
    so a 16:9 webcam and a 4:3 video give the same shapes.

Mirroring (left-handed signers) is a pure feature operation: swap R and L blocks
and negate every x value. Training uses it as augmentation, so the model works
for either dominant hand.
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
HAND_BLOCK = 1 + 2 + NUM_LANDMARKS * 3        # 66
FEATURE_DIM = 2 * HAND_BLOCK + 3              # 135
R0, L0, P0 = 0, HAND_BLOCK, 2 * HAND_BLOCK    # block offsets
POSE_IDS = (0, 11, 12)                        # nose, left shoulder, right shoulder (MediaPipe Pose)
FEATURE_VERSION = 1


def _hand_block(hand: np.ndarray | None, center, scale, out: np.ndarray, a: float) -> None:
    """hand (21, 3) raw normalized coords -> out (66,)."""
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


def frame_features(right: np.ndarray | None, left: np.ndarray | None,
                   pose: np.ndarray | None, aspect: float) -> np.ndarray:
    """right/left: (21, 3) or None; pose: (3, 3) = nose, l_shoulder, r_shoulder, or None."""
    out = np.zeros(FEATURE_DIM, np.float32)
    center = scale = None
    if pose is not None and np.isfinite(pose[:, :2]).all():
        q = pose[:, :2].astype(np.float32).copy()
        q[:, 0] *= aspect
        c = (q[1] + q[2]) / 2
        s = float(np.linalg.norm(q[1] - q[2]))
        if s > 1e-4:
            center, scale = c, s
            out[P0] = 1.0
            out[P0 + 1:P0 + 3] = (q[0] - c) / s
    _hand_block(right, center, scale, out[R0:R0 + HAND_BLOCK], aspect)
    _hand_block(left, center, scale, out[L0:L0 + HAND_BLOCK], aspect)
    return out


def clip_features(right: np.ndarray, left: np.ndarray, pose: np.ndarray, aspect: float) -> np.ndarray:
    """(n, 21, 3), (n, 21, 3), (n, 3, 3) with NaN for missing -> (n, FEATURE_DIM)."""
    n = len(right)
    out = np.zeros((n, FEATURE_DIM), np.float32)
    for i in range(n):
        out[i] = frame_features(right[i], left[i], pose[i], aspect)
    return out


# x components inside one frame vector (for mirroring)
def _x_mask() -> np.ndarray:
    m = np.zeros(FEATURE_DIM, bool)
    for b in (R0, L0):
        m[b + 1] = True                                 # pos x
        m[b + 3: b + HAND_BLOCK: 3] = True              # shape x
    m[P0 + 1] = True                                    # nose x
    return m


X_MASK = _x_mask()


def mirror(x: np.ndarray) -> np.ndarray:
    """Mirror features (..., FEATURE_DIM): left-handed <-> right-handed."""
    y = x.copy()
    y[..., R0:R0 + HAND_BLOCK], y[..., L0:L0 + HAND_BLOCK] = x[..., L0:L0 + HAND_BLOCK], x[..., R0:R0 + HAND_BLOCK]
    y[..., X_MASK] *= -1.0
    return y


def resample(seq: np.ndarray, t0: float, t1: float, T: int) -> np.ndarray:
    """Nearest-frame resampling of seq (n, F) between fractional indices t0..t1 to T frames.
    (Nearest keeps presence flags exact; the browser does the same by timestamps.)"""
    idx = np.rint(np.clip(np.linspace(t0, t1, T), 0, len(seq) - 1)).astype(int)
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
