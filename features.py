"""Landmark features for the ISL translator (NumPy only - no OpenCV, MediaPipe or torch).

Kept dependency-free so the same code runs in training, on the desktop and on a
small server that only receives landmark coordinates.

Feature pipeline (FEATURE_VERSION 2), applied identically everywhere:
    1. aspect correction: x and z are multiplied by width/height so a hand has
       the same shape on a 16:9 webcam and a 4:3 dataset video
    2. translation: the wrist (landmark 0) becomes the origin
    3. mirroring: the signer's LEFT hand is mirrored (x -> -x) so both hands
       produce the same shape for the same handshape
    4. scaling: divide by the palm size (mean wrist->index/middle/pinky knuckle
       distance), so the distance from the camera no longer matters
    -> flat (63,) float32 vector (21 landmarks x (x, y, z))
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63
FEATURE_VERSION = 2

WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17

NONE_LABEL = "none"  # "no sign / resting hand" class; never displayed or spoken

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def canonical_features_batch(points, aspect=None, is_left=None,
                             mirror_left: bool = True, scale: bool = True) -> np.ndarray:
    """Vectorized canonical_features: (N, 21, 3) raw landmarks -> (N, 63).

    aspect   width/height per frame, scalar or (N,); None = no aspect correction
    is_left  bool per frame (N,) or None (= all right hands)
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, NUM_LANDMARKS, 3).copy()
    n = len(pts)
    if aspect is not None:
        a = np.broadcast_to(np.asarray(aspect, dtype=np.float32), (n,))[:, None]
        pts[:, :, 0] *= a
        pts[:, :, 2] *= a
    pts -= pts[:, WRIST:WRIST + 1, :]
    if mirror_left and is_left is not None:
        left = np.asarray(is_left, dtype=bool).reshape(n)
        pts[left, :, 0] *= -1.0
    if scale:
        palm = np.linalg.norm(pts[:, [INDEX_MCP, MIDDLE_MCP, PINKY_MCP], :], axis=2).mean(axis=1)
        palm = np.where(palm > 1e-6, palm, 1.0)
        pts /= palm[:, None, None]
    return pts.reshape(n, FEATURE_DIM)


def canonical_features(points, image_size: tuple[int, int] | None = None,
                       is_left: bool = False, mirror_left: bool = True,
                       scale: bool = True) -> np.ndarray:
    """Raw landmarks (21, 3) in MediaPipe normalized image coords -> (63,) vector.

    points      x, y in [0, 1] of image width/height; z on roughly the x scale
    image_size  (width, height) of the frame the landmarks came from; None = no
                aspect correction (use when coordinates are already square)
    is_left     True if this is the signer's left hand
    """
    aspect = None
    if image_size is not None and image_size[0] > 0 and image_size[1] > 0:
        aspect = image_size[0] / image_size[1]
    return canonical_features_batch(points, aspect, [is_left], mirror_left, scale)[0]


def legacy_features(points) -> np.ndarray:
    """FEATURE_VERSION 1 (the original scripts): wrist-relative only."""
    pts = np.asarray(points, dtype=np.float32).reshape(NUM_LANDMARKS, 3).copy()
    pts -= pts[WRIST]
    return pts.reshape(FEATURE_DIM)


def featurize(hand: "Hand", feature_cfg: dict) -> np.ndarray:
    """Apply whichever feature version a trained model expects."""
    if int(feature_cfg.get("version", 1)) < 2:
        return legacy_features(hand.points)
    return canonical_features(
        hand.points,
        image_size=hand.image_size if feature_cfg.get("aspect_correct", True) else None,
        is_left=hand.is_left,
        mirror_left=feature_cfg.get("mirror_left", True),
        scale=feature_cfg.get("scale", True),
    )


DEFAULT_FEATURE_CFG = {"version": FEATURE_VERSION, "aspect_correct": True,
                       "mirror_left": True, "scale": True}
