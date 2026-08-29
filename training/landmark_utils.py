"""
landmark_utils.py
------------------
Normalization of MediaPipe landmarks, shared between training (here) and
inference in the browser (web/js/normalize.js). Both implementations must
remain mathematically identical, otherwise the model gets a different input
distribution in production than during training.

Landmark format: (21, 3) numpy array, MediaPipe coordinates in range 0..1.
"""

import numpy as np

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP = 5, 6
MIDDLE_MCP = 9
MIDDLE_PIP = 10
RING_MCP, RING_PIP = 13, 14
PINKY_MCP = 17

# Distance of the thumb from these joints distinguishes A/S/T and M/N (thumb position).
THUMB_DIST_TARGETS = [INDEX_MCP, INDEX_PIP, MIDDLE_MCP, MIDDLE_PIP,
                      RING_MCP, RING_PIP, PINKY_MCP]
BASE_DIM = 63
# +2: (cos, sin) of the hand tilt angle before the rotation correction. G/Q, H/U, K/P
# differ only by rotation of the whole hand, so this must not be discarded.
FEATURE_DIM = BASE_DIM + len(THUMB_DIST_TARGETS) + 2  # 72


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translation (wrist to origin) + rotation in the image plane
    (aligns wrist->middle finger to (0,-1), for rotation invariance)
    + scaling (dividing by palm length) + thumb distances + original angle.
    Returns a vector of FEATURE_DIM (72) values."""
    pts = landmarks.astype(np.float32).copy()
    pts -= pts[WRIST]

    mx, my = pts[MIDDLE_MCP][0], pts[MIDDLE_MCP][1]
    r_xy = float(np.hypot(mx, my))
    if r_xy > 1e-6:
        cos_a, sin_a = -my / r_xy, mx / r_xy
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = cos_a * x + sin_a * y
        pts[:, 1] = -sin_a * x + cos_a * y
        # z is not rotated, MediaPipe's depth is too weak for that
    else:
        cos_a, sin_a = 1.0, 0.0

    scale = np.linalg.norm(pts[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1.0
    pts /= scale

    thumb = pts[THUMB_TIP]
    extra = np.array([np.linalg.norm(thumb - pts[i]) for i in THUMB_DIST_TARGETS],
                      dtype=np.float32)
    rot = np.array([cos_a, sin_a], dtype=np.float32)
    return np.concatenate([pts.flatten(), extra, rot])


def mirror_vector(vec: np.ndarray) -> np.ndarray:
    """Mirrors along the x-axis for augmenting both hands. Thumb distances are
    invariant to mirroring. The sin component of the angle flips sign, cos does not."""
    out = vec.copy()
    out[0:BASE_DIM:3] *= -1.0
    if out.shape[0] >= 2:
        out[-1] *= -1.0
    return out


def resample_sequence(seq: np.ndarray, target_len: int = 30) -> np.ndarray:
    """Linear interpolation of a sequence to a fixed length (GRU needs a fixed
    number of frames). Identical function in web/js/normalize.js."""
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    if T == target_len:
        return seq
    if T == 1:
        return np.repeat(seq, target_len, axis=0)
    src_idx = np.linspace(0.0, T - 1.0, target_len)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (src_idx - lo)[:, None]
    return seq[lo] * (1.0 - frac) + seq[hi] * frac


def jitter(vec63: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """Gaussian noise, simulates hand tremor / detection noise."""
    return vec63 + np.random.normal(0.0, sigma, vec63.shape).astype(np.float32)
