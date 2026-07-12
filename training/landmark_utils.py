"""
landmark_utils.py
-----------------
Zajedničke funkcije za normalizaciju MediaPipe landmarkova.

VAŽNO: funkcija normalize_landmarks() mora biti MATEMATIČKI IDENTIČNA
funkciji u web/js/normalize.js. Svaka promjena ovdje zahtijeva promjenu i tamo,
inače će model u browseru dobivati drugačije ulaze nego na treningu.

Format landmarkova: numpy array oblika (21, 3) — 21 točka šake, svaka (x, y, z).
MediaPipe daje koordinate normalizirane na dimenzije slike (0..1).
"""

import numpy as np

WRIST = 0          # zapešće
MIDDLE_MCP = 9     # korijen srednjeg prsta (metacarpophalangeal zglob)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    1. Translacija: zapešće u ishodište -> neovisnost o poziciji ruke u kadru.
    2. Skaliranje: dijeljenje udaljenošću zapešće->korijen srednjeg prsta
       -> neovisnost o udaljenosti ruke od kamere i veličini šake.
    Vraća ravni vektor od 63 float vrijednosti.
    """
    pts = landmarks.astype(np.float32).copy()
    pts -= pts[WRIST]                       # translacija
    scale = np.linalg.norm(pts[MIDDLE_MCP]) # duljina "dlana"
    if scale < 1e-6:
        scale = 1.0
    pts /= scale                            # skaliranje
    return pts.flatten()                    # (63,)


def mirror_vector(vec63: np.ndarray) -> np.ndarray:
    """
    Zrcali normalizirani vektor po x-osi (x -> -x).
    Koristi se za augmentaciju: model tako nauči i lijevu i desnu ruku,
    pa u aplikaciji ne moramo uopće gledati MediaPipe 'handedness' oznaku.
    """
    out = vec63.copy()
    out[0::3] *= -1.0   # svaka treća vrijednost je x koordinata
    return out


def resample_sequence(seq: np.ndarray, target_len: int = 30) -> np.ndarray:
    """
    Linearno interpolira sekvencu vektora (T, 63) na fiksnu duljinu (target_len, 63).
    Time GRU model uvijek dobiva jednak broj frameova, neovisno o tome
    koliko je dugo trajao pokret (netko J napravi za 0.4 s, netko za 1.2 s).
    Identična funkcija postoji u web/js/normalize.js (resampleSequence).
    """
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
    """Blagi Gaussov šum — augmentacija koja simulira drhtanje ruke / šum detekcije."""
    return vec63 + np.random.normal(0.0, sigma, vec63.shape).astype(np.float32)
