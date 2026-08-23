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
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP = 5, 6
MIDDLE_MCP = 9     # korijen srednjeg prsta (metacarpophalangeal zglob)
MIDDLE_PIP = 10
RING_MCP, RING_PIP = 13, 14
PINKY_MCP = 17

# Zglobovi čija udaljenost od vrha palca eksplicitno kodira POLOŽAJ PALCA —
# glavni razlikovni detalj kod A/S/T (palac uz kažiprst / preko šake / između
# kažiprsta i srednjaka) i M/N (palac ispod 3 ili 2 prsta). Sirove xyz
# koordinate tu razliku nose posredno kroz 63 vrijednosti; ove udaljenosti je
# čine izravnim, snažnim signalom modelu.
THUMB_DIST_TARGETS = [INDEX_MCP, INDEX_PIP, MIDDLE_MCP, MIDDLE_PIP,
                      RING_MCP, RING_PIP, PINKY_MCP]
BASE_DIM = 63
# Zadnja 2 elementa vektora: (cos, sin) IZVORNOG kuta nagiba šake prije
# ispravka rotacije (vidi niže) — G/Q, H/U, K/P razlikuju se ISKLJUČIVO
# rotacijom cijele šake, pa rotacijski ispravak koji čini ostatak vektora
# rotacijski-invarijantnim upravo TU informaciju briše. Ova dva broja je
# vraćaju natrag, eksplicitno, bez da kvare invarijantnost ostatka vektora.
FEATURE_DIM = BASE_DIM + len(THUMB_DIST_TARGETS) + 2  # 72


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    1. Translacija: zapešće u ishodište -> neovisnost o poziciji ruke u kadru.
    2. Rotacija (u ravnini slike): zapešće->korijen srednjeg prsta poravna se
       s referentnim smjerom (0, -1) -> neovisnost o naginjanju/rotaciji šake
       u kadru (bez ovoga isti znak pod drugim kutom drukčije "izgleda" modelu,
       što je osobito štetno za slova koja se razlikuju samo položajem palca
       ili stupnjem savijenosti prstiju).
    3. Skaliranje: dijeljenje udaljenošću zapešće->korijen srednjeg prsta
       -> neovisnost o udaljenosti ruke od kamere i veličini šake.
    4. Dodatne značajke: euklidske udaljenosti vrha palca do zglobova
       kažiprsta/srednjaka/prstenjaka/malog prsta (vidi THUMB_DIST_TARGETS)
       — eksplicitno kodiraju položaj palca.
    5. (cos, sin) izvornog kuta nagiba šake — vraća signal koji je korak 2.
       namjerno uklonio iz ostatka vektora, za slova koja se razlikuju
       ROTACIJOM cijele šake (G/Q, H/U, K/P), a ne oblikom prstiju.
    Vraća ravni vektor od FEATURE_DIM (72) float vrijednosti.
    """
    pts = landmarks.astype(np.float32).copy()
    pts -= pts[WRIST]                       # translacija

    mx, my = pts[MIDDLE_MCP][0], pts[MIDDLE_MCP][1]
    r_xy = float(np.hypot(mx, my))
    if r_xy > 1e-6:
        cos_a, sin_a = -my / r_xy, mx / r_xy  # rotacija koja mx,my šalje na (0, -r_xy)
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0] = cos_a * x + sin_a * y
        pts[:, 1] = -sin_a * x + cos_a * y
        # z ostaje netaknut — MediaPipeova z-koordinata je gruba procjena
        # dubine i nije pouzdana osnova za rotaciju izvan ravnine slike.
    else:
        cos_a, sin_a = 1.0, 0.0

    scale = np.linalg.norm(pts[MIDDLE_MCP]) # duljina "dlana"
    if scale < 1e-6:
        scale = 1.0
    pts /= scale                            # skaliranje

    thumb = pts[THUMB_TIP]
    extra = np.array([np.linalg.norm(thumb - pts[i]) for i in THUMB_DIST_TARGETS],
                      dtype=np.float32)
    rot = np.array([cos_a, sin_a], dtype=np.float32)
    return np.concatenate([pts.flatten(), extra, rot])  # (72,)


def mirror_vector(vec: np.ndarray) -> np.ndarray:
    """
    Zrcali normalizirani vektor po x-osi (x -> -x).
    Koristi se za augmentaciju: model tako nauči i lijevu i desnu ruku,
    pa u aplikaciji ne moramo uopće gledati MediaPipe 'handedness' oznaku.
    Napomena: udaljenosti (indeksi 63..69) su invarijantne na zrcaljenje pa se
    ne diraju. Zadnja dva elementa (cos, sin) JESU osjetljiva na zrcaljenje:
    zrcaljenje po x-osi drži cos isti, a mijenja predznak sin (kut se reflektira
    oko okomite osi).
    """
    out = vec.copy()
    out[0:BASE_DIM:3] *= -1.0   # svaka treća vrijednost je x koordinata
    if out.shape[0] >= 2:
        out[-1] *= -1.0         # sin komponenta kuta nagiba
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
