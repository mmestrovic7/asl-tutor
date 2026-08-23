/**
 * normalize.js
 * Normalizacija landmarkova — MORA biti identična training/landmark_utils.py!
 * Ulaz: MediaPipe polje od 21 landmarka {x, y, z}.
 * Izlaz: Float32Array(FEATURE_DIM).
 */

const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_MCP = 5, INDEX_PIP = 6;
const MIDDLE_MCP = 9;
const MIDDLE_PIP = 10;
const RING_MCP = 13, RING_PIP = 14;
const PINKY_MCP = 17;

// Zglobovi čija udaljenost od vrha palca eksplicitno kodira POLOŽAJ PALCA —
// glavni razlikovni detalj kod A/S/T i M/N. MORA biti identično
// THUMB_DIST_TARGETS u training/landmark_utils.py (isti redoslijed!).
const THUMB_DIST_TARGETS = [INDEX_MCP, INDEX_PIP, MIDDLE_MCP, MIDDLE_PIP,
                            RING_MCP, RING_PIP, PINKY_MCP];
export const BASE_DIM = 63;
// Zadnja 2 elementa vektora: (cos, sin) IZVORNOG kuta nagiba šake prije
// ispravka rotacije — vidi napomenu u training/landmark_utils.py. G/Q, H/U,
// K/P razlikuju se ISKLJUČIVO rotacijom cijele šake, pa ova dva broja vraćaju
// signal koji bi inače rotacijski ispravak (korak 2. niže) izbrisao.
export const FEATURE_DIM = BASE_DIM + THUMB_DIST_TARGETS.length + 2; // 72

export function normalizeLandmarks(landmarks) {
  const base = new Float32Array(BASE_DIM);
  const wx = landmarks[WRIST].x, wy = landmarks[WRIST].y, wz = landmarks[WRIST].z;

  // 1) translacija: zapešće u ishodište
  for (let i = 0; i < 21; i++) {
    base[i * 3]     = landmarks[i].x - wx;
    base[i * 3 + 1] = landmarks[i].y - wy;
    base[i * 3 + 2] = landmarks[i].z - wz;
  }

  // 2) rotacija (u ravnini slike): zapešće->korijen srednjeg prsta poravna
  // se s (0, -1), tako da naginjanje šake u kadru ne mijenja ulazni vektor.
  // MORA biti identično rotaciji u training/landmark_utils.py.
  const mx0 = base[MIDDLE_MCP * 3], my0 = base[MIDDLE_MCP * 3 + 1];
  const rxy = Math.hypot(mx0, my0);
  let cosA = 1.0, sinA = 0.0;
  if (rxy > 1e-6) {
    cosA = -my0 / rxy; sinA = mx0 / rxy;
    for (let i = 0; i < 21; i++) {
      const x = base[i * 3], y = base[i * 3 + 1];
      base[i * 3]     = cosA * x + sinA * y;
      base[i * 3 + 1] = -sinA * x + cosA * y;
      // z ostaje netaknut
    }
  }

  // 3) skaliranje: udaljenost zapešće -> korijen srednjeg prsta
  const mx = base[MIDDLE_MCP * 3], my = base[MIDDLE_MCP * 3 + 1], mz = base[MIDDLE_MCP * 3 + 2];
  let scale = Math.hypot(mx, my, mz);
  if (scale < 1e-6) scale = 1.0;
  for (let i = 0; i < BASE_DIM; i++) base[i] /= scale;

  // 4) dodatne značajke: udaljenosti vrha palca do zglobova ostalih prstiju
  const out = new Float32Array(FEATURE_DIM);
  out.set(base);
  const tx = base[THUMB_TIP * 3], ty = base[THUMB_TIP * 3 + 1], tz = base[THUMB_TIP * 3 + 2];
  THUMB_DIST_TARGETS.forEach((idx, k) => {
    const dx = tx - base[idx * 3], dy = ty - base[idx * 3 + 1], dz = tz - base[idx * 3 + 2];
    out[BASE_DIM + k] = Math.hypot(dx, dy, dz);
  });
  // 5) (cos, sin) izvornog kuta nagiba šake
  out[BASE_DIM + THUMB_DIST_TARGETS.length] = cosA;
  out[BASE_DIM + THUMB_DIST_TARGETS.length + 1] = sinA;
  return out;
}

/** Prosječni apsolutni pomak između dva normalizirana vektora ("energija pokreta").
 *  Računa se SAMO nad BASE_DIM (položaji zglobova) — dodatne udaljinske
 *  značajke (63+) imaju veći i nestabilniji raspon vrijednosti pa bi
 *  napuhale energiju i pokvarile pragove MOTION_HI/MOTION_LO ugođene za
 *  poziciski raspon. */
export function motionEnergy(a, b) {
  if (!a || !b) return 0;
  let s = 0;
  for (let i = 0; i < BASE_DIM; i++) s += Math.abs(a[i] - b[i]);
  return s / BASE_DIM;
}

/** Linearno resampliranje sekvence [T][D] na točno targetLen frameova.
 *  Identično resample_sequence() u landmark_utils.py. */
export function resampleSequence(seq, targetLen = 30) {
  const T = seq.length;
  const D = seq[0].length;
  if (T === targetLen) return seq;
  const out = [];
  if (T === 1) {
    for (let i = 0; i < targetLen; i++) out.push(Float32Array.from(seq[0]));
    return out;
  }
  for (let i = 0; i < targetLen; i++) {
    const src = (i * (T - 1)) / (targetLen - 1);
    const lo = Math.floor(src);
    const hi = Math.min(lo + 1, T - 1);
    const frac = src - lo;
    const f = new Float32Array(D);
    for (let k = 0; k < D; k++) f[k] = seq[lo][k] * (1 - frac) + seq[hi][k] * frac;
    out.push(f);
  }
  return out;
}
