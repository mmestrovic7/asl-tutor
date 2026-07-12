/**
 * normalize.js
 * Normalizacija landmarkova — MORA biti identična training/landmark_utils.py!
 * Ulaz: MediaPipe polje od 21 landmarka {x, y, z}.
 * Izlaz: Float32Array(63).
 */

const WRIST = 0;
const MIDDLE_MCP = 9;

export function normalizeLandmarks(landmarks) {
  const out = new Float32Array(63);
  const wx = landmarks[WRIST].x, wy = landmarks[WRIST].y, wz = landmarks[WRIST].z;

  // 1) translacija: zapešće u ishodište
  for (let i = 0; i < 21; i++) {
    out[i * 3]     = landmarks[i].x - wx;
    out[i * 3 + 1] = landmarks[i].y - wy;
    out[i * 3 + 2] = landmarks[i].z - wz;
  }
  // 2) skaliranje: udaljenost zapešće -> korijen srednjeg prsta
  const mx = out[MIDDLE_MCP * 3], my = out[MIDDLE_MCP * 3 + 1], mz = out[MIDDLE_MCP * 3 + 2];
  let scale = Math.hypot(mx, my, mz);
  if (scale < 1e-6) scale = 1.0;
  for (let i = 0; i < 63; i++) out[i] /= scale;
  return out;
}

/** Prosječni apsolutni pomak između dva normalizirana vektora ("energija pokreta"). */
export function motionEnergy(a, b) {
  if (!a || !b) return 0;
  let s = 0;
  for (let i = 0; i < 63; i++) s += Math.abs(a[i] - b[i]);
  return s / 63;
}

/** Linearno resampliranje sekvence [T][63] na točno targetLen frameova.
 *  Identično resample_sequence() u landmark_utils.py. */
export function resampleSequence(seq, targetLen = 30) {
  const T = seq.length;
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
    const f = new Float32Array(63);
    for (let k = 0; k < 63; k++) f[k] = seq[lo][k] * (1 - frac) + seq[hi][k] * frac;
    out.push(f);
  }
  return out;
}
