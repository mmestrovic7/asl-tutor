/**
 * recognizer.js
 * State machine that decides when to run static vs. dynamic
 * classification, and when a sign is "officially" recognized.
 *
 * STILL  - the hand is still, every frame goes into the MLP (static letters).
 *          A letter is accepted once the same letter is predicted with
 *          confidence >= CONF_STATIC for STABLE_FRAMES consecutive frames.
 * MOVING - motion energy crossed the threshold, frames are collected into
 *          a buffer. Once the motion settles, the buffer goes into the
 *          GRU model (J / Z / OTHER).
 */

import { motionEnergy, resampleSequence, FEATURE_DIM } from "./normalize.js?v=12";

const CONF_STATIC   = 0.80;
const CONF_DYNAMIC  = 0.88; // 0.75 gave too many false positives
const STABLE_FRAMES = 15;   // ~0.5s at 30fps
const MOTION_HI     = 0.030; // entering MOVING, hand-shape energy (EMA)
const MOTION_LO     = 0.014; // leaving MOVING

// normalizeLandmarks() discards the absolute wrist position, but J/Z are
// defined precisely by that movement through space: hand-shape energy stays
// too low throughout the whole gesture to trigger MOVING by itself. So we
// also track the raw (non-normalized) wrist position as a second, independent
// trigger.
const RAW_MOTION_HI = 0.010;
const RAW_MOTION_LO = 0.004;
const STILL_END     = 10;   // stillness frames that close out a gesture
const MIN_SEQ       = 8;
const MAX_SEQ       = 90;
const PREROLL       = 5;    // last still frames prepended to the start of a gesture

// MediaPipe can lose the hand for a moment mid fast-motion (motion blur,
// extreme angle); without tolerance, one missed frame would wipe the whole buffer.
const DROPOUT_GRACE = 4;

// The tail of a motion after a recognized J/Z can immediately trigger a new
// MOVING and get misclassified as another J/Z. The cooldown prevents that.
const DYNAMIC_COOLDOWN_FRAMES = 20;

// Natural hand tremor (e.g. an uncomfortable pose for P) can briefly cross
// the motion threshold for a single frame. We require several consecutive
// frames over the threshold before entering MOVING, so a jitter doesn't
// interrupt STILL counting.
const MOTION_STREAK_REQUIRED = 3;

export class Recognizer {
  constructor({ staticModel, staticLabels, dynModel, dynLabels }) {
    this.staticModel = staticModel;
    this.staticLabels = staticLabels || [];
    this.dynModel = dynModel;
    this.dynLabels = dynLabels || [];
    this.dynamicCooldown = 0; // not part of reset(), must survive short-lived resets
    this.reset();
  }

  reset() {
    this.state = "STILL";
    this.lastVec = null;
    this.energyEMA = 0;
    this.lastRawWrist = null;
    this.rawEnergyEMA = 0;
    this.stableLetter = null;
    this.stableCount = 0;
    this.seq = [];
    this.preroll = [];
    this.stillCount = 0;
    this.missingFrames = 0;
    this.motionStreak = 0;
    this.cooldown = false;
  }

  /** Called every frame. vec = Float32Array or null (no hand).
   *  rawWrist = {x, y} raw wrist position, used only for motion detection,
   *  the model still receives only the normalized vec. */
  update(vec, rawWrist = null) {
    if (this.dynamicCooldown > 0) this.dynamicCooldown--;
    if (!vec) {
      const hadHand = this.lastVec !== null;
      this.missingFrames++;
      if (this.state === "MOVING" && this.missingFrames <= DROPOUT_GRACE) {
        return { hand: true, state: "MOVING", seqLen: this.seq.length };
      }
      if (this.state === "MOVING" && this.seq.length >= MIN_SEQ) {
        const result = this.#classifyDynamic();
        this.reset();
        if (result) {
          this.dynamicCooldown = DYNAMIC_COOLDOWN_FRAMES;
          if (result.matched) {
            return { hand: false, state: "NOHAND", handWasRemoved: hadHand, dynamic: result };
          }
          return { hand: false, state: "NOHAND", handWasRemoved: hadHand, dynamicMiss: true };
        }
      } else {
        this.reset();
      }
      return { hand: false, state: "NOHAND", handWasRemoved: hadHand };
    }
    this.missingFrames = 0;

    const e = motionEnergy(vec, this.lastVec);
    this.energyEMA = this.lastVec ? 0.6 * this.energyEMA + 0.4 * e : 0;
    this.lastVec = vec;

    const rawE = (rawWrist && this.lastRawWrist)
      ? Math.hypot(rawWrist.x - this.lastRawWrist.x, rawWrist.y - this.lastRawWrist.y)
      : 0;
    this.rawEnergyEMA = this.lastRawWrist ? 0.6 * this.rawEnergyEMA + 0.4 * rawE : 0;
    this.lastRawWrist = rawWrist;

    this.preroll.push(vec);
    if (this.preroll.length > PREROLL) this.preroll.shift();

    if (this.state === "STILL") {
      const overThreshold = this.dynamicCooldown === 0 &&
        (this.energyEMA > MOTION_HI || this.rawEnergyEMA > RAW_MOTION_HI);
      this.motionStreak = overThreshold ? this.motionStreak + 1 : 0;

      if (this.motionStreak >= MOTION_STREAK_REQUIRED) {
        this.state = "MOVING";
        this.seq = [...this.preroll];
        this.stillCount = 0;
        this.motionStreak = 0;
        this.stableLetter = null;
        this.stableCount = 0;
        return { hand: true, state: "MOVING", seqLen: this.seq.length };
      }
      return this.#stepStatic(vec);
    }

    // ---- MOVING ----
    this.seq.push(vec);
    if (this.energyEMA < MOTION_LO && this.rawEnergyEMA < RAW_MOTION_LO) this.stillCount++;
    else this.stillCount = 0;

    if (this.stillCount >= STILL_END || this.seq.length >= MAX_SEQ) {
      const result = this.#classifyDynamic();
      this.state = "STILL";
      this.seq = [];
      if (result) {
        this.dynamicCooldown = DYNAMIC_COOLDOWN_FRAMES;
        if (result.matched) return { hand: true, state: "STILL", dynamic: result };
        return { hand: true, state: "STILL", dynamicMiss: true };
      }
      return { hand: true, state: "STILL" };
    }
    return { hand: true, state: "MOVING", seqLen: this.seq.length };
  }

  #stepStatic(vec) {
    if (!this.staticModel) return { hand: true, state: "STILL", demo: true };

    const { letter, prob } = this.#predictStatic(vec);

    if (letter === this.stableLetter && prob >= CONF_STATIC) {
      this.stableCount++;
    } else {
      this.stableLetter = prob >= CONF_STATIC ? letter : null;
      this.stableCount = this.stableLetter ? 1 : 0;
      this.cooldown = false;
    }

    const progress = Math.min(this.stableCount / STABLE_FRAMES, 1);
    const out = { hand: true, state: "STILL", letter: this.stableLetter, prob, progress };

    if (progress >= 1 && !this.cooldown) {
      this.cooldown = true;
      out.recognized = { letter: this.stableLetter, kind: "static", prob };
    }
    return out;
  }

  #predictStatic(vec) {
    return tf.tidy(() => {
      const t = tf.tensor2d(vec, [1, FEATURE_DIM]);
      const p = this.staticModel.predict(t).dataSync();
      let best = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
      return { letter: this.staticLabels[best], prob: p[best] };
    });
  }

  /** Returns null only if classification wasn't even attempted (no model or
   *  the sequence is too short). Otherwise it always returns `matched`, so
   *  the caller can distinguish "recognized" from "attempted, not recognized". */
  #classifyDynamic() {
    if (!this.dynModel || this.seq.length < MIN_SEQ) return null;
    const rs = resampleSequence(this.seq, 30);
    const flat = new Float32Array(30 * FEATURE_DIM);
    rs.forEach((f, i) => flat.set(f, i * FEATURE_DIM));
    const { label, prob } = tf.tidy(() => {
      const t = tf.tensor3d(flat, [1, 30, FEATURE_DIM]);
      const p = this.dynModel.predict(t).dataSync();
      let best = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
      return { label: this.dynLabels[best], prob: p[best] };
    });
    const matched = (label === "J" || label === "Z") && prob >= CONF_DYNAMIC;
    return { letter: label, kind: "dynamic", prob, matched };
  }
}

/** Loads a TF.js model, returns null if the model doesn't exist (yet).
 *  cache: "no-store" on all fetches because the browser would otherwise
 *  keep the old model.json/.bin after retraining. */
export async function tryLoadModel(baseUrl) {
  try {
    const model = await tf.loadLayersModel(`${baseUrl}/model.json`,
      { requestInit: { cache: "no-store" } });
    const labels = await (await fetch(`${baseUrl}/labels.json`, { cache: "no-store" })).json();
    return { model, labels };
  } catch (e) {
    console.warn(`Failed to load model from ${baseUrl}:`, e);
    return null;
  }
}
