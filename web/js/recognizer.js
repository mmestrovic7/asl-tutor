/**
 * recognizer.js
 * Srce sustava: stroj stanja koji odlučuje kada raditi statičku, a kada
 * dinamičku klasifikaciju, te kada je znak "službeno" prepoznat.
 *
 * Stanja:
 *   STILL  — ruka miruje -> svaki frame ide u MLP (statička slova).
 *            Slovo je prihvaćeno kad je isto slovo predviđeno s
 *            pouzdanošću >= CONF kroz STABLE_FRAMES uzastopnih frameova.
 *   MOVING — energija pokreta prešla prag -> frameovi se skupljaju u buffer.
 *            Kad se pokret smiri, buffer se resamplira na 30 frameova i
 *            šalje GRU modelu (J / Z / OTHER).
 */

import { motionEnergy, resampleSequence, FEATURE_DIM } from "./normalize.js?v=4";

const CONF_STATIC   = 0.80;  // min. pouzdanost statičkog slova
const CONF_DYNAMIC  = 0.75;  // min. pouzdanost J/Z
const STABLE_FRAMES = 15;    // ~0.5 s stabilnog držanja (na 30 fps)
const MOTION_HI     = 0.030; // ulazak u MOVING (energija, EMA)
const MOTION_LO     = 0.014; // izlazak iz MOVING
const STILL_END     = 10;    // frameova mirovanja koji zatvaraju gestu
const MIN_SEQ       = 8;
const MAX_SEQ       = 90;
const PREROLL       = 5;     // koliko zadnjih mirnih frameova ubaciti na početak geste

export class Recognizer {
  constructor({ staticModel, staticLabels, dynModel, dynLabels }) {
    this.staticModel = staticModel;
    this.staticLabels = staticLabels || [];
    this.dynModel = dynModel;
    this.dynLabels = dynLabels || [];
    this.reset();
  }

  reset() {
    this.state = "STILL";
    this.lastVec = null;
    this.energyEMA = 0;
    this.stableLetter = null;
    this.stableCount = 0;
    this.seq = [];
    this.preroll = [];
    this.stillCount = 0;
    this.cooldown = false;   // nakon prepoznavanja čekamo promjenu znaka
  }

  /** Poziva se svaki frame. vec = Float32Array(63) ili null (nema ruke).
   *  Vraća objekt stanja koji UI koristi za prikaz. */
  update(vec) {
    if (!vec) {
      const hadHand = this.lastVec !== null;
      this.reset();
      return { hand: false, state: "NOHAND", handWasRemoved: hadHand };
    }

    const e = motionEnergy(vec, this.lastVec);
    this.energyEMA = this.lastVec ? 0.6 * this.energyEMA + 0.4 * e : 0;
    this.lastVec = vec;

    // -- ring buffer mirnih frameova (preroll za početak geste) --
    this.preroll.push(vec);
    if (this.preroll.length > PREROLL) this.preroll.shift();

    if (this.state === "STILL") {
      if (this.energyEMA > MOTION_HI) {
        // pokret počeo -> prelazak u dinamički mod
        this.state = "MOVING";
        this.seq = [...this.preroll];
        this.stillCount = 0;
        this.stableLetter = null;
        this.stableCount = 0;
        return { hand: true, state: "MOVING", seqLen: this.seq.length };
      }
      return this.#stepStatic(vec);
    }

    // ---- MOVING ----
    this.seq.push(vec);
    if (this.energyEMA < MOTION_LO) this.stillCount++;
    else this.stillCount = 0;

    if (this.stillCount >= STILL_END || this.seq.length >= MAX_SEQ) {
      const result = this.#classifyDynamic();
      this.state = "STILL";
      this.seq = [];
      if (result) return { hand: true, state: "STILL", dynamic: result };
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
      this.cooldown = false; // znak se promijenio -> smije se prihvatiti novi
    }

    const progress = Math.min(this.stableCount / STABLE_FRAMES, 1);
    const out = { hand: true, state: "STILL", letter: this.stableLetter, prob, progress };

    if (progress >= 1 && !this.cooldown) {
      this.cooldown = true;        // isti znak se ne broji dvaput zaredom
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
    if ((label === "J" || label === "Z") && prob >= CONF_DYNAMIC) {
      return { letter: label, kind: "dynamic", prob };
    }
    return null; // OTHER ili premala pouzdanost -> ignoriraj
  }
}

/** Učitavanje TF.js modela; vraća null ako model (još) ne postoji. */
export async function tryLoadModel(baseUrl) {
  try {
    const model = await tf.loadLayersModel(`${baseUrl}/model.json`);
    const labels = await (await fetch(`${baseUrl}/labels.json`)).json();
    return { model, labels };
  } catch {
    return null;
  }
}
