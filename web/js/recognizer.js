/**
 * recognizer.js
 * Stroj stanja koji odlučuje kada raditi statičku, a kada dinamičku
 * klasifikaciju, i kada je znak "službeno" prepoznat.
 *
 * STILL  - ruka miruje, svaki frame ide u MLP (statička slova). Slovo se
 *          prihvaća kad je isto slovo predviđeno s pouzdanošću >= CONF_STATIC
 *          kroz STABLE_FRAMES uzastopnih frameova.
 * MOVING - energija pokreta prešla prag, frameovi se skupljaju u buffer.
 *          Kad se pokret smiri, buffer ide u GRU model (J / Z / OTHER).
 */

import { motionEnergy, resampleSequence, FEATURE_DIM } from "./normalize.js?v=12";

const CONF_STATIC   = 0.80;
const CONF_DYNAMIC  = 0.88; // 0.75 je davao previše lažnih pogodaka
const STABLE_FRAMES = 15;   // ~0.5s pri 30fps
const MOTION_HI     = 0.030; // ulazak u MOVING, energija oblika šake (EMA)
const MOTION_LO     = 0.014; // izlazak iz MOVING

// normalizeLandmarks() briše apsolutnu poziciju zapešća, ali J/Z se
// definiraju baš tim kretanjem kroz prostor: energija oblika šake ostaje
// prenisko kroz cijelu gestu da bi sama pokrenula MOVING. Zato pratimo i
// sirovu (nenormaliziranu) poziciju zapešća kao drugi, neovisan okidač.
const RAW_MOTION_HI = 0.010;
const RAW_MOTION_LO = 0.004;
const STILL_END     = 10;   // frameova mirovanja koji zatvaraju gestu
const MIN_SEQ       = 8;
const MAX_SEQ       = 90;
const PREROLL       = 5;    // zadnjih mirnih frameova ubačenih na početak geste

// MediaPipe zna izgubiti ruku na trenutak usred brzog pokreta (motion blur,
// ekstremni kut); bez tolerancije jedan promašeni frame briše cijeli buffer.
const DROPOUT_GRACE = 4;

// Rep pokreta nakon prepoznatog J/Z zna odmah okinuti novi MOVING i
// pogrešno se klasificirati kao još jedan J/Z. Cooldown to sprječava.
const DYNAMIC_COOLDOWN_FRAMES = 20;

// Prirodno drhtanje ruke (npr. neudobna poza kod P) zna na jedan frame
// probiti prag pokreta. Tražimo nekoliko uzastopnih frameova preko praga
// prije ulaska u MOVING, da trzaj ne prekine STILL brojanje.
const MOTION_STREAK_REQUIRED = 3;

export class Recognizer {
  constructor({ staticModel, staticLabels, dynModel, dynLabels }) {
    this.staticModel = staticModel;
    this.staticLabels = staticLabels || [];
    this.dynModel = dynModel;
    this.dynLabels = dynLabels || [];
    this.dynamicCooldown = 0; // nije dio reset(), mora preživjeti kratkotrajne resete
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

  /** Poziva se svaki frame. vec = Float32Array ili null (nema ruke).
   *  rawWrist = {x, y} sirova pozicija zapešća, samo za detekciju pokreta,
   *  model i dalje prima isključivo normalizirani vec. */
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

  /** Vraća null samo ako klasifikacija nije ni pokušana (nema modela ili
   *  je sekvenca prekratka). Inače uvijek vraća `matched`, da pozivatelj
   *  može razlikovati "prepoznato" od "pokušano, nije prepoznato". */
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

/** Učitavanje TF.js modela, vraća null ako model (još) ne postoji.
 *  cache: "no-store" na svim fetchevima jer browser inače zna zadržati
 *  stari model.json/.bin nakon retreniranja. */
export async function tryLoadModel(baseUrl) {
  try {
    const model = await tf.loadLayersModel(`${baseUrl}/model.json`,
      { requestInit: { cache: "no-store" } });
    const labels = await (await fetch(`${baseUrl}/labels.json`, { cache: "no-store" })).json();
    return { model, labels };
  } catch (e) {
    console.warn(`Neuspjelo učitavanje modela iz ${baseUrl}:`, e);
    return null;
  }
}
