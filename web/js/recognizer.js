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

import { motionEnergy, resampleSequence, FEATURE_DIM } from "./normalize.js?v=12";

const CONF_STATIC   = 0.80;  // min. pouzdanost statičkog slova
const CONF_DYNAMIC  = 0.88;  // min. pouzdanost J/Z (podignuto — previše lažnih pogodaka na 0.75)
const STABLE_FRAMES = 15;    // ~0.5 s stabilnog držanja (na 30 fps)
const MOTION_HI     = 0.030; // ulazak u MOVING (energija oblika šake, EMA)
const MOTION_LO     = 0.014; // izlazak iz MOVING (energija oblika šake)
// normalize_landmarks() namjerno briše apsolutnu poziciju zapešća (translacija
// na ishodište) da bi MLP prepoznavao oblik šake neovisno o poziciji u kadru.
// Ali J/Z se DEFINIRAJU upravo tom pozicijom kroz prostor (ruka crta luk) —
// izmjereno na stvarnim snimkama J-a, energija oblika šake ostaje ~3x ispod
// MOTION_HI kroz cijelu gestu, pa se MOVING stanje često uopće ne aktivira.
// Zato pratimo i SIROVU poziciju zapešća (prije normalizacije) kao drugi,
// neovisan signal pokreta — bilo koji od dva praga smije pokrenuti MOVING.
const RAW_MOTION_HI = 0.010; // ulazak u MOVING (sirovi pomak zapešća, EMA)
const RAW_MOTION_LO = 0.004; // izlazak iz MOVING (sirovi pomak zapešća)
const STILL_END     = 10;    // frameova mirovanja koji zatvaraju gestu
const MIN_SEQ       = 8;
const MAX_SEQ       = 90;
const PREROLL       = 5;     // koliko zadnjih mirnih frameova ubaciti na početak geste
// MediaPipe zna na trenutak izgubiti ruku usred brzog pokreta (motion blur,
// ekstremni kut) — bez ovoga jedan promašeni frame odmah briše cijeli buffer
// geste, pa MOVING stalno iznova kreće a nikad ne stigne do klasifikacije.
const DROPOUT_GRACE = 4;     // frameova bez detekcije koje toleriramo usred MOVING
// Nakon prepoznatog J/Z ruka se često još nastavlja spuštati/smirivati —
// bez ove pauze taj rep pokreta zna odmah okinuti NOVO MOVING i model ga
// pogrešno klasificira kao još jedan J/Z (analogno "cooldown" polju kod
// statičkih slova, koje sprječava dvostruko brojanje istog znaka).
const DYNAMIC_COOLDOWN_FRAMES = 20;
// Prirodno drhtanje ruke dok se drži statičko slovo (osobito neudobne poze
// poput P, gdje je ruka okrenuta prema dolje) zna trenutačno probiti
// RAW_MOTION_HI/MOTION_HI prag na samo jednom frameu — bez ovoga to odmah
// prekida STILL brojanje (resetira stableCount) i statičko slovo se nikad
// ne stigne potvrditi. Zahtijevamo da prag bude probijen nekoliko frameova
// zaredom prije nego stvarno uđemo u MOVING (namjeran pokret traje dulje
// od trenutačnog trzaja).
const MOTION_STREAK_REQUIRED = 3;

export class Recognizer {
  constructor({ staticModel, staticLabels, dynModel, dynLabels }) {
    this.staticModel = staticModel;
    this.staticLabels = staticLabels || [];
    this.dynModel = dynModel;
    this.dynLabels = dynLabels || [];
    this.dynamicCooldown = 0; // NIJE dio reset() — mora preživjeti kratkotrajne resete
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
    this.cooldown = false;   // nakon prepoznavanja čekamo promjenu znaka
  }

  /** Poziva se svaki frame. vec = Float32Array(63) ili null (nema ruke).
   *  rawWrist = {x, y} sirova (nenormalizirana) pozicija zapešća u kadru,
   *  ili null. Koristi se SAMO za detekciju pokreta (MOVING okidač) — model
   *  i dalje prima isključivo normalizirani vec (vidi komentar uz RAW_MOTION_HI).
   *  Vraća objekt stanja koji UI koristi za prikaz. */
  update(vec, rawWrist = null) {
    if (this.dynamicCooldown > 0) this.dynamicCooldown--;
    if (!vec) {
      const hadHand = this.lastVec !== null;
      this.missingFrames++;
      // Usred geste toleriramo par uzastopnih frameova bez detekcije
      // (vidi DROPOUT_GRACE) — samo preskačemo frame, buffer ostaje netaknut.
      if (this.state === "MOVING" && this.missingFrames <= DROPOUT_GRACE) {
        return { hand: true, state: "MOVING", seqLen: this.seq.length };
      }
      // Prekoračena tolerancija (ili ruka nestala izvan MOVING) — ako je
      // sekvenca dovoljno duga, tretiraj to kao prirodan kraj geste i
      // pokušaj klasifikaciju umjesto da tiho odbacimo cijeli buffer.
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

    // -- ring buffer mirnih frameova (preroll za početak geste) --
    this.preroll.push(vec);
    if (this.preroll.length > PREROLL) this.preroll.shift();

    if (this.state === "STILL") {
      const overThreshold = this.dynamicCooldown === 0 &&
        (this.energyEMA > MOTION_HI || this.rawEnergyEMA > RAW_MOTION_HI);
      this.motionStreak = overThreshold ? this.motionStreak + 1 : 0;

      if (this.motionStreak >= MOTION_STREAK_REQUIRED) {
        // pokret počeo -> prelazak u dinamički mod
        // DEBUG (privremeno): brojke za kalibraciju MOTION_HI/RAW_MOTION_HI.
        console.log("[motion] MOVING start", `shapeE=${this.energyEMA.toFixed(4)}`, `rawE=${this.rawEnergyEMA.toFixed(4)}`);
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

  /** Vraća null SAMO ako klasifikacija uopće nije ni pokušana (nema modela ili
   *  je sekvenca prekratka). Inače uvijek vraća objekt s `matched` — pozivatelj
   *  na temelju toga razlikuje "prepoznato" od "pokušano, ali nije prepoznato"
   *  (korisniku treba povratna informacija i za neuspjeh, ne samo za uspjeh). */
  #classifyDynamic() {
    if (!this.dynModel || this.seq.length < MIN_SEQ) return null;
    const rs = resampleSequence(this.seq, 30);
    const flat = new Float32Array(30 * FEATURE_DIM);
    rs.forEach((f, i) => flat.set(f, i * FEATURE_DIM));
    const { label, prob, probs } = tf.tidy(() => {
      const t = tf.tensor3d(flat, [1, 30, FEATURE_DIM]);
      const p = this.dynModel.predict(t).dataSync();
      let best = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
      return { label: this.dynLabels[best], prob: p[best], probs: Array.from(p) };
    });
    // DEBUG (privremeno): ispis stvarnih vjerojatnosti za svaku uživo uhvaćenu
    // gestu, radi dijagnoze J/Z prepoznavanja -> ukloniti kad se problem riješi.
    console.log("[dynamic]", `seqLen=${this.seq.length}`,
      Object.fromEntries(this.dynLabels.map((l, i) => [l, +probs[i].toFixed(3)])));
    const matched = (label === "J" || label === "Z") && prob >= CONF_DYNAMIC;
    return { letter: label, kind: "dynamic", prob, matched };
  }
}

/** Učitavanje TF.js modela; vraća null ako model (još) ne postoji.
 *  cache: "no-store" na SVIM fetchevima (model.json + svaki .bin shard, jer
 *  tf.loadLayersModel prosljeđuje requestInit i na fetch weighta) — bez ovoga
 *  browser zna zadržati STARI model.json/.bin trajno u HTTP cacheu i nakon
 *  ponovnog treniranja/exporta, pa se nova verzija nikad stvarno ne učita. */
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
