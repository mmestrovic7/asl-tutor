/**
 * app.js — glavna logika aplikacije (VERZIJA 3)
 * Načini rada:
 *   UČI    — level po level kroz abecedu (referenca lijevo, kamera desno)
 *   PIŠI   — slobodno sricanje: prepoznata slova se zapisuju u tekst
 *   POKAŽI — zadana rečenica: svako točno pokazano slovo postaje neprozirno
 *
 * Ispravci u odnosu na v1:
 *  - kamera je TRAJNI DOM element koji se premješta između pogleda
 *    (render više nikad ne uništava <video>, pa stream ne puca)
 *  - requestAnimationFrame se zakazuje PRVOM linijom petlje + try/catch
 *    (jedan loš frame više ne može ubiti cijelu petlju)
 */

import { createLandmarker, startCamera, drawSkeleton } from "./landmarker.js";
import { normalizeLandmarks } from "./normalize.js";
import { Recognizer, tryLoadModel } from "./recognizer.js";

console.log("app.js VERZIJA 3 učitan");

/* ------------------------------------------------------------------ */
/* Podaci                                                              */
/* ------------------------------------------------------------------ */

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const DYNAMIC = new Set(["J", "Z"]);

const DESCRIPTIONS = {
    A: "Stisnuta šaka, palac uspravno uz bok kažiprsta.",
    B: "Dlan prema kameri, četiri prsta skupljena i ispružena uvis, palac preko dlana.",
    C: "Prsti i palac savijeni tako da tvore oblik slova C.",
    D: "Kažiprst ispružen uvis, ostali prsti i palac tvore krug.",
    E: "Prsti savijeni prema dolje, palac položen ispod njih.",
    F: "Kažiprst i palac tvore krug, ostala tri prsta ispružena i razmaknuta.",
    G: "Šaka bočno, kažiprst i palac ispruženi vodoravno, paralelno.",
    H: "Šaka bočno, kažiprst i srednji prst ispruženi vodoravno, skupljeni.",
    I: "Mali prst ispružen uvis, ostali prsti stisnuti, palac preko njih.",
    J: "Znak I (mali prst) koji u zraku crta luk slova J — ovo je pokret!",
    K: "Kažiprst i srednji prst uvis u obliku V, palac naslonjen između njih.",
    L: "Kažiprst uvis, palac vodoravno — ruka tvori slovo L.",
    M: "Palac uvučen ispod tri savijena prsta.",
    N: "Palac uvučen ispod dva savijena prsta.",
    O: "Svi prsti i palac savijeni u krug — oblik slova O.",
    P: "Kao K, ali šaka usmjerena prema dolje.",
    Q: "Kao G (palac i kažiprst), ali usmjereno prema dolje.",
    R: "Kažiprst i srednji prst ispruženi i prekriženi.",
    S: "Čvrsto stisnuta šaka, palac preko prednje strane prstiju.",
    T: "Palac provučen između kažiprsta i srednjeg prsta.",
    U: "Kažiprst i srednji prst uvis, skupljeni zajedno.",
    V: "Kažiprst i srednji prst uvis, rašireni u V.",
    W: "Kažiprst, srednji i prstenjak ispruženi i rašireni.",
    X: "Kažiprst savijen u obliku kuke, ostalo stisnuto.",
    Y: "Palac i mali prst rašireni, ostala tri prsta stisnuta.",
    Z: "Ispruženim kažiprstom u zraku nacrtaj slovo Z — ovo je pokret!",
};

const SENTENCES = [
    "HI", "CAT", "DOG", "SUN", "MAP", "FISH", "BLUE", "JAZZ", "QUIZ",
    "PIZZA", "ROBOT", "ZEBRA", "JUICE", "HELLO", "WORLD", "DANCE",
    "MY NAME IS ANA", "I LIKE PIZZA", "GOOD JOB", "JUST RELAX",
];

/* ------------------------------------------------------------------ */
/* Stanje aplikacije                                                   */
/* ------------------------------------------------------------------ */

const app = {
    view: "home",
    landmarker: null,
    recognizer: null,
    stream: null,             // MediaStream kamere — kreira se jednom
    demoMode: false,          // true kad modeli još nisu istrenirani
    lastVideoTime: -1,
    // UČI
    learnIndex: 0,
    // PIŠI
    written: "",
    noHandFrames: 0,
    spaceInserted: false,
    // POKAŽI
    sentence: "",
    sentenceIndex: 0,
};

const progress = {
    load() { return new Set(JSON.parse(localStorage.getItem("asl-done") || "[]")); },
    save(set) { localStorage.setItem("asl-done", JSON.stringify([...set])); },
};
let doneLetters = progress.load();

/* ------------------------------------------------------------------ */
/* Inicijalizacija                                                     */
/* ------------------------------------------------------------------ */

async function init() {
    render();
    const [stat, dyn] = await Promise.all([
        tryLoadModel("models/static"),
        tryLoadModel("models/dynamic"),
    ]);
    app.demoMode = !stat;
    app.recognizer = new Recognizer({
        staticModel: stat?.model, staticLabels: stat?.labels,
        dynModel: dyn?.model, dynLabels: dyn?.labels,
    });
    app.landmarker = await createLandmarker();
    document.getElementById("loading")?.remove();
    render();
}

/* ------------------------------------------------------------------ */
/* Kamera: trajni element + petlja koja ne može umrijeti               */
/* ------------------------------------------------------------------ */

let camPanelEl = null; // trajni DOM čvor kamere — kreira se jednom

function cameraPanelHTML() {
    return `
    <div class="camera-panel">
      <video id="cam" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
      <div class="ring">
        <svg viewBox="0 0 64 64">
          <circle class="ring-bg" cx="32" cy="32" r="26"/>
          <circle id="ring-fill" class="ring-fg" cx="32" cy="32" r="26"
                  stroke-dasharray="${2 * Math.PI * 26}"
                  stroke-dashoffset="${2 * Math.PI * 26}"/>
        </svg>
        <span id="ring-letter"></span>
      </div>
      <div id="status-chip" class="chip">Uključujem kameru…</div>
    </div>`;
}

/** Umetne trajni panel kamere u <div id="camera-slot"> aktivnog pogleda. */
function mountCamera() {
    const slot = document.getElementById("camera-slot");
    if (!slot) return;
    if (!camPanelEl) {
        slot.innerHTML = cameraPanelHTML();
        camPanelEl = slot.firstElementChild;
    } else {
        slot.appendChild(camPanelEl); // isti element, samo premješten
    }
}

async function ensureCamera() {
    const video = document.getElementById("cam");
    if (!video) return;
    if (app.stream) {
        if (video.srcObject !== app.stream) {
            video.srcObject = app.stream;      // sigurnosna mreža (ne bi trebalo trebati)
        }
        await video.play().catch(() => {});
        return;
    }
    app.stream = await startCamera(video);
    loop();
}

function loop() {
    requestAnimationFrame(loop); // zakazano ODMAH — petlja ne može umrijeti

    const video = document.getElementById("cam");
    if (!video || !video.srcObject || video.readyState < 2) return;
    if (video.currentTime === app.lastVideoTime || !app.landmarker) return;
    app.lastVideoTime = video.currentTime;

    try {
        const res = app.landmarker.detectForVideo(video, performance.now());
        const lm = res.landmarks?.[0] || null;

        const canvas = document.getElementById("overlay");
        if (canvas) {
            canvas.width = video.videoWidth; canvas.height = video.videoHeight;
            drawSkeleton(canvas.getContext("2d"), lm, canvas.width, canvas.height);
        }

        const vec = lm ? normalizeLandmarks(lm) : null;
        const state = app.recognizer.update(vec);
        handleFrame(state);
    } catch (e) {
        console.warn("Preskačem frame zbog greške:", e);
    }
}

/* ------------------------------------------------------------------ */
/* Obrada rezultata po načinu rada                                     */
/* ------------------------------------------------------------------ */

function handleFrame(s) {
    updateHud(s);
    const recognized = s.recognized || s.dynamic; // statički ili dinamički pogodak
    if (app.view === "learn") handleLearn(s, recognized);
    else if (app.view === "write") handleWrite(s, recognized);
    else if (app.view === "show") handleShow(s, recognized);
}

function updateHud(s) {
    const chip = document.getElementById("status-chip");
    const ring = document.getElementById("ring-fill");
    const ringLetter = document.getElementById("ring-letter");
    if (!chip) return;

    if (!s.hand) { chip.textContent = "Pokaži ruku kameri"; chip.className = "chip"; }
    else if (s.state === "MOVING") { chip.textContent = "Pratim pokret…"; chip.className = "chip chip-move"; }
    else if (s.demo) { chip.textContent = "Demo način — model nije učitan"; chip.className = "chip"; }
    else if (s.letter) { chip.textContent = `Vidim: ${s.letter}`; chip.className = "chip chip-see"; }
    else { chip.textContent = "Držite znak mirno"; chip.className = "chip"; }

    if (ring) {
        const p = s.progress ?? 0;
        const C = 2 * Math.PI * 26;
        ring.style.strokeDashoffset = C * (1 - p);
        ringLetter.textContent = s.letter ?? "";
    }
}

function flashSuccess() {
    const panel = document.querySelector(".camera-panel");
    panel?.classList.remove("success");
    void panel?.offsetWidth; // restart animacije
    panel?.classList.add("success");
}

/* ---- UČI ---- */
let advancing = false; // spriječi dvostruki setTimeout dok traje prijelaz

function handleLearn(s, rec) {
    const target = ALPHABET[app.learnIndex];
    if (rec && rec.letter === target && !advancing) {
        advancing = true;
        flashSuccess();
        doneLetters.add(target);
        progress.save(doneLetters);
        setTimeout(() => {
            advancing = false;
            if (app.learnIndex < ALPHABET.length - 1) app.learnIndex++;
            render();
        }, 1100);
    }
}

/* ---- PIŠI ---- */
function handleWrite(s, rec) {
    if (rec) {
        app.written += rec.letter;
        app.spaceInserted = false;
        flashSuccess();
        refreshWritten();
    }
    // razmak: makni ruku iz kadra na ~1 s
    if (!s.hand) {
        app.noHandFrames++;
        if (app.noHandFrames > 30 && app.written && !app.spaceInserted
            && !app.written.endsWith(" ")) {
            app.written += " ";
            app.spaceInserted = true;
            refreshWritten();
        }
    } else app.noHandFrames = 0;
}

function refreshWritten() {
    const el = document.getElementById("written");
    if (el) el.textContent = app.written || " ";
}

/* ---- POKAŽI ---- */
function handleShow(s, rec) {
    const chars = app.sentence.split("");
    // preskoči razmake
    while (app.sentenceIndex < chars.length && chars[app.sentenceIndex] === " ")
        app.sentenceIndex++;
    const target = chars[app.sentenceIndex];
    if (!target) return;
    if (rec && rec.letter === target) {
        app.sentenceIndex++;
        flashSuccess();
        refreshSentence();
        if (app.sentenceIndex >= chars.length)
            document.getElementById("sentence-done")?.classList.add("visible");
    }
}

function refreshSentence() {
    const box = document.getElementById("sentence-box");
    if (!box) return;
    box.innerHTML = "";
    app.sentence.split("").forEach((ch, i) => {
        const span = document.createElement("span");
        span.textContent = ch;
        span.className = "s-letter" +
            (i < app.sentenceIndex ? " done" : "") +
            (i === app.sentenceIndex ? " current" : "") +
            (ch === " " ? " space" : "");
        box.appendChild(span);
    });
}

/* ------------------------------------------------------------------ */
/* Render pogleda                                                      */
/* ------------------------------------------------------------------ */

const root = () => document.getElementById("view");

function render() {
    ({ home: renderHome, learn: renderLearn, write: renderWrite, show: renderShow })[app.view]();
    if (app.view !== "home") {
        mountCamera();
        ensureCamera().catch(err => {
            root().insertAdjacentHTML("afterbegin",
                `<div class="banner warn">Kamera nije dostupna: ${err.message}</div>`);
        });
    }
}

function go(view) { app.view = view; render(); }
window.go = go;

function demoBanner() {
    return app.demoMode
        ? `<div class="banner">Modeli još nisu istrenirani — aplikacija prikazuje kostur šake,
       ali ne prepoznaje znakove. Pokreni <code>train_static.py</code> i
       <code>train_dynamic.py</code> pa osvježi stranicu.</div>`
        : "";
}

/* ---- POČETNA ---- */
function renderHome() {
    const grid = ALPHABET.map((L, i) => {
        const done = doneLetters.has(L);
        const dyn = DYNAMIC.has(L) ? " dyn" : "";
        return `<button class="tile${done ? " done" : ""}${dyn}"
              onclick="app_openLetter(${i})" title="${DESCRIPTIONS[L]}">
              ${L}${DYNAMIC.has(L) ? "<small>pokret</small>" : ""}
            </button>`;
    }).join("");

    root().innerHTML = `
    ${demoBanner()}
    <section class="hero">
      <h1>Nauči <span class="accent">ASL</span> abecedu</h1>
      <p>Ručna abeceda američkog znakovnog jezika — slovo po slovo,
         uz kameru i trenutnu povratnu informaciju.</p>
    </section>
    <section class="modes">
      <button class="mode-card" onclick="go('learn')">
        <span class="mode-k">01</span><h3>Uči abecedu</h3>
        <p>Prolazi kroz slova redom. Znak držiš dok se prsten ne napuni.</p>
      </button>
      <button class="mode-card" onclick="app_startWrite()">
        <span class="mode-k">02</span><h3>Piši slobodno</h3>
        <p>Sriči što god želiš — prepoznata slova zapisujemo u tekst.</p>
      </button>
      <button class="mode-card" onclick="app_startShow()">
        <span class="mode-k">03</span><h3>Pokaži rečenicu</h3>
        <p>Zadamo ti rečenicu, a slova svijetle kako ih pokazuješ.</p>
      </button>
    </section>
    <section>
      <h2 class="section-title">Karta slova
        <span class="progress-note">${doneLetters.size}/26 naučeno</span></h2>
      <div class="letter-grid">${grid}</div>
    </section>`;
}
window.app_openLetter = (i) => { app.learnIndex = i; go("learn"); };
window.app_startWrite = () => { app.written = ""; app.noHandFrames = 0; go("write"); };
window.app_startShow = () => {
    app.sentence = SENTENCES[Math.floor(Math.random() * SENTENCES.length)];
    app.sentenceIndex = 0; go("show");
};

/* ---- UČI ---- */
function renderLearn() {
    const L = ALPHABET[app.learnIndex];
    const dyn = DYNAMIC.has(L);
    root().innerHTML = `
    ${demoBanner()}
    <div class="toolbar">
      <button class="btn ghost" onclick="go('home')">← Karta slova</button>
      <span class="crumb">Slovo ${app.learnIndex + 1} / 26</span>
      <span class="spacer"></span>
      <button class="btn ghost" onclick="app_prevLetter()">Prethodno</button>
      <button class="btn" onclick="app_skipLetter()">Preskoči →</button>
    </div>
    <div class="split">
      <div class="ref-card ${dyn ? "ref-dyn" : ""}">
        <span class="ref-letter">${L}</span>
        <img class="ref-img" src="assets/signs/${L}.png" alt=""
             onerror="this.remove()">
        <p class="ref-desc">${DESCRIPTIONS[L]}</p>
        ${dyn ? `<p class="ref-hint">Dinamički znak: izvedi pokret u jednom
                  potezu, pa umiri ruku.</p>`
        : `<p class="ref-hint">Drži znak mirno dok se prsten ne napuni.</p>`}
      </div>
      <div id="camera-slot"></div>
    </div>`;
}
window.app_skipLetter = () => {
    if (app.learnIndex < ALPHABET.length - 1) { app.learnIndex++; render(); }
    else go("home");
};
window.app_prevLetter = () => {
    if (app.learnIndex > 0) { app.learnIndex--; render(); }
};

/* ---- PIŠI ---- */
function renderWrite() {
    root().innerHTML = `
    ${demoBanner()}
    <div class="toolbar">
      <button class="btn ghost" onclick="go('home')">← Natrag</button>
      <span class="crumb">Slobodno pisanje</span>
      <span class="spacer"></span>
      <button class="btn ghost" onclick="app_backspace()">⌫ Obriši slovo</button>
      <button class="btn ghost" onclick="app_clear()">Očisti sve</button>
    </div>
    <div class="split">
      <div class="write-card">
        <p class="write-label">Tvoj tekst</p>
        <p id="written" class="written">${app.written || " "}</p>
        <p class="ref-hint">Razmak: makni ruku iz kadra na sekundu.
           Za dvostruko slovo kratko promijeni znak pa se vrati.</p>
      </div>
      <div id="camera-slot"></div>
    </div>`;
}
window.app_backspace = () => { app.written = app.written.slice(0, -1); refreshWritten(); };
window.app_clear = () => { app.written = ""; refreshWritten(); };

/* ---- POKAŽI ---- */
function renderShow() {
    root().innerHTML = `
    ${demoBanner()}
    <div class="toolbar">
      <button class="btn ghost" onclick="go('home')">← Natrag</button>
      <span class="crumb">Pokaži rečenicu</span>
      <span class="spacer"></span>
      <button class="btn ghost" onclick="app_skipChar()">Preskoči slovo</button>
      <button class="btn" onclick="app_startShow()">Nova rečenica</button>
    </div>
    <div class="split">
      <div class="write-card">
        <p class="write-label">Pokaži redom:</p>
        <p id="sentence-box" class="sentence"></p>
        <p id="sentence-done" class="done-note">Bravo, cijela rečenica! 🎉</p>
      </div>
      <div id="camera-slot"></div>
    </div>`;
    refreshSentence();
}
window.app_skipChar = () => {
    if (app.sentenceIndex < app.sentence.length) {
        app.sentenceIndex++; refreshSentence();
    }
};

init();