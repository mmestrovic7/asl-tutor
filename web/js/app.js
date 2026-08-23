import { createLandmarker, startCamera, drawSkeleton } from "./landmarker.js?v=12";
import { normalizeLandmarks } from "./normalize.js?v=12";
import { Recognizer, tryLoadModel } from "./recognizer.js?v=12";

/* ------------------------------------------------------------------ */
/* Data                                                              */
/* ------------------------------------------------------------------ */

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const DYNAMIC = new Set(["J", "Z"]);

const SURVEY_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeZ5pAdlUFKtPFsbCRE5O8rNTtONRbo0DhcZSrz99Bsl8Q8aA/viewform";
const SURVEY_ENTRY_LETTERS_CORRECT = "entry.1496988052"; // pitanje 3: "How many letters were you able to sign correctly?"
const SURVEY_MIN = 3; // minimalni angažman prije nego anketa postane klikabilna
const SURVEY_THRESHOLD = Math.ceil(ALPHABET.length * 0.75); // 20/26 - prag za proslavni popup

const DESCRIPTIONS = {
    A: "Closed fist, thumb upright alongside the index finger.",
    B: "Palm facing the camera, four fingers together and extended upward, thumb across the palm.",
    C: "Fingers and thumb curved to form the shape of the letter C.",
    D: "Index finger extended upward, other fingers and thumb forming a circle.",
    E: "Fingers bent downward, thumb placed underneath them.",
    F: "Index finger and thumb form a circle, the other three fingers extended and spread apart.",
    G: "Hand turned sideways, index finger and thumb extended horizontally and parallel.",
    H: "Hand turned sideways, index and middle fingers extended horizontally and held together.",
    I: "Pinky finger extended upward, other fingers closed, thumb across them.",
    J: "The I sign (pinky finger) drawing the curve of the letter J in the air: this is a movement!",
    K: "Index and middle fingers extended upward in a V shape, thumb resting between them.",
    L: "Index finger upright, thumb horizontal, forming the letter L.",
    M: "Thumb tucked underneath three bent fingers.",
    N: "Thumb tucked underneath two bent fingers.",
    O: "All fingers and thumb curved into a circle, forming the shape of the letter O.",
    P: "Like K, but with the hand pointing downward.",
    Q: "Like G (thumb and index finger), but pointing downward.",
    R: "Index and middle fingers extended and crossed.",
    S: "Tightly closed fist, thumb across the front of the fingers.",
    T: "Thumb placed between the index and middle fingers.",
    U: "Index and middle fingers extended upward and held together.",
    V: "Index and middle fingers extended upward, spread into a V.",
    W: "Index, middle, and ring fingers extended and spread apart.",
    X: "Index finger bent into a hook shape, the rest closed.",
    Y: "Thumb and pinky spread apart, the other three fingers closed.",
    Z: "Draw the letter Z in the air with your extended index finger: this is a movement!",
};

const SENTENCES = [
    "HI", "CAT", "DOG", "SUN", "MAP", "FISH", "BLUE", "JAZZ", "QUIZ",
    "PIZZA", "ROBOT", "ZEBRA", "JUICE", "HELLO", "WORLD", "DANCE",
    "MY NAME IS ANA", "I LIKE PIZZA", "GOOD JOB", "JUST RELAX",
];

/* ------------------------------------------------------------------ */
/* Application state                                                   */
/* ------------------------------------------------------------------ */

const app = {
    view: "home",
    landmarker: null,
    recognizer: null,
    stream: null,             // Camera MediaStream, created once
    demoMode: false,          // true when the models have not been trained yet
    lastVideoTime: -1,
    // LEARN
    learnIndex: 0,
    // WRITE
    written: "",
    noHandFrames: 0,
    spaceInserted: false,
    // SHOW
    sentence: "",
    sentenceIndex: 0,
};

const progress = {
    load() { return new Set(JSON.parse(localStorage.getItem("asl-done") || "[]")); },
    save(set) { localStorage.setItem("asl-done", JSON.stringify([...set])); },
};
let doneLetters = progress.load();

/* ------------------------------------------------------------------ */
/* Initialization                                                     */
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
    window.__debugRecognizer = app.recognizer; // DEBUG (privremeno)
    app.landmarker = await createLandmarker();
    document.getElementById("loading")?.remove();
    render();
}

let camPanelEl = null; // persistent camera DOM node, created once

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
      <div id="status-chip" class="chip">Turning on camera…</div>
      <div id="success-overlay" class="success-overlay">
        <div class="success-check">✓</div>
        <div id="success-letter" class="success-letter"></div>
      </div>
    </div>`;
}

function mountCamera() {
    const slot = document.getElementById("camera-slot");
    if (!slot) return;
    if (!camPanelEl) {
        slot.innerHTML = cameraPanelHTML();
        camPanelEl = slot.firstElementChild;
    } else {
        slot.appendChild(camPanelEl); // same element, just moved
    }
}

async function ensureCamera() {
    const video = document.getElementById("cam");
    if (!video) return;
    if (app.stream) {
        if (video.srcObject !== app.stream) {
            video.srcObject = app.stream;      // safety net (should not be necessary)
        }
        await video.play().catch(() => {});
        return;
    }
    app.stream = await startCamera(video);
    loop();
}

function loop() {
    requestAnimationFrame(loop); // scheduled IMMEDIATELY, the loop cannot die

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
        const rawWrist = lm ? { x: lm[0].x, y: lm[0].y } : null;
        const state = app.recognizer.update(vec, rawWrist);
        handleFrame(state);
    } catch (e) {
        console.warn("Skipping frame due to error:", e);
    }
}

/* ------------------------------------------------------------------ */
/* Process results by mode                                     */
/* ------------------------------------------------------------------ */

function handleFrame(s) {
    updateHud(s);
    const recognized = s.recognized || s.dynamic; // static or dynamic match
    if (app.view === "learn") handleLearn(s, recognized);
    else if (app.view === "write") handleWrite(s, recognized);
    else if (app.view === "show") handleShow(s, recognized);
}

let dynamicMissUntil = 0; // dok je u budućnosti, chip prikazuje poruku o neuspjelom J/Z pokušaju

function updateHud(s) {
    const chip = document.getElementById("status-chip");
    const ring = document.getElementById("ring-fill");
    const ringLetter = document.getElementById("ring-letter");
    if (!chip) return;

    if (s.dynamicMiss) dynamicMissUntil = Date.now() + 1500;

    if (Date.now() < dynamicMissUntil) {
        chip.textContent = "Didn't quite catch that, try again"; chip.className = "chip chip-miss";
    } else if (!s.hand) { chip.textContent = "Show your hand to the camera"; chip.className = "chip"; }
    else if (s.state === "MOVING") { chip.textContent = "Tracking movement…"; chip.className = "chip chip-move"; }
    else if (s.demo) { chip.textContent = "Demo mode: model not loaded"; chip.className = "chip"; }
    else if (s.letter) { chip.textContent = `I see: ${s.letter}`; chip.className = "chip chip-see"; }
    else { chip.textContent = "Hold the sign still"; chip.className = "chip"; }

    if (ring) {
        const p = s.progress ?? 0;
        const C = 2 * Math.PI * 26;
        ring.style.strokeDashoffset = C * (1 - p);
        ringLetter.textContent = s.letter ?? "";
    }
}

let flashSuccessTimer = null;

function flashSuccess(letter, duration = 650) {
    const panel = document.querySelector(".camera-panel");
    const overlay = document.getElementById("success-overlay");
    const letterEl = document.getElementById("success-letter");
    if (letterEl) letterEl.textContent = letter || "";

    panel?.classList.remove("success");
    void panel?.offsetWidth; // restart the animation
    panel?.classList.add("success");

    overlay?.classList.add("show");
    clearTimeout(flashSuccessTimer);
    flashSuccessTimer = setTimeout(() => overlay?.classList.remove("show"), duration);
}

/* ---- LEARN ---- */
let advancing = false;

function handleLearn(s, rec) {
    const target = ALPHABET[app.learnIndex];
    if (rec && rec.letter === target && !advancing) {
        advancing = true;
        flashSuccess(target, 1050);
        doneLetters.add(target);
        progress.save(doneLetters);
        checkSurveyMinReached();
        checkSurveyUnlock();
        setTimeout(() => {
            advancing = false;
            if (app.learnIndex < ALPHABET.length - 1) app.learnIndex++;
            render();
        }, 1100);
    }
}

/* ---- Survey notifications: eligibility (3 slova) i napredak (75%) ---- */
function checkSurveyMinReached() {
    if (doneLetters.size < SURVEY_MIN) return;
    if (localStorage.getItem("asl-survey-min-notified")) return;
    localStorage.setItem("asl-survey-min-notified", "1");
    setTimeout(() => showSurveyModal({
        title: "Survey unlocked",
        body: "You can now fill out a short survey about your experience so far.",
    }), 1300); // nakon success animacije
}

function checkSurveyUnlock() {
    if (doneLetters.size < SURVEY_THRESHOLD) return;
    if (localStorage.getItem("asl-survey-notified")) return;
    localStorage.setItem("asl-survey-notified", "1");
    setTimeout(() => showSurveyModal({
        title: `You've learned ${SURVEY_THRESHOLD}/${ALPHABET.length} letters`,
        body: "You can now fill out a short survey about your experience.",
    }), 1300);
}

function showSurveyModal({ title, body }) {
    const modal = document.createElement("div");
    modal.className = "modal-overlay open";
    modal.innerHTML = `
      <div class="modal-box survey-modal">
        <button class="modal-close" onclick="this.closest('.modal-overlay').remove()" aria-label="Close">✕</button>
        <h3>${title}</h3>
        <p>${body}</p>
        <div class="survey-modal-actions">
          <button class="btn" onclick="app_openSurvey(); this.closest('.modal-overlay').remove()">Fill out survey</button>
          <button class="btn ghost" onclick="this.closest('.modal-overlay').remove()">Maybe later</button>
        </div>
      </div>`;
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });
    document.body.appendChild(modal);
}
window.app_openSurvey = () => {
    if (!SURVEY_URL) return;
    const url = `${SURVEY_URL}?usp=pp_url&${SURVEY_ENTRY_LETTERS_CORRECT}=${doneLetters.size}`;
    window.open(url, "_blank");
};

function surveySectionHTML() {
    const n = doneLetters.size;
    const milestone = n >= SURVEY_THRESHOLD;
    const eligible = n >= SURVEY_MIN;
    return `
    <section class="survey-card ${eligible ? "unlocked" : ""}">
      <div class="survey-text">
        <h3>${milestone ? "Great progress, survey time!" : eligible ? "Survey unlocked" : "Survey"}</h3>
        <p>${milestone
            ? "You've learned most of the alphabet. Tell us about your experience."
            : eligible
                ? `Tell us about your experience any time but try to learn the whole
                   alphabet first.`
                : `Try at least ${SURVEY_MIN} letters first, then come back and tell us
                   how it went (${n}/${SURVEY_MIN} so far).`}</p>
      </div>
      <button class="btn ${eligible ? "" : "disabled"}" ${eligible ? "" : "disabled"}
              onclick="app_openSurvey()">
        ${eligible ? "Fill out survey" : `${n}/${SURVEY_MIN}`}
      </button>
    </section>`;
}

/* ---- WRITE ---- */
function handleWrite(s, rec) {
    if (rec) {
        app.written += rec.letter;
        app.spaceInserted = false;
        flashSuccess(rec.letter);
        refreshWritten();
    }
    // space: move your hand out of frame for ~1s
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

/* ---- SHOW ---- */
function handleShow(s, rec) {
    const chars = app.sentence.split("");
    // skip spaces
    while (app.sentenceIndex < chars.length && chars[app.sentenceIndex] === " ")
        app.sentenceIndex++;
    const target = chars[app.sentenceIndex];
    if (!target) return;
    if (rec && rec.letter === target) {
        app.sentenceIndex++;
        flashSuccess(target);
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
/* Render views                                                        */
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
        ? `<div class="banner">The models have not been trained yet: the app displays the hand skeleton,
       but does not recognize signs. Run <code>train_static.py</code> and
       <code>train_dynamic.py</code> then refresh the page.</div>`
        : "";
}

/* ---- HOME ---- */
function renderHome() {
    const grid = ALPHABET.map((L, i) => {
        const done = doneLetters.has(L);
        const dyn = DYNAMIC.has(L) ? " dyn" : "";
        return `<button class="tile${done ? " done" : ""}${dyn}"
              onclick="app_openLetter(${i})" title="${DESCRIPTIONS[L]}">
              ${L}${DYNAMIC.has(L) ? "<small>movement</small>" : ""}
            </button>`;
    }).join("");

    root().innerHTML = `
    ${demoBanner()}
    <section class="hero">
      <h1>Learn the <span class="accent">ASL</span> alphabet</h1>
      <p>The American Sign Language manual alphabet, one letter at a time,
         with a camera and instant feedback.</p>
    </section>
    <section class="modes">
      <button class="mode-card" onclick="go('learn')">
        <span class="mode-k">01</span><h3>Learn the alphabet</h3>
        <p>Go through the letters in order. Hold the sign until the ring fills up.</p>
      </button>
      <button class="mode-card" onclick="app_startWrite()">
        <span class="mode-k">02</span><h3>Write freely</h3>
        <p>Spell anything you want: recognized letters are added to the text.</p>
      </button>
      <button class="mode-card" onclick="app_startShow()">
        <span class="mode-k">03</span><h3>Show a sentence</h3>
        <p>We give you a sentence, and the letters light up as you sign them.</p>
      </button>
    </section>
    ${surveySectionHTML()}
    <section>
      <h2 class="section-title">Letter map
        <span class="progress-note">${doneLetters.size}/26 learned</span></h2>
      <div class="letter-grid">${grid}</div>
      <button class="btn ghost" style="margin-top:16px" onclick="app_showAlphabet()">See whole alphabet</button>
    </section>`;
}
window.app_openLetter = (i) => { app.learnIndex = i; go("learn"); };
window.app_startWrite = () => { app.written = ""; app.noHandFrames = 0; go("write"); };
window.app_startShow = () => {
    app.sentence = SENTENCES[Math.floor(Math.random() * SENTENCES.length)];
    app.sentenceIndex = 0; go("show");
};

/* ---- Reference Image/Video toggle ---- */
function refBlockHTML(letter) {
    const mode = app.refMode || "image";
    return `<div id="ref-media-wrap">
      <div class="ref-toggle">
        <button class="ref-tab ${mode === "image" ? "active" : ""}" onclick="app_setRefMode('image')">Image</button>
        <button class="ref-tab ${mode === "video" ? "active" : ""}" onclick="app_setRefMode('video')">Video</button>
      </div>
      <div class="ref-media">
        ${mode === "video"
            ? `<video class="ref-vid" src="assets/signs/${letter}.mp4" autoplay loop muted playsinline
                 onerror="app_refVideoMissing()"></video>`
            : `<img class="ref-img" src="assets/signs/${letter}.png" alt="" onerror="this.remove()">`}
      </div>
    </div>`;
}
window.app_setRefMode = (mode) => {
    app.refMode = mode;
    const wrap = document.getElementById("ref-media-wrap");
    if (wrap) wrap.outerHTML = refBlockHTML(ALPHABET[app.learnIndex]);
};
window.app_refVideoMissing = () => {
    const media = document.querySelector("#ref-media-wrap .ref-media");
    if (media) media.innerHTML = `<p class="ref-video-missing">Video coming soon for this letter.</p>`;
};

/* ---- LEARN ---- */
function renderLearn() {
    const L = ALPHABET[app.learnIndex];
    const dyn = DYNAMIC.has(L);
    root().innerHTML = `
    ${demoBanner()}
    <div class="toolbar">
      <button class="btn ghost" onclick="go('home')">← Letter map</button>
      <span class="crumb">Letter ${app.learnIndex + 1} / 26</span>
      <span class="spacer"></span>
      <button class="btn ghost" onclick="app_showAlphabet()">See whole alphabet</button>
      <button class="btn ghost" onclick="app_prevLetter()">Previous</button>
      <button class="btn" onclick="app_skipLetter()">Skip →</button>
    </div>
    <div class="split">
      <div class="ref-card ${dyn ? "ref-dyn" : ""}">
        <span class="ref-letter">${L}</span>
        ${refBlockHTML(L)}
        <p class="ref-desc">${DESCRIPTIONS[L]}</p>
        ${dyn ? `<p class="ref-hint">Dynamic sign: perform the motion in one
                  smooth movement, then hold your hand still.</p>`
        : `<p class="ref-hint">Hold the sign still until the ring fills up.</p>`}
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

/* ---- WRITE ---- */
function renderWrite() {
    root().innerHTML = `
    ${demoBanner()}
    <div class="toolbar">
      <button class="btn ghost" onclick="go('home')">← Back</button>
      <span class="crumb">Free writing</span>
      <span class="spacer"></span>
      <button class="btn ghost" onclick="app_showAlphabet()">See whole alphabet</button>
      <button class="btn ghost" onclick="app_backspace()">⌫ Delete letter</button>
      <button class="btn ghost" onclick="app_clear()">Clear all</button>
    </div>
    <div class="split">
      <div class="write-card">
        <p class="write-label">Your text</p>
        <p id="written" class="written">${app.written || " "}</p>
        <p class="ref-hint">Space: move your hand out of the frame for one second.
           For a double letter, briefly change the sign and then return to it.</p>
      </div>
      <div id="camera-slot"></div>
    </div>`;
}
window.app_backspace = () => { app.written = app.written.slice(0, -1); refreshWritten(); };
window.app_clear = () => { app.written = ""; refreshWritten(); };

/* ---- SHOW ---- */
function renderShow() {
    root().innerHTML = `
    ${demoBanner()}
    <div class="toolbar">
      <button class="btn ghost" onclick="go('home')">← Back</button>
      <span class="crumb">Show a sentence</span>
      <span class="spacer"></span>
      <button class="btn ghost" onclick="app_showAlphabet()">See whole alphabet</button>
      <button class="btn ghost" onclick="app_skipChar()">Skip letter</button>
      <button class="btn" onclick="app_startShow()">New sentence</button>
    </div>
    <div class="split">
      <div class="write-card">
        <p class="write-label">Show in order:</p>
        <p id="sentence-box" class="sentence"></p>
        <p id="sentence-done" class="done-note">Great job, the whole sentence!</p>
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

/* ---- ALPHABET MODAL ---- */
function showAlphabetModal() {
    let modal = document.getElementById("alphabet-modal");
    if (!modal) {
        modal = document.createElement("div");
        modal.id = "alphabet-modal";
        modal.className = "modal-overlay";
        modal.innerHTML = `
      <div class="modal-box">
        <a class="modal-download" href="assets/signs/alphabet.png" download="asl-alphabet.png">Download</a>
        <button class="modal-close" onclick="app_closeAlphabet()" aria-label="Close">✕</button>
        <img src="assets/signs/alphabet.png" alt="Cijela ASL abeceda" class="modal-img">
      </div>`;
        modal.addEventListener("click", (e) => { if (e.target === modal) closeAlphabetModal(); });
        document.body.appendChild(modal);
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") closeAlphabetModal();
        });
    }
    modal.classList.add("open");
}
function closeAlphabetModal() {
    document.getElementById("alphabet-modal")?.classList.remove("open");
}
window.app_showAlphabet = showAlphabetModal;
window.app_closeAlphabet = closeAlphabetModal;

init();