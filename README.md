# Slovka — sustav za prepoznavanje i učenje ASL ručne abecede

Diplomski projekt: prepoznavanje statičkih **i dinamičkih** znakova američkog
znakovnog jezika (ASL) u stvarnom vremenu, u web pregledniku.

## Kako sustav radi (arhitektura)

```
kamera (30 fps)
   │
   ▼
MediaPipe HandLandmarker (WebAssembly, u browseru)
   │  21 točka šake × (x, y, z)  =  63 broja po frameu
   ▼
Normalizacija (translacija na zapešće + skaliranje veličinom dlana)
   │
   ▼
Detektor pokreta (energija = prosječni pomak vektora između frameova)
   │
   ├── ruka MIRUJE ──► MLP (63 → 128 → 64 → 24)  ─► statička slova A–Y (bez J, Z)
   │                    slovo prihvaćeno nakon ~0.5 s stabilnog držanja
   │                    s pouzdanošću ≥ 0.8 (korisnik vidi prsten koji se puni)
   │
   └── ruka se KREĆE ─► buffer frameova → resample na 30 → GRU(64) → {J, Z, OSTALO}
                        gesta se klasificira kad pokret završi
```

Sva inferencija (MediaPipe + TensorFlow.js) izvodi se **lokalno u browseru** —
nema backenda, video nikad ne napušta računalo korisnika (bitan argument
privatnosti za rad).

## Struktura projekta

```
asl-tutor/
├── training/                  Python — priprema podataka i treniranje
│   ├── requirements.txt
│   ├── landmark_utils.py      normalizacija (ISTA logika kao web/js/normalize.js!)
│   ├── extract_static.py      dataset slika → CSV landmarkova
│   ├── collect_static.py      snimanje vlastitih statičkih uzoraka (kamera)
│   ├── collect_dynamic.py     snimanje sekvenci za J / Z / OSTALO (kamera)
│   ├── train_static.py        MLP + Random Forest usporedba → TF.js izvoz
│   └── train_dynamic.py       GRU → TF.js izvoz
└── web/                       aplikacija (čisti HTML/CSS/JS, bez build alata)
    ├── index.html
    ├── style.css
    ├── js/  (normalize.js, landmarker.js, recognizer.js, app.js)
    ├── models/   ← ovdje završe istrenirani TF.js modeli
    └── assets/signs/  ← opcionalno: A.png … Y.png ilustracije znakova
```

## Korak po korak

### 1. Okruženje

```bash
cd training
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Skup podataka za statička slova

Dvije opcije (najbolje obje, pa spojiti):

**a) Javni dataset** — na Kaggleu potraži *"ASL Alphabet"* (autor grassknoted,
~87 000 slika, po ~3 000 na slovo). Raspakiraj pa:

```bash
python extract_static.py --dataset putanja/do/asl_alphabet_train --out static_landmarks.csv
```

Skripta svaku sliku provuče kroz MediaPipe i sprema normalizirane landmarkove.
Postotak neuspjelih detekcija koji ispiše na kraju iskoristi u poglavlju
o skupu podataka.

**b) Vlastiti uzorci** (jako podiže točnost na *tvojoj* kameri):

```bash
python collect_static.py --out my_static.csv
# drži tipku slova dok pokazuješ znak; 5–10 s po slovu je dovoljno
```

### 3. Treniranje statičkog modela

```bash
python train_static.py --csv static_landmarks.csv --csv my_static.csv
```

Ispisuje točnost MLP-a i Random Foresta (usporedba za rad), sprema
`confusion_matrix.png` (ide direktno u poglavlje Evaluacija) i izvozi
TF.js model u `web/models/static/`.

### 4. Dinamički znakovi J i Z

```bash
python collect_dynamic.py --out dynamic_data
# 'j' → izvedi J;  'z' → izvedi Z;  'o' → nasumični pokreti / prijelazi
# cilj: ~80 sekvenci J, ~80 Z, ~180 OSTALO (obje ruke, razne brzine)

python train_dynamic.py --data dynamic_data
```

Klasa **OSTALO** je ključna: bez nje bi model svaki prijelaz ruke između slova
proglašavao J-om ili Z-om. Snimi je raznovrsno!

### 5. Pokretanje aplikacije

Aplikacija se mora posluživati preko HTTP-a (module skripte + kamera ne rade
s `file://`):

```bash
cd ../web
python -m http.server 8000
# otvori http://localhost:8000
```

Bez istreniranih modela aplikacija radi u "demo" načinu (crta kostur šake,
ne prepoznaje) i to jasno kaže banerom.

### 6. (Opcionalno) Ilustracije znakova

U `web/assets/signs/` stavi `A.png` … `Y.png` — prikazuju se na referentnoj
kartici. Ako slike nema, kartica prikazuje veliko slovo + tekstualni opis
(opisi su ugrađeni u `app.js`). Pazi na licencu slika; postoje public-domain
ASL grafikoni, a možeš ih i sama nacrtati.

## Ključni detalji za poglavlje Implementacija

* **Normalizacija** (`landmark_utils.py` ↔ `normalize.js`): translacija svih
  točaka tako da je zapešće u ishodištu + dijeljenje udaljenošću
  zapešće→korijen srednjeg prsta. Model tako ne ovisi o poziciji ruke u kadru,
  udaljenosti od kamere ni veličini šake. *Funkcije moraju ostati identične u
  oba jezika* — svaka nesimetrija tiho ruši točnost u produkciji.
* **Zrcalna augmentacija** umjesto handedness logike: trening skup se udvostruči
  zrcalnim kopijama (x → −x), pa model radi za lijevu i desnu ruku, a aplikacija
  ne mora tumačiti MediaPipeovu oznaku ruke (koja se kod zrcaljene selfie
  kamere lako krivo interpretira).
* **Resampliranje sekvenci** na fiksnih 30 frameova linearnom interpolacijom:
  GRU dobiva jednak oblik ulaza bez obzira na to traje li gesta 0.4 s ili 1.2 s.
* **Stroj stanja STILL/MOVING** s histerezom (dva praga, 0.030 za ulaz i 0.014
  za izlaz iz pokreta, na EMA-uglađenoj energiji) sprječava treperenje između
  statičkog i dinamičkog puta.
* **Prihvaćanje znaka**: pouzdanost ≥ 0.8 kroz 15 uzastopnih frameova +
  "cooldown" (isti znak se ne broji dvaput dok se ne promijeni ili ruka ne
  makne) — sprječava slučajne pogotke tijekom prijelaza i duplo upisivanje.

## Poznata ograničenja (iskreno navesti u radu)

* Slova **A, E, M, N, S, T** vizualno su vrlo slična (šaka, razlika je položaj
  palca) — očekuj najviše zabuna upravo tu; pokaži matricom konfuzije.
* MediaPipe z-koordinata je gruba procjena dubine; znakovi koji se razlikuju
  rotacijom prema kameri (**G/Q, H/U, K/P**) osjetljivi su na kut snimanja.
* Sustav uči *ručnu abecedu*, ne ASL kao jezik (ASL ima vlastitu gramatiku);
  fingerspelling je ulazna točka, i to treba jasno reći u uvodu i zaključku.

## Verzije i kompatibilnost

Python 3.10/3.11, `tensorflow==2.15` + `tensorflowjs==4.17` (usklađeni parovi
— novije verzije TF-a koriste Keras 3 i lome konverter), `mediapipe==0.10.14`.
U browseru: `@tensorflow/tfjs@4.20`, `@mediapipe/tasks-vision@0.10.14` (CDN).
