"""
collect_dynamic.py
------------------
Snimanje SEKVENCI za dinamičke znakove J i Z (+ klasa OSTALO).

J = šaka u obliku slova I (ispružen mali prst) koja crta luk slova J.
Z = ispružen kažiprst koji u zraku crta slovo Z.
OSTALO = sve što NIJE J ni Z: prijelazi između statičkih slova, nasumično
         mahanje, podizanje/spuštanje ruke... Ova klasa sprječava da model
         svaki pokret ruke proglasi J-om ili Z-om - snimi je BAREM koliko
         i J i Z zajedno, što raznovrsnije.

Upute:
    python collect_dynamic.py --out dynamic_data
    - tipka 'j' => počinje snimanje jedne sekvence za J; pritisni 'j' PONOVNO
      da ODMAH završiš i spremiš (ručno označavanje kraja geste - precizni je
      od automatskog čekanja na mirovanje). Ako ne pritisneš ništa, snimanje
      se ipak automatski zatvara ~0.5 s nakon što se pokret smiri (fallback).
    - tipka 'z' => sekvenca za Z (isto pravilo: 'z' ponovno = kraj)
    - tipka 'o' => sekvenca za OSTALO (isto pravilo: 'o' ponovno = kraj)
    - SPACE tijekom snimanja => odbaci trenutnu sekvencu bez spremanja
    - ESC izlaz

Preporuka: 60–100 sekvenci za J, isto za Z, 150–200 za OSTALO.
Snimaj s obje ruke, različitim brzinama i s malo različitih kutova.
Sekvence se spremaju kao .npy datoteke oblika (T, 63) - resampliranje na
fiksnih 30 frameova radi se tek u train_dynamic.py.
"""

import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import normalize_landmarks, BASE_DIM

LABELS = {"j": "J", "z": "Z", "o": "OTHER"}
MOTION_STOP_SECONDS = 0.5   # koliko mirovanja označava kraj geste
MOTION_EPS = 0.03           # prag energije pokreta (prosjecni pomak vektora)
MIN_FRAMES = 8
MAX_FRAMES = 90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dynamic_data")
    args = ap.parse_args()
    for lab in LABELS.values():
        os.makedirs(os.path.join(args.out, lab), exist_ok=True)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.6, min_tracking_confidence=0.6)
    drawer = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    recording = None      # None ili naziv klase
    buffer = []           # lista 63-dim vektora
    last_vec = None
    still_since = None
    counts = {lab: len(os.listdir(os.path.join(args.out, lab))) for lab in LABELS.values()}

    def save():
        nonlocal buffer, recording
        if recording and MIN_FRAMES <= len(buffer):
            arr = np.stack(buffer).astype(np.float32)
            idx = counts[recording]
            np.save(os.path.join(args.out, recording, f"{recording}_{idx:04d}.npy"), arr)
            counts[recording] += 1
            print(f"Spremljeno {recording} #{idx} ({len(buffer)} frameova)")
        elif recording:
            print(f"Preskačem {recording}: premalo frameova ({len(buffer)} < {MIN_FRAMES})")
        buffer = []
        recording = None

    def discard():
        nonlocal buffer, recording
        print(f"Odbačeno {recording} ({len(buffer)} frameova)")
        buffer = []
        recording = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        vec = None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            arr = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
            vec = normalize_landmarks(arr)

        if recording:
            if vec is not None:
                buffer.append(vec)
                # energija pokreta = prosječni pomak u odnosu na prethodni frame
                if last_vec is not None:
                    # samo BASE_DIM (položaji) - vidi napomenu u normalize.js motionEnergy()
                    energy = float(np.mean(np.abs(vec[:BASE_DIM] - last_vec[:BASE_DIM])))
                    if energy < MOTION_EPS:
                        if still_since is None:
                            still_since = time.time()
                        elif time.time() - still_since > MOTION_STOP_SECONDS:
                            save()
                    else:
                        still_since = None
                last_vec = vec
                if len(buffer) >= MAX_FRAMES:
                    save()
            else:
                save()  # ruka nestala iz kadra => kraj sekvence

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        ch = chr(key) if 0 < key < 128 else ""
        if ch in LABELS and not recording:
            recording = LABELS[ch]
            buffer, last_vec, still_since = [], None, None
            print(f"Snimam {recording}... izvedi gestu (ista tipka opet = kraj)")
        elif ch in LABELS and recording == LABELS[ch]:
            save()  # ručni kraj: ista tipka ponovno
        elif key == 32 and recording:
            discard()  # SPACE = odbaci

        status = (f"SNIMAM {recording} ({len(buffer)}) - ista tipka=kraj, SPACE=odbaci"
                  if recording else "j / z / o = snimi | ESC = izlaz")
        color = (0, 0, 255) if recording else (30, 30, 30)
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"J:{counts['J']}  Z:{counts['Z']}  OSTALO:{counts['OTHER']}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 220), 2)
        cv2.imshow("Prikupljanje dinamickih znakova", frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    sys.exit(main())
