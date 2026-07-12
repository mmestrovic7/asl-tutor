"""
extract_static.py
-----------------
Prolazi kroz dataset slika statičkih ASL slova i pretvara ga u CSV landmarkova.

Očekivana struktura dataseta (npr. Kaggle "ASL Alphabet"):
    dataset/
        A/  img001.jpg, img002.jpg, ...
        B/  ...
        ...
        Y/  ...
(J i Z se preskaču — oni su dinamički i obrađuju se u collect_dynamic.py)

Pokretanje:
    python extract_static.py --dataset putanja/do/dataseta --out static_landmarks.csv

Izlaz: CSV s 64 stupca — 'label' + f0..f62 (normalizirani landmarkovi).
Napomena za rad: postotak slika na kojima MediaPipe NE detektira ruku
(ispisuje se na kraju) zanimljiv je podatak za poglavlje o skupu podataka.
"""

import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import normalize_landmarks

STATIC_LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # bez J i Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Korijenski direktorij dataseta")
    ap.add_argument("--out", default="static_landmarks.csv")
    ap.add_argument("--max-per-class", type=int, default=3000,
                    help="Maksimalan broj slika po slovu (za brzinu)")
    args = ap.parse_args()

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,      # svaka slika neovisno (bez trackinga)
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    total, detected = 0, 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])

        for letter in STATIC_LETTERS:
            folder = os.path.join(args.dataset, letter)
            if not os.path.isdir(folder):
                print(f"[!] Preskačem {letter} — nema direktorija {folder}")
                continue

            files = sorted(os.listdir(folder))[: args.max_per_class]
            found = 0
            for name in files:
                path = os.path.join(folder, name)
                img = cv2.imread(path)
                if img is None:
                    continue
                total += 1
                # MediaPipe očekuje RGB, OpenCV čita BGR
                res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not res.multi_hand_landmarks:
                    continue
                lm = res.multi_hand_landmarks[0].landmark
                arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                vec = normalize_landmarks(arr)
                writer.writerow([letter] + [f"{v:.6f}" for v in vec])
                detected += 1
                found += 1

            print(f"{letter}: {found}/{len(files)} slika s detektiranom rukom")

    hands.close()
    if total:
        print(f"\nUkupno: {detected}/{total} ({100*detected/total:.1f}%) uspješnih detekcija")
    print(f"Spremljeno u {args.out}")


if __name__ == "__main__":
    sys.exit(main())
