"""
extract_dynamic_videos.py
--------------------------
Pretvara dataset VIDEO isječaka dinamičkih slova (J, Z) u .npy sekvence
kompatibilne s train_dynamic.py - analogno onome što extract_static.py radi
za statička slova, samo za video umjesto slika.

Očekivana struktura dataseta (npr. Kaggle "ASL Sign Language Alphabet
Videos [J, Z]" - reorganiziraj datoteke u ovu strukturu ako dataset dolazi
drugačije posložen):
    dataset/
        J/  clip001.mp4, clip002.mp4, ...
        Z/  clip001.mp4, ...

Svaki video tretiramo kao JEDNU gestu od početka do kraja klipa (bez
detekcije pokreta - pretpostavka je da je klip već obrezan na samu gestu,
što je uobičajeno za ovakve datasetove). Frameovi bez detektirane ruke se
preskaču; klip s premalo detektiranih frameova se odbacuje.

Pokretanje:
    python extract_dynamic_videos.py --dataset putanja/do/dataseta --out dynamic_data

Sekvence se spremaju kao dynamic_data/<LABEL>/<LABEL>_kaggle_XXXX.npy -
nastavlja brojanje od postojećih datoteka (isti konvencija kao
collect_dynamic.py), pa se mogu bezbrižno spojiti s ručno snimljenim
uzorcima. Nakon ovoga pokreni train_dynamic.py kao i inače.
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import normalize_landmarks

LABELS = ["J", "Z"]
MIN_FRAMES = 8   # ista granica kao MIN_FRAMES u collect_dynamic.py
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def extract_sequence(path, hands):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0, 0

    frames = []
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            continue
        lm = res.multi_hand_landmarks[0].landmark
        arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        frames.append(normalize_landmarks(arr))
    cap.release()

    if len(frames) < MIN_FRAMES:
        return None, len(frames), total
    return np.stack(frames).astype(np.float32), len(frames), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Korijenski direktorij video dataseta")
    ap.add_argument("--out", default="dynamic_data")
    ap.add_argument("--max-per-class", type=int, default=500,
                    help="Maksimalan broj klipova po slovu (za brzinu)")
    args = ap.parse_args()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,   # video mod - koristi tracking između frameova
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for lab in LABELS:
        os.makedirs(os.path.join(args.out, lab), exist_ok=True)

    total_clips, saved_clips = 0, 0
    for label in LABELS:
        folder = os.path.join(args.dataset, label)
        if not os.path.isdir(folder):
            print(f"[!] Preskačem {label} - nema direktorija {folder}")
            continue

        out_dir = os.path.join(args.out, label)
        start_idx = len(os.listdir(out_dir))
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS))
        files = files[: args.max_per_class]

        found = 0
        for i, name in enumerate(files):
            path = os.path.join(folder, name)
            seq, n_hand, n_total = extract_sequence(path, hands)
            total_clips += 1
            if seq is None:
                print(f"  [!] {name}: preskočeno ({n_hand}/{n_total} frameova s rukom, < {MIN_FRAMES})")
                continue
            idx = start_idx + found
            out_path = os.path.join(out_dir, f"{label}_kaggle_{idx:04d}.npy")
            np.save(out_path, seq)
            found += 1
            saved_clips += 1
            print(f"  {name}: spremljeno ({n_hand}/{n_total} frameova)")

        print(f"{label}: {found}/{len(files)} klipova uspješno obrađeno")

    hands.close()
    print(f"\nUkupno: {saved_clips}/{total_clips} klipova spremljeno u {args.out}")


if __name__ == "__main__":
    sys.exit(main())
