"""
extract_static.py
-----------------
Walks through a dataset of static ASL letter images and converts it into a CSV of landmarks.

Expected dataset structure (e.g. Kaggle "ASL Alphabet"):
    dataset/
        A/  img001.jpg, img002.jpg, ...
        B/  ...
        ...
        Y/  ...
(J and Z are skipped - they are dynamic and are handled in collect_dynamic.py)

Usage:
    python extract_static.py --dataset path/to/dataset --out static_landmarks.csv

Output: CSV with 64 columns - 'label' + f0..f62 (normalized landmarks).
Note: the percentage of images where MediaPipe does NOT detect a hand
(printed at the end) is an interesting figure for the dataset chapter.
"""

import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import normalize_landmarks, FEATURE_DIM

STATIC_LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # excluding J and Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Root directory of the dataset")
    ap.add_argument("--out", default="static_landmarks.csv")
    ap.add_argument("--max-per-class", type=int, default=3000,
                    help="Maximum number of images per letter (for speed)")
    args = ap.parse_args()

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,      # each image independently (no tracking)
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    total, detected = 0, 0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM)])

        for letter in STATIC_LETTERS:
            folder = os.path.join(args.dataset, letter)
            if not os.path.isdir(folder):
                print(f"[!] Skipping {letter} - directory {folder} not found")
                continue

            files = sorted(os.listdir(folder))[: args.max_per_class]
            found = 0
            for name in files:
                path = os.path.join(folder, name)
                img = cv2.imread(path)
                if img is None:
                    continue
                total += 1
                # MediaPipe expects RGB, OpenCV reads BGR
                res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not res.multi_hand_landmarks:
                    continue
                lm = res.multi_hand_landmarks[0].landmark
                arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                vec = normalize_landmarks(arr)
                writer.writerow([letter] + [f"{v:.6f}" for v in vec])
                detected += 1
                found += 1

            print(f"{letter}: {found}/{len(files)} images with detected hand")

    hands.close()
    if total:
        print(f"\nTotal: {detected}/{total} ({100*detected/total:.1f}%) successful detections")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
