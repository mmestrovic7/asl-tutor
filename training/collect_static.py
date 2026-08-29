"""
collect_static.py
-----------------
Recording YOUR OWN samples of static letters via webcam.
Highly recommended alongside the Kaggle dataset: a model also trained on
your camera, your lighting and your hand will be noticeably more accurate
in the application.
(In the thesis: "combination of a publicly available and a self-collected dataset")

Instructions:
    python collect_static.py --out my_static.csv
    - show the sign to the camera
    - hold down the letter's key (e.g. 'a') - every frame while held becomes one sample
      (30 fps => ~150 samples for 5 seconds of holding)
    - ESC to exit

If --out already exists, new samples are APPENDED to the existing ones (not deleted),
so you can run the script multiple times in a row without fear of losing
previous recordings.

CSV format identical to extract_static.py, so the files can simply be merged:
    python -c "import pandas as pd; pd.concat([pd.read_csv('static_landmarks.csv'), pd.read_csv('my_static.csv')]).to_csv('combined.csv', index=False)"
"""

import argparse
import csv
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import normalize_landmarks, FEATURE_DIM

STATIC_LETTERS = set("ABCDEFGHIKLMNOPQRSTUVWXY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="my_static.csv")
    args = ap.parse_args()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,  # video mode - uses tracking, faster and more stable
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    drawer = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    counts = {c: 0 for c in STATIC_LETTERS}

    file_exists = os.path.isfile(args.out) and os.path.getsize(args.out) > 0
    if file_exists:
        with open(args.out, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
        print(f"Continuing on existing {args.out}: {sum(counts.values())} samples already recorded")

    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM)])

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, more natural for the user
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            vec = None
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                arr = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
                vec = normalize_landmarks(arr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            ch = chr(key).upper() if 0 < key < 128 else ""
            if ch in STATIC_LETTERS and vec is not None:
                writer.writerow([ch] + [f"{v:.6f}" for v in vec])
                counts[ch] += 1

            info = " ".join(f"{c}:{n}" for c, n in sorted(counts.items()) if n)
            cv2.putText(frame, "Hold a letter key to record | ESC to exit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
            cv2.putText(frame, info[:90], (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 220), 1)
            cv2.imshow("Collecting static signs", frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print({c: n for c, n in counts.items() if n})


if __name__ == "__main__":
    sys.exit(main())
