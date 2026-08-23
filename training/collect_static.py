"""
collect_static.py
-----------------
Snimanje VLASTITIH uzoraka statičkih slova preko web kamere.
Jako preporučeno uz Kaggle dataset: model treniran i na tvojoj kameri,
tvom osvjetljenju i tvojoj ruci bit će osjetno točniji u aplikaciji.
(U radu: "kombinacija javno dostupnog i vlastito prikupljenog skupa podataka")

Upute:
    python collect_static.py --out my_static.csv
    - pokaži znak u kameru
    - drži tipku slova (npr. 'a') - svaki frame dok držiš postaje jedan uzorak
      (30 fps => ~150 uzoraka za 5 sekundi držanja)
    - ESC za izlaz

CSV format identičan extract_static.py, pa se datoteke mogu jednostavno spojiti:
    python -c "import pandas as pd; pd.concat([pd.read_csv('static_landmarks.csv'), pd.read_csv('my_static.csv')]).to_csv('combined.csv', index=False)"
"""

import argparse
import csv
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
        static_image_mode=False,  # video mod - koristi tracking, brže i stabilnije
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    drawer = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    counts = {c: 0 for c in STATIC_LETTERS}

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_DIM)])

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # zrcalo, prirodnije za korisnika
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
            cv2.putText(frame, "Drzi tipku slova za snimanje | ESC izlaz", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
            cv2.putText(frame, info[:90], (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 220), 1)
            cv2.imshow("Prikupljanje statickih znakova", frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print({c: n for c, n in counts.items() if n})


if __name__ == "__main__":
    sys.exit(main())
