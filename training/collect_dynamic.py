"""
collect_dynamic.py
------------------
Recording SEQUENCES for the dynamic signs J and Z (+ the OTHER class).

J = hand shaped like the letter I (extended pinky) drawing the arc of the letter J.
Z = extended index finger drawing the letter Z in the air.
OTHER = everything that is NOT J or Z: transitions between static letters, random
        waving, raising/lowering the hand... This class prevents the model from
        labeling every hand movement as J or Z - record it for AT LEAST as much
        as J and Z combined, as varied as possible.

Instructions:
    python collect_dynamic.py --out dynamic_data
    - key 'j' => starts recording one sequence for J; press 'j' AGAIN
      to immediately finish and save it (manually marking the end of the gesture is
      more precise than automatically waiting for stillness). If you don't press anything,
      the recording still closes automatically ~0.5 s after the movement settles (fallback).
    - key 'z' => sequence for Z (same rule: pressing 'z' again = end)
    - key 'o' => sequence for OTHER (same rule: pressing 'o' again = end)
    - SPACE during recording => discard the current sequence without saving
    - ESC to exit

Recommendation: 60-100 sequences for J, the same for Z, 150-200 for OTHER.
Record with both hands, at different speeds and from slightly different angles.
Sequences are saved as .npy files of shape (T, 63) - resampling to a
fixed 30 frames is done only in train_dynamic.py.
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
MOTION_STOP_SECONDS = 0.5   # how much stillness marks the end of a gesture
MOTION_EPS = 0.03           # motion energy threshold (average vector displacement)
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
    recording = None      # None or the class name
    buffer = []           # list of 63-dim vectors
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
            print(f"Saved {recording} #{idx} ({len(buffer)} frames)")
        elif recording:
            print(f"Skipping {recording}: too few frames ({len(buffer)} < {MIN_FRAMES})")
        buffer = []
        recording = None

    def discard():
        nonlocal buffer, recording
        print(f"Discarded {recording} ({len(buffer)} frames)")
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
                # motion energy = average displacement relative to the previous frame
                if last_vec is not None:
                    # only BASE_DIM (positions) - see note in normalize.js motionEnergy()
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
                save()  # hand disappeared from frame => end of sequence

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        ch = chr(key) if 0 < key < 128 else ""
        if ch in LABELS and not recording:
            recording = LABELS[ch]
            buffer, last_vec, still_since = [], None, None
            print(f"Recording {recording}... perform the gesture (same key again = end)")
        elif ch in LABELS and recording == LABELS[ch]:
            save()  # manual end: same key again
        elif key == 32 and recording:
            discard()  # SPACE = discard

        status = (f"RECORDING {recording} ({len(buffer)}) - same key=end, SPACE=discard"
                  if recording else "j / z / o = record | ESC = exit")
        color = (0, 0, 255) if recording else (30, 30, 30)
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"J:{counts['J']}  Z:{counts['Z']}  OTHER:{counts['OTHER']}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 220), 2)
        cv2.imshow("Collecting dynamic signs", frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    sys.exit(main())
