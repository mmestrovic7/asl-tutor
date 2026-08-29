"""
extract_dynamic_videos.py
--------------------------
Converts a dataset of VIDEO clips of dynamic letters (J, Z) into .npy sequences
compatible with train_dynamic.py - analogous to what extract_static.py does
for static letters, just for video instead of images.

Expected dataset structure (e.g. Kaggle "ASL Sign Language Alphabet
Videos [J, Z]" - reorganize the files into this structure if the dataset comes
laid out differently):
    dataset/
        J/  clip001.mp4, clip002.mp4, ...
        Z/  clip001.mp4, ...

Each video is treated as ONE gesture from the start to the end of the clip (no
motion detection - the assumption is that the clip is already trimmed to just
the gesture, which is typical for datasets like this). Frames without a
detected hand are skipped; a clip with too few detected frames is discarded.

Usage:
    python extract_dynamic_videos.py --dataset path/to/dataset --out dynamic_data

Sequences are saved as dynamic_data/<LABEL>/<LABEL>_kaggle_XXXX.npy -
continuing the numbering from existing files (same convention as
collect_dynamic.py), so they can be freely combined with manually recorded
samples. After this, run train_dynamic.py as usual.
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import normalize_landmarks

LABELS = ["J", "Z"]
MIN_FRAMES = 8   # same threshold as MIN_FRAMES in collect_dynamic.py
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
    ap.add_argument("--dataset", required=True, help="Root directory of the video dataset")
    ap.add_argument("--out", default="dynamic_data")
    ap.add_argument("--max-per-class", type=int, default=500,
                    help="Maximum number of clips per letter (for speed)")
    args = ap.parse_args()

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,   # video mode - uses tracking between frames
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
            print(f"[!] Skipping {label} - directory {folder} not found")
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
                print(f"  [!] {name}: skipped ({n_hand}/{n_total} frames with hand, < {MIN_FRAMES})")
                continue
            idx = start_idx + found
            out_path = os.path.join(out_dir, f"{label}_kaggle_{idx:04d}.npy")
            np.save(out_path, seq)
            found += 1
            saved_clips += 1
            print(f"  {name}: saved ({n_hand}/{n_total} frames)")

        print(f"{label}: {found}/{len(files)} clips successfully processed")

    hands.close()
    print(f"\nTotal: {saved_clips}/{total_clips} clips saved to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
