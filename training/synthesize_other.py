"""
synthesize_other.py
--------------------
Generates SYNTHETIC sequences for the OTHER class (dynamic J/Z/OTHER model) from
already existing static landmarks (static_landmarks.csv), without needing
additional camera recording.

Why this makes sense: OTHER needs to cover "hand movement that is NOT J or Z" -
the most common such movement in real usage is the TRANSITION of the hand from
the shape of one letter to the shape of another (e.g. from A to B while the user
spells a word). Such a transition can be approximated by linear interpolation
between two real, measured static vectors - the same math as resample_sequence()
which is used anyway for normalizing the length of real recorded gestures.

Generates two kinds of sequences:
  - "transition" - two DIFFERENT randomly chosen letters, interpolation
    between their vectors over a random number of frames + slight noise
  - "still_change" - the same letter repeated with more pronounced noise -
    simulates small corrections/hand tremor while holding a pose

NOTE: This does NOT replace actually recorded J/Z samples or at least
thirty real OTHER sequences recorded with a camera (ideally combine
both - see collect_dynamic.py) - but it quickly and cheaply fills the class
to the required amount and diversity without any additional recording.

Usage:
    python synthesize_other.py --csv static_landmarks.csv --n 200 --out dynamic_data
"""

import argparse
import os

import numpy as np
import pandas as pd

from landmark_utils import jitter, FEATURE_DIM

STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


def load_vectors(csv_paths):
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    return {
        l: df[df["label"] == l][[f"f{i}" for i in range(FEATURE_DIM)]].to_numpy(dtype=np.float32)
        for l in STATIC_LETTERS
    }


def make_transition(by_letter, rng, min_len=10, max_len=40, sigma=0.015):
    a, b = rng.choice(STATIC_LETTERS, size=2, replace=False)
    va = by_letter[a][rng.integers(len(by_letter[a]))]
    vb = by_letter[b][rng.integers(len(by_letter[b]))]
    T = int(rng.integers(min_len, max_len + 1))
    alphas = np.linspace(0.0, 1.0, T)[:, None]
    seq = (1 - alphas) * va[None, :] + alphas * vb[None, :]
    return np.stack([jitter(f, sigma=sigma) for f in seq]).astype(np.float32)


def make_hold_jitter(by_letter, rng, min_len=10, max_len=30, sigma=0.02):
    a = rng.choice(STATIC_LETTERS)
    v = by_letter[a][rng.integers(len(by_letter[a]))]
    T = int(rng.integers(min_len, max_len + 1))
    return np.stack([jitter(v, sigma=sigma) for _ in range(T)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="static_landmarks CSV (can be given multiple times)")
    ap.add_argument("--n", type=int, default=200, help="Number of OTHER sequences to generate")
    ap.add_argument("--hold-ratio", type=float, default=0.25,
                     help="Fraction of 'still_change' sequences (the rest are transitions)")
    ap.add_argument("--out", default="dynamic_data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    by_letter = load_vectors(args.csv)

    out_dir = os.path.join(args.out, "OTHER")
    os.makedirs(out_dir, exist_ok=True)
    start_idx = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])

    n_hold = int(args.n * args.hold_ratio)
    n_trans = args.n - n_hold
    kinds = ["hold"] * n_hold + ["trans"] * n_trans
    rng.shuffle(kinds)

    for i, kind in enumerate(kinds):
        seq = make_hold_jitter(by_letter, rng) if kind == "hold" else make_transition(by_letter, rng)
        idx = start_idx + i
        np.save(os.path.join(out_dir, f"OTHER_synth_{idx:04d}.npy"), seq)

    print(f"Generated {len(kinds)} synthetic OTHER sequences "
          f"({n_trans} transitions, {n_hold} still) in {out_dir}")


if __name__ == "__main__":
    main()
