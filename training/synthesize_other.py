"""
synthesize_other.py
--------------------
Generira SINTETICKE sekvence za klasu OSTALO (dinamicki model J/Z/OSTALO) iz
vec postojecih statickih landmarkova (static_landmarks.csv), bez potrebe za
dodatnim snimanjem kamerom.

Zasto ovo ima smisla: OSTALO treba pokriti "pokret ruke koji NIJE J ni Z" —
najcesci takav pokret u stvarnoj upotrebi je PRIJELAZ ruke iz oblika jednog
slova u oblik drugog slova (npr. iz A u B dok korisnik srice rijec). Takav
prijelaz se moze aproksimirati linearnom interpolacijom izmedju dva stvarna,
izmjerena staticka vektora — ista matematika kao resample_sequence() koja
se ionako koristi za normalizaciju duljine pravih snimljenih gesti.

Generira dvije vrste sekvenci:
  - "prijelaz" — dva RAZLICITA nasumicno odabrana slova, interpolacija
    izmedju njihovih vektora kroz nasumican broj frameova + blagi sum
  - "mirna_promjena" — jedno te isto slovo, ponovljeno uz izrazeniji sum —
    simulira sitne korekcije/drhtanje ruke tijekom zadrzavanja pokreta

NAPOMENA: Ovo NE zamjenjuje stvarno snimljene J/Z primjere niti barem
tridesetak pravih OSTALO sekvenci snimljenih kamerom (idealno kombinirati
oboje — vidi collect_dynamic.py) — ali brzo i besplatno popuni klasu do
potrebne kolicine i raznolikosti bez ijednog dodatnog snimanja.

Pokretanje:
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
    ap.add_argument("--csv", action="append", required=True, help="static_landmarks CSV (moze vise puta)")
    ap.add_argument("--n", type=int, default=200, help="Broj OSTALO sekvenci za generiranje")
    ap.add_argument("--hold-ratio", type=float, default=0.25,
                     help="Udio 'mirna_promjena' sekvenci (ostatak su prijelazi)")
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

    print(f"Generirano {len(kinds)} sintetickih OSTALO sekvenci "
          f"({n_trans} prijelaza, {n_hold} mirnih) u {out_dir}")


if __name__ == "__main__":
    main()
