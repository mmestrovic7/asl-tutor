"""
train_dynamic.py
----------------
Trenira GRU model za dinamičke znakove: J, Z i OTHER (sve ostalo).

Ulaz:  sekvence iz collect_dynamic.py (dynamic_data/J/*.npy itd.)
Model: (30, 63) -> GRU(64) -> Dense(32) -> softmax(3)
Izlaz: ../web/models/dynamic/ (TF.js format) + labels.json

Pokretanje:
    python train_dynamic.py --data dynamic_data

Zašto GRU, a ne LSTM? GRU ima manje parametara, brže trenira, a na malim
skupovima sekvenci tipično postiže jednake rezultate — dobar argument za rad
(može se navesti i eksperimentalna usporedba: samo zamijeni GRU s LSTM slojem).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from landmark_utils import resample_sequence, mirror_vector

LABELS = ["J", "Z", "OTHER"]   # redoslijed = indeksi klasa (čita ga i web app)
SEQ_LEN = 30


def load_sequences(root):
    X, y = [], []
    for i, lab in enumerate(LABELS):
        folder = os.path.join(root, lab)
        if not os.path.isdir(folder):
            print(f"[!] Nema direktorija {folder}")
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".npy"):
                continue
            seq = np.load(os.path.join(folder, name))       # (T, 63)
            X.append(resample_sequence(seq, SEQ_LEN))       # (30, 63)
            y.append(i)
        print(f"{lab}: {sum(1 for v in y if v == i)} sekvenci")
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def augment(X, y):
    """Zrcalne kopije (obje ruke) + vremenski šum (blago ubrzanje/usporenje)."""
    X_mir = np.stack([np.stack([mirror_vector(f) for f in seq]) for seq in X])
    out_X = [X, X_mir]
    out_y = [y, y]
    # vremenska augmentacija: nasumično odreži 10-20% s početka ili kraja pa resampliraj
    rng = np.random.default_rng(42)
    crops = []
    for seq in X:
        cut = rng.integers(3, 7)
        if rng.random() < 0.5:
            cropped = seq[cut:]
        else:
            cropped = seq[:-cut]
        crops.append(resample_sequence(cropped, SEQ_LEN))
    out_X.append(np.stack(crops)); out_y.append(y)
    return np.concatenate(out_X), np.concatenate(out_y)


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, 63)),
        keras.layers.GRU(64, return_sequences=False),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dynamic_data")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--tfjs-out", default="../web/models/dynamic")
    args = ap.parse_args()

    X, y = load_sequences(args.data)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, y_tr = augment(X_tr, y_tr)
    print(f"Trening: {len(X_tr)}, test: {len(X_te)}")

    model = build_model()
    model.summary()
    cb = [keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True,
                                        monitor="val_accuracy")]
    model.fit(X_tr, y_tr, validation_split=0.15, epochs=args.epochs,
              batch_size=32, callbacks=cb, verbose=2)

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    print(classification_report(y_te, y_pred, target_names=LABELS))
    print("Matrica konfuzije:\n", confusion_matrix(y_te, y_pred))

    model.save("dynamic_model.h5")
    os.makedirs(args.tfjs_out, exist_ok=True)
    subprocess.run([sys.executable, "-m", "tensorflowjs.converters.converter",
                    "--input_format=keras", "dynamic_model.h5", args.tfjs_out],
                   check=True)
    with open(os.path.join(args.tfjs_out, "labels.json"), "w") as f:
        json.dump(LABELS, f)
    print(f"TF.js model izvezen u {args.tfjs_out}")


if __name__ == "__main__":
    main()
