"""
train_static.py
---------------
Trenira MLP klasifikator statičkih slova (24 klase, bez J i Z) i izvozi ga
u TensorFlow.js format za korištenje u browseru.

Pokretanje:
    python train_static.py --csv static_landmarks.csv [--csv my_static.csv ...]

Radi sljedeće (sve korisno za poglavlje "Evaluacija" u radu):
  1. učitava jedan ili više CSV-ova i spaja ih
  2. augmentacija: zrcalne kopije (obje ruke) + blagi šum
  3. train/test podjela (stratificirana, 80/20)
  4. trenira MLP (63 -> 128 -> 64 -> 24)
  5. usporedba s Random Forestom (baseline za rad)
  6. sprema: matricu konfuzije (confusion_matrix.png), classification report,
     Keras model (static_model.h5) i TF.js model (../web/models/static/)
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

from landmark_utils import mirror_vector, jitter, FEATURE_DIM

STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 klase, redoslijed = indeksi klasa


def load_data(csv_paths):
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["label"].isin(STATIC_LETTERS)]
    X = df[[f"f{i}" for i in range(FEATURE_DIM)]].to_numpy(dtype=np.float32)
    y = np.array([STATIC_LETTERS.index(l) for l in df["label"]], dtype=np.int64)
    return X, y


def augment(X, y):
    """Zrcalo (za obje ruke) + šum. Učetverostručuje skup."""
    Xm = np.stack([mirror_vector(v) for v in X])
    Xj = np.stack([jitter(v) for v in X])
    Xmj = np.stack([jitter(v) for v in Xm])
    return (np.concatenate([X, Xm, Xj, Xmj]),
            np.concatenate([y, y, y, y]))


def build_mlp():
    model = keras.Sequential([
        keras.layers.Input(shape=(FEATURE_DIM,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(len(STATIC_LETTERS), activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def plot_confusion(cm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(11, 9))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=STATIC_LETTERS, yticklabels=STATIC_LETTERS)
        plt.xlabel("Predviđeno"); plt.ylabel("Stvarno")
        plt.title("Matrica konfuzije — statička slova")
        plt.tight_layout(); plt.savefig(path, dpi=150)
        print(f"Matrica konfuzije: {path}")
    except Exception as e:
        print(f"[!] Preskačem crtanje matrice: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="CSV s landmarkovima (može više puta)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tfjs-out", default="../web/models/static")
    args = ap.parse_args()

    X, y = load_data(args.csv)
    print(f"Učitano {len(X)} uzoraka, {len(set(y))} klasa")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # augmentiramo SAMO trening skup (test mora ostati "čist")
    X_tr, y_tr = augment(X_tr, y_tr)
    print(f"Trening nakon augmentacije: {len(X_tr)} uzoraka")

    # ---- baseline: Random Forest (za usporedbu u radu) ----
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te))
    print(f"\nRandom Forest točnost: {rf_acc:.4f}")

    # ---- glavni model: MLP ----
    model = build_mlp()
    model.summary()
    cb = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                      monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]
    model.fit(X_tr, y_tr, validation_split=0.1, epochs=args.epochs,
              batch_size=128, callbacks=cb, verbose=2)

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    acc = accuracy_score(y_te, y_pred)
    print(f"\nMLP točnost na test skupu: {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=STATIC_LETTERS))
    plot_confusion(confusion_matrix(y_te, y_pred), "confusion_matrix.png")

    # ---- spremanje i izvoz u TensorFlow.js ----
    model.save("static_model.h5")
    os.makedirs(args.tfjs_out, exist_ok=True)
    subprocess.run([sys.executable, "-m", "tensorflowjs.converters.converter",
                    "--input_format=keras", "static_model.h5", args.tfjs_out],
                   check=True)
    # metapodaci koje web aplikacija čita (redoslijed klasa!)
    with open(os.path.join(args.tfjs_out, "labels.json"), "w") as f:
        json.dump(STATIC_LETTERS, f)
    print(f"\nTF.js model izvezen u {args.tfjs_out}")


if __name__ == "__main__":
    main()
