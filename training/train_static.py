"""
train_static.py
---------------
Trains an MLP classifier for static letters (24 classes, excluding J and Z) and exports it
to TensorFlow.js format for use in the browser.

Usage:
    python train_static.py --csv static_landmarks.csv [--csv my_static.csv ...]

Does the following (all useful for the "Evaluation" chapter of the thesis):
  1. loads one or more CSVs and concatenates them
  2. augmentation: mirrored copies (both hands) + slight noise
  3. train/test split (stratified, 80/20)
  4. trains an MLP (63 -> 128 -> 64 -> 24)
  5. comparison with a Random Forest (baseline for the thesis)
  6. saves: confusion matrix (confusion_matrix.png), classification report,
     Keras model (static_model.h5) and TF.js model (../web/models/static/)
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

STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 classes, order = class indices


def load_data(csv_paths):
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["label"].isin(STATIC_LETTERS)]
    X = df[[f"f{i}" for i in range(FEATURE_DIM)]].to_numpy(dtype=np.float32)
    y = np.array([STATIC_LETTERS.index(l) for l in df["label"]], dtype=np.int64)
    return X, y


def augment(X, y):
    """Mirror (for both hands) + noise. Quadruples the dataset."""
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
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title("Confusion matrix - static letters")
        plt.tight_layout(); plt.savefig(path, dpi=150)
        print(f"Confusion matrix: {path}")
    except Exception as e:
        print(f"[!] Skipping matrix plot: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="CSV with landmarks (can be given multiple times)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tfjs-out", default="../web/models/static")
    args = ap.parse_args()

    X, y = load_data(args.csv)
    print(f"Loaded {len(X)} samples, {len(set(y))} classes")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # augment ONLY the training set (the test set must stay "clean")
    X_tr, y_tr = augment(X_tr, y_tr)
    print(f"Training set after augmentation: {len(X_tr)} samples")

    # ---- baseline: Random Forest (for comparison in the thesis) ----
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(X_te))
    print(f"\nRandom Forest accuracy: {rf_acc:.4f}")

    # ---- main model: MLP ----
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
    print(f"\nMLP accuracy on test set: {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=STATIC_LETTERS))
    plot_confusion(confusion_matrix(y_te, y_pred), "confusion_matrix.png")

    # ---- save and export to TensorFlow.js ----
    model.save("static_model.h5")
    os.makedirs(args.tfjs_out, exist_ok=True)
    subprocess.run([sys.executable, "-m", "tensorflowjs.converters.converter",
                    "--input_format=keras", "static_model.h5", args.tfjs_out],
                   check=True)
    # metadata read by the web app (class order!)
    with open(os.path.join(args.tfjs_out, "labels.json"), "w") as f:
        json.dump(STATIC_LETTERS, f)
    print(f"\nTF.js model exported to {args.tfjs_out}")


if __name__ == "__main__":
    main()
