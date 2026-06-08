r"""
run_experiments.py
==================
Robust, resumable training + evaluation harness for the three roof-defect
models (CNN-SVM, AlexNet, GoogLeNet) on the leakage-free `Split` dataset,
with repeated runs for confidence intervals.

Built for multi-day UNATTENDED runs:
  * BackupAndRestore  -> if the process dies mid-run, just launch the SAME
                         command again; it resumes that run from the last
                         finished epoch (optimizer state included).
  * run ledger        -> finished (model, seed) runs are skipped on restart.
  * CSVLogger + JSON  -> per-epoch history and per-run metrics are flushed to
                         disk immediately, so a crash never loses results.
  * per-run try/except-> one failed run never aborts the whole batch.
  * single-pass export-> test_predictions.csv per run (shuffle=False) for the
                         PR/ROC + confusion-matrix figures (no shuffle bug).

HOW TO RUN
  1) First do a SMOKE TEST: set SMOKE_TEST = True below and run it once. It
     trains 1 epoch of 1 model and exports predictions -- this proves the data
     loads, the GPU is used, and the whole pipeline works end to end.
  2) Then set SMOKE_TEST = False and launch the real run:
         python run_experiments.py
     If it ever crashes, run the exact same command again; it continues.

GPU NOTES
  TensorFlow dropped native-Windows GPU support after TF 2.10. On Windows with a
  recent NVIDIA GPU, run this under WSL2:
         python3 -m venv ~/tf && source ~/tf/bin/activate
         pip install "tensorflow[and-cuda]"
         python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
  The startup banner below confirms whether the GPU is detected.
"""

import os
import gc
import csv
import json
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers, losses, metrics, optimizers, regularizers

# =========================== CONFIG (edit me) ============================
# Path to the leakage-free split produced by resplit_dataset.py (with Training/,
# Validation/, Testing/ subfolders). A relative path resolves against the CWD.
DATA   = Path("Binary_Blended - Split")
OUTDIR = Path("experiments")

SMOKE_TEST = True               # run once True (1 epoch, 1 model) to verify, then set False
SEEDS  = [42, 43, 44]           # repeats per model for confidence intervals
EPOCHS = 200                    # early stopping usually caps ~80
BATCH  = 36
IMG_H, IMG_W = 360, 640
USE_CLASS_WEIGHTS = True        # applied consistently to ALL three models
MODELS = ["cnnsvm", "alexnet", "googlenet"]
CLASS_NAMES = ["Defect", "No_Defect"]   # Defect = class 0 = POSITIVE
# ========================================================================

if SMOKE_TEST:
    SEEDS, EPOCHS, MODELS = [42], 1, ["cnnsvm"]

OUTDIR.mkdir(parents=True, exist_ok=True)
LEDGER = OUTDIR / "ledger.json"

# ------------------------------ GPU setup -------------------------------
gpus = tf.config.list_physical_devices("GPU")
print("=" * 64)
print(f" TensorFlow {tf.__version__}")
print(f" GPUs detected: {gpus if gpus else 'NONE  -> training will be SLOW (CPU)'}")
if not gpus:
    print(" !! No GPU. For an RTX 5070 on Windows you need WSL2 +")
    print(" !! 'pip install tensorflow[and-cuda]'.  See the GPU NOTES at top.")
print("=" * 64)
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("set_memory_growth:", e)

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------- data loading -----------------------------
def load_split(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        DATA / split, image_size=(IMG_H, IMG_W), shuffle=shuffle, seed=42,
        color_mode="rgb", batch_size=BATCH, label_mode="binary",
        class_names=CLASS_NAMES, verbose=False)


def squared_hinge_01(y_true, y_pred):
    # squared-hinge (L2-SVM) loss; keeps labels in {0,1} and maps to {-1,+1}
    # INTERNALLY, so Keras class_weight (keyed 0/1) still applies to the SVM.
    yt = 2.0 * y_true - 1.0
    return tf.reduce_mean(tf.square(tf.maximum(0.0, 1.0 - yt * y_pred)), axis=-1)


def compute_class_weights():
    n0 = len(list((DATA / "Training" / "Defect").glob("*.png")))
    n1 = len(list((DATA / "Training" / "No_Defect").glob("*.png")))
    n = n0 + n1
    if n0 == 0 or n1 == 0:
        raise SystemExit(
            f"ERROR: no training images found under {DATA / 'Training'}.\n"
            f"Set DATA (top of file) to the Resplit dataset path on THIS machine.")
    w = {0: n / (2.0 * n0), 1: n / (2.0 * n1)}
    print(f" class balance (train): Defect={n0}, No_Defect={n1}  ->  weights {w}")
    return w


CLASS_WEIGHT = compute_class_weights() if USE_CLASS_WEIGHTS else None


# --------------------------- model architectures ------------------------
def build_cnnsvm():
    # proposed model: resize+augment stem, then convs, linear SVM head (L2)
    return keras.Sequential([
        layers.Input(shape=(IMG_H, IMG_W, 3)),
        layers.Resizing(180, 320),
        layers.RandomRotation(0.1),
        layers.RandomBrightness(0.1),
        layers.Conv2D(32, 3, padding="same", activation="relu", strides=2),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.Conv2D(512, 3, padding="same", activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Flatten(),
        layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.L2(0.01)),
        layers.Dense(1, kernel_regularizer=regularizers.L2(0.01)),  # linear SVM head
    ], name="cnnsvm")


def build_alexnet():
    from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                              BatchNormalization)
    return keras.Sequential([
        layers.Input(shape=(IMG_H, IMG_W, 3)),
        Conv2D(96, 11, strides=4, activation="relu"),
        BatchNormalization(), MaxPooling2D(3, 2),
        Conv2D(256, 5, padding="same", activation="relu"),
        BatchNormalization(), MaxPooling2D(3, 2),
        Conv2D(384, 3, padding="same", activation="relu"),
        Conv2D(384, 3, padding="same", activation="relu"),
        Conv2D(256, 3, padding="same", activation="relu"),
        BatchNormalization(), MaxPooling2D(3, 2),
        Flatten(),
        Dense(4096, activation="relu"), Dropout(0.5),
        Dense(4096, activation="relu"), Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ], name="alexnet")


def _inception(x, f1, f3r, f3, f5r, f5, fpp):
    from keras.layers import Conv2D, MaxPooling2D, Concatenate
    p1 = Conv2D(f1, 1, padding="same", activation="relu")(x)
    p2 = Conv2D(f3r, 1, padding="same", activation="relu")(x)
    p2 = Conv2D(f3, 3, padding="same", activation="relu")(p2)
    p3 = Conv2D(f5r, 1, padding="same", activation="relu")(x)
    p3 = Conv2D(f5, 5, padding="same", activation="relu")(p3)
    p4 = MaxPooling2D(3, strides=1, padding="same")(x)
    p4 = Conv2D(fpp, 1, padding="same", activation="relu")(p4)
    return Concatenate(axis=-1)([p1, p2, p3, p4])


def build_googlenet():
    from keras.layers import (Input, Conv2D, MaxPooling2D, AveragePooling2D,
                              Dropout, Flatten, Dense)
    from keras.models import Model
    inp = Input(shape=(IMG_H, IMG_W, 3))
    x = Conv2D(64, 7, strides=2, padding="same", activation="relu")(inp)
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = Conv2D(64, 1, padding="same", activation="relu")(x)
    x = Conv2D(192, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = _inception(x, 64, 96, 128, 16, 32, 32)    # 3a
    x = _inception(x, 128, 128, 192, 32, 96, 64)  # 3b
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = AveragePooling2D(7, padding="valid")(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="googlenet")


def svm_accuracy(y_true, y_pred):
    # y_true in {0,1}, y_pred = raw decision score; score>0 -> class 1
    pred = tf.cast(y_pred > 0.0, y_true.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(pred, y_true), tf.float32))


# per-model training configuration (each keeps its own tuned hyperparameters)
CONFIG = {
    "cnnsvm":    dict(build=build_cnnsvm, svm=True,  lr=0.005, momentum=0.2,
                      clipnorm=1.0, min_delta=0.0005,
                      loss=squared_hinge_01, metric=svm_accuracy),
    "alexnet":   dict(build=build_alexnet, svm=False, lr=0.002, momentum=0.2,
                      clipnorm=None, min_delta=0.001,
                      loss=losses.BinaryCrossentropy(from_logits=False),
                      metric=metrics.BinaryAccuracy(name="binary_accuracy")),
    "googlenet": dict(build=build_googlenet, svm=False, lr=0.002, momentum=0.2,
                      clipnorm=None, min_delta=0.002,
                      loss=losses.BinaryCrossentropy(from_logits=False),
                      metric=metrics.BinaryAccuracy(name="binary_accuracy")),
}


# --------------------------- metrics (Defect = positive) ----------------
def confusion_metrics(y_true, y_pred):
    # y_true/y_pred in {0,1}; class 0 = Defect = positive
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    TP = int(np.sum((yt == 0) & (yp == 0)))
    FN = int(np.sum((yt == 0) & (yp == 1)))
    FP = int(np.sum((yt == 1) & (yp == 0)))
    TN = int(np.sum((yt == 1) & (yp == 1)))
    n = TP + FN + FP + TN
    safe = lambda a, b: (a / b) if b else 0.0
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, n=n,
                accuracy=safe(TP + TN, n),
                recall=safe(TP, TP + FN), precision=safe(TP, TP + FP),
                FPR=safe(FP, FP + TN), FNR=safe(FN, TP + FN), TNR=safe(TN, FP + TN))


# ------------------------------ one run ---------------------------------
def run_one(model_name, seed):
    run_id = f"{model_name}_seed{seed}"
    run_dir = OUTDIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    keras.backend.clear_session()   # free previous run's GPU graph/memory (prevents cross-run OOM)
    gc.collect()
    keras.utils.set_random_seed(seed)
    cfg = CONFIG[model_name]

    train = load_split("Training", shuffle=True).prefetch(AUTOTUNE)
    val = load_split("Validation", shuffle=False).prefetch(AUTOTUNE)
    if SMOKE_TEST:                       # tiny + fast: just prove the pipeline + GPU
        train, val = train.take(6), val.take(2)

    opt_kw = {}
    if cfg["clipnorm"]:
        opt_kw["clipnorm"] = cfg["clipnorm"]
    model = cfg["build"]()
    model.compile(optimizer=optimizers.SGD(learning_rate=cfg["lr"],
                                           momentum=cfg["momentum"], **opt_kw),
                  loss=cfg["loss"], metrics=[cfg["metric"]])

    cbs = [
        keras.callbacks.BackupAndRestore(backup_dir=str(run_dir / "backup")),
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=cfg["min_delta"],
                                      patience=4, mode="min", verbose=1,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(run_dir / "best.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
    ]
    model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=cbs,
              class_weight=CLASS_WEIGHT, verbose=2)
    model.save(run_dir / "final.keras")

    # ---- evaluate on the FIXED test set, single pass (shuffle=False) ----
    test = load_split("Testing", shuffle=False)
    y_true = np.concatenate([y.numpy().ravel() for _, y in test]).astype(int)
    scores = model.predict(test, verbose=0).ravel()
    thr = 0.0 if cfg["svm"] else 0.5
    y_pred = (scores > thr).astype(int)          # 1 = No_Defect
    res = confusion_metrics(y_true, y_pred)

    with open(run_dir / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y_true", "y_score"])         # y_true: 0=Defect ; y_score raw
        for t, s in zip(y_true, scores):
            w.writerow([int(t), float(s)])
    json.dump(res, open(run_dir / "metrics.json", "w"), indent=2)
    print(f"  [{run_id}] test acc={res['accuracy']:.4f}  recall={res['recall']:.4f}"
          f"  precision={res['precision']:.4f}")
    return res


# ----------------------------- aggregation ------------------------------
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_ci(xs):
    k = len(xs)
    m = sum(xs) / k
    if k < 2:
        return m, 0.0
    sd = (sum((x - m) ** 2 for x in xs) / (k - 1)) ** 0.5
    return m, _T95.get(k - 1, 1.96) * sd / (k ** 0.5)


def aggregate(ledger):
    from collections import defaultdict
    by_model = defaultdict(list)
    for run_id, r in ledger.items():
        if r.get("done"):
            by_model[run_id.rsplit("_seed", 1)[0]].append(r)
    rows = [("model", "runs", "acc_mean", "acc_ci95", "rec_mean", "rec_ci95",
             "prec_mean", "prec_ci95")]
    for model_name, runs in sorted(by_model.items()):
        am, ah = mean_ci([r["accuracy"] for r in runs])
        rm, rh = mean_ci([r["recall"] for r in runs])
        pm, ph = mean_ci([r["precision"] for r in runs])
        rows.append((model_name, len(runs), f"{am:.4f}", f"{ah:.4f}",
                     f"{rm:.4f}", f"{rh:.4f}", f"{pm:.4f}", f"{ph:.4f}"))
        print(f"  {model_name:10s}  n={len(runs)}  "
              f"acc={am:.4f}+/-{ah:.4f}  rec={rm:.4f}+/-{rh:.4f}  "
              f"prec={pm:.4f}+/-{ph:.4f}")
    with open(OUTDIR / "results_summary.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


# ------------------------------- main loop ------------------------------
def main():
    ledger = json.load(open(LEDGER)) if LEDGER.exists() else {}
    for model_name in MODELS:
        for seed in SEEDS:
            run_id = f"{model_name}_seed{seed}"
            if ledger.get(run_id, {}).get("done"):
                print(f"[skip] {run_id} already done")
                continue
            print(f"\n===== {run_id} =====")
            try:
                res = run_one(model_name, seed)
                ledger[run_id] = {"done": True, **res}
            except Exception as e:
                print(f"[FAIL] {run_id}: {e}")
                traceback.print_exc()
                ledger[run_id] = {"done": False, "error": str(e)}
            json.dump(ledger, open(LEDGER, "w"), indent=2)

    print("\n===== SUMMARY (mean +/- 95% CI) =====")
    aggregate(ledger)
    print("\nDone. Send me the 'experiments' folder (or ledger.json +")
    print("results_summary.csv + each run's history.csv / test_predictions.csv).")


if __name__ == "__main__":
    main()
