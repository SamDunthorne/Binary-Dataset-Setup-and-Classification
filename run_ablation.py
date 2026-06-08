r"""
run_ablation.py
===============
Architecture sweep for the proposed CNN-SVM model: number of convolutional
layers (2-6) x number of dense layers (2-4) = 15 configurations, single seed,
to (re)populate the architecture table (tab:ModelPerformance) on the
LEAKAGE-FREE Resplit dataset for the Journal Paper #1 revision.

Scheme (reduces EXACTLY to the chosen 5-conv/4-dense build_cnnsvm() at (5,4);
verified against the paper's reported Dense-1 param count = 115,344,384):
  * conv filters : first n_conv of [32,64,128,256,512,1024]; conv1 strides=2
  * MaxPool      : after conv 1,3,5  (i % 2 == 0)
  * hidden dense : first (n_dense-1) of [1024,128,16]; then a LINEAR SVM head
  * stem + hp    : same Resizing(180,320)+aug stem, SGD(lr .005, mom .2,
                   clipnorm 1.0), squared-hinge (L2-SVM) loss, threshold-0 eval,
                   class weights, EarlyStopping(min_delta .0005, patience 4).
Only the layer COUNTS change vs the proposed model -- no other hyperparameter.

Resumable: ledger skips done configs; a child that hard-aborts (OOM SIGABRT) is
recorded done=false by the driver and never blocks the rest. Writes
ablation_summary.csv (the table grid) + ablation_flat.csv.

RUN (from WSL, venv active):
  source ~/tf-gpu/bin/activate
  TF_GPU_ALLOCATOR=cuda_malloc_async python -u run_ablation.py          # full sweep (driver)
  ABLATION_SMOKE=1 TF_GPU_ALLOCATOR=cuda_malloc_async python -u run_ablation.py   # smoke
  python -u run_ablation.py --one 5 4    # (internal) train exactly one config
  python -u run_ablation.py --summary    # rebuild summary csvs from the ledger
"""
import os
import gc
import csv
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path

import numpy as np

# =============================== CONFIG ================================
DATA   = Path("Binary_Blended - Split")    # leakage-free split (relative to CWD)
SMOKE  = os.environ.get("ABLATION_SMOKE") == "1"
OUTDIR = Path("experiments_ablation_smoke" if SMOKE else "experiments_ablation")
SEED   = 42                       # single seed (the table reports single values)
EPOCHS = 1 if SMOKE else 100      # paper: "capped at 100 epochs"
BATCH  = 36
IMG_H, IMG_W = 360, 640
N_CONV_LIST  = [2, 3, 4, 5, 6]
N_DENSE_LIST = [2, 3, 4]
CONV_FILTERS = [32, 64, 128, 256, 512, 1024]   # take first n_conv; conv1 strides=2
DENSE_UNITS  = [1024, 128, 16]                  # take first (n_dense-1); + SVM head
USE_CLASS_WEIGHTS = True
CLASS_NAMES = ["Defect", "No_Defect"]           # Defect = class 0 = POSITIVE
# ======================================================================

OUTDIR.mkdir(parents=True, exist_ok=True)
LEDGER = OUTDIR / "ledger.json"

# TF handles are initialised lazily, ONLY in per-config child processes, so the
# parent driver never creates a CUDA context.
tf = keras = layers = optimizers = regularizers = None
AUTOTUNE = None


def _init_tf():
    global tf, keras, layers, optimizers, regularizers, AUTOTUNE
    if tf is not None:
        return
    import tensorflow as _tf
    import keras as _keras
    from keras import layers as _layers, optimizers as _optimizers, regularizers as _regularizers
    tf, keras = _tf, _keras
    layers, optimizers, regularizers = _layers, _optimizers, _regularizers
    AUTOTUNE = tf.data.AUTOTUNE
    gpus = tf.config.list_physical_devices("GPU")
    print("=" * 64)
    print(f" TF {tf.__version__}   GPUs: {gpus if gpus else 'NONE -> CPU (SLOW)'}")
    print(f" SMOKE={SMOKE}  OUTDIR={OUTDIR}  EPOCHS={EPOCHS}  SEED={SEED}")
    print("=" * 64, flush=True)
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("set_memory_growth:", e)


def load_split(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        DATA / split, image_size=(IMG_H, IMG_W), shuffle=shuffle, seed=42,
        color_mode="rgb", batch_size=BATCH, label_mode="binary",
        class_names=CLASS_NAMES, verbose=False)


def squared_hinge_01(y_true, y_pred):
    yt = 2.0 * y_true - 1.0
    return tf.reduce_mean(tf.square(tf.maximum(0.0, 1.0 - yt * y_pred)), axis=-1)


def svm_accuracy(y_true, y_pred):
    pred = tf.cast(y_pred > 0.0, y_true.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(pred, y_true), tf.float32))


def compute_class_weights():
    n0 = len(list((DATA / "Training" / "Defect").glob("*.png")))
    n1 = len(list((DATA / "Training" / "No_Defect").glob("*.png")))
    if n0 == 0 or n1 == 0:
        raise SystemExit(f"ERROR: no training images under {DATA / 'Training'}.")
    n = n0 + n1
    w = {0: n / (2.0 * n0), 1: n / (2.0 * n1)}
    print(f" class balance (train): Defect={n0}, No_Defect={n1}  ->  weights {w}")
    return w


def build_variant(n_conv, n_dense):
    """CNN-SVM with n_conv conv layers and n_dense dense layers (last = SVM head).
    Reduces byte-for-byte to build_cnnsvm() at (5, 4)."""
    L = [layers.Input(shape=(IMG_H, IMG_W, 3)),
         layers.Resizing(180, 320),
         layers.RandomRotation(0.1),
         layers.RandomBrightness(0.1)]
    for i in range(n_conv):
        L.append(layers.Conv2D(CONV_FILTERS[i], 3, padding="same",
                               activation="relu", strides=2 if i == 0 else 1))
        if i % 2 == 0:                       # pool after conv 1, 3, 5
            L.append(layers.MaxPool2D(2, 2))
    L.append(layers.Flatten())
    for u in DENSE_UNITS[:n_dense - 1]:      # hidden dense layers
        L.append(layers.Dense(u, activation="relu",
                              kernel_regularizer=regularizers.L2(0.01)))
    L.append(layers.Dense(1, kernel_regularizer=regularizers.L2(0.01)))  # linear SVM head
    return keras.Sequential(L, name=f"cnnsvm_c{n_conv}_d{n_dense}")


def confusion_metrics(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    TP = int(np.sum((yt == 0) & (yp == 0)))
    FN = int(np.sum((yt == 0) & (yp == 1)))
    FP = int(np.sum((yt == 1) & (yp == 0)))
    TN = int(np.sum((yt == 1) & (yp == 1)))
    n = TP + FN + FP + TN
    safe = lambda a, b: (a / b) if b else 0.0
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, n=n, accuracy=safe(TP + TN, n),
                recall=safe(TP, TP + FN), precision=safe(TP, TP + FP))


def run_one(n_conv, n_dense):
    _init_tf()
    rid = f"cnnsvm_c{n_conv}_d{n_dense}"
    rdir = OUTDIR / rid
    rdir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    cw = compute_class_weights() if USE_CLASS_WEIGHTS else None

    train = load_split("Training", shuffle=True).prefetch(AUTOTUNE)
    val = load_split("Validation", shuffle=False).prefetch(AUTOTUNE)
    if SMOKE:
        train, val = train.take(6), val.take(2)

    model = build_variant(n_conv, n_dense)
    nparams = int(model.count_params())
    model.compile(optimizer=optimizers.SGD(learning_rate=0.005, momentum=0.2,
                                           clipnorm=1.0),
                  loss=squared_hinge_01, metrics=[svm_accuracy])

    cbs = [
        keras.callbacks.BackupAndRestore(backup_dir=str(rdir / "backup")),
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0005,
                                      patience=4, mode="min", verbose=1,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(rdir / "best.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(str(rdir / "history.csv"), append=True),
    ]
    t0 = time.perf_counter()
    h = model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=cbs,
                  class_weight=cw, verbose=2)
    dt = time.perf_counter() - t0

    H = h.history
    epochs_run = len(H.get("loss", []))
    vl = H.get("val_loss", [])
    best_i = int(np.nanargmin(vl)) if vl and not all(np.isnan(vl)) else (len(vl) - 1)

    def at_best(key):
        v = H.get(key, [])
        return round(float(v[best_i]), 4) if v and 0 <= best_i < len(v) else None

    train_acc = at_best("svm_accuracy")
    val_acc = at_best("val_svm_accuracy")

    # ---- test set, single pass, threshold 0 (Defect = class 0 = positive) ----
    test = load_split("Testing", shuffle=False)
    if SMOKE:
        test = test.take(2)
    y_true = np.concatenate([y.numpy().ravel() for _, y in test]).astype(int)
    scores = model.predict(test, verbose=0).ravel()
    cm = confusion_metrics(y_true, (scores > 0.0).astype(int))

    h_, m_, s_ = int(dt // 3600), int((dt % 3600) // 60), int(dt % 60)
    res = dict(done=True, n_conv=n_conv, n_dense=n_dense, params=nparams,
               epochs_run=epochs_run, train_time_sec=round(dt, 1),
               train_time_hms=f"{h_}h {m_}m {s_}s",
               train_acc=train_acc, val_acc=val_acc,
               test_acc=round(cm["accuracy"], 4),
               test_recall=round(cm["recall"], 4),
               test_precision=round(cm["precision"], 4),
               TP=cm["TP"], FN=cm["FN"], FP=cm["FP"], TN=cm["TN"])

    with open(rdir / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y_true", "y_score"])
        for t, s in zip(y_true, scores):
            w.writerow([int(t), float(s)])
    json.dump(res, open(rdir / "metrics.json", "w"), indent=2)
    print(f"  [{rid}] params={nparams:,}  ep={epochs_run}  "
          f"time={res['train_time_hms']}  train_acc={train_acc}  "
          f"val_acc={val_acc}  test_acc={res['test_acc']}", flush=True)
    return res


def run_single(n_conv, n_dense):
    """Train exactly one config and record it in the ledger (child-process entry)."""
    ledger = json.load(open(LEDGER)) if LEDGER.exists() else {}
    rid = f"cnnsvm_c{n_conv}_d{n_dense}"
    if ledger.get(rid, {}).get("done"):
        print(f"[skip] {rid} already done")
        return
    try:
        res = run_one(n_conv, n_dense)
        ledger[rid] = res
    except Exception as e:
        print(f"[FAIL] {rid}: {e}")
        traceback.print_exc()
        ledger[rid] = {"done": False, "n_conv": n_conv, "n_dense": n_dense,
                       "error": str(e)[:300]}
    json.dump(ledger, open(LEDGER, "w"), indent=2)


def write_summary(ledger):
    # grid CSV mirroring the paper table: one block per metric, rows=dense, cols=conv
    rows = [["metric \\ conv"] + [f"{c}conv" for c in N_CONV_LIST]]
    for metric in ("test_acc", "train_acc", "train_time_hms", "epochs_run"):
        rows.append([f"--- {metric} ---"] + [""] * len(N_CONV_LIST))
        for d in N_DENSE_LIST:
            row = [f"{d} dense"]
            for c in N_CONV_LIST:
                r = ledger.get(f"cnnsvm_c{c}_d{d}", {})
                row.append(r.get(metric, "FAIL" if r else "—"))
            rows.append(row)
    with open(OUTDIR / "ablation_summary.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    flat = [["config", "n_conv", "n_dense", "params", "epochs_run",
             "train_time_hms", "train_time_sec", "train_acc", "val_acc",
             "test_acc", "test_recall", "test_precision", "done"]]
    for d in N_DENSE_LIST:
        for c in N_CONV_LIST:
            r = ledger.get(f"cnnsvm_c{c}_d{d}", {})
            flat.append([f"c{c}_d{d}", c, d, r.get("params"), r.get("epochs_run"),
                         r.get("train_time_hms"), r.get("train_time_sec"),
                         r.get("train_acc"), r.get("val_acc"), r.get("test_acc"),
                         r.get("test_recall"), r.get("test_precision"),
                         r.get("done", False)])
    with open(OUTDIR / "ablation_flat.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(flat)
    print("\nWrote ablation_summary.csv + ablation_flat.csv", flush=True)


def driver():
    """Run each config in its own child process (GPU memory isolation)."""
    # 2-dense row first (collapses fast -> quick all-conv-count validation),
    # then the heavier 3- and 4-dense rows.
    order = [(c, d) for d in N_DENSE_LIST for c in N_CONV_LIST]
    print(f"[driver] {len(order)} configs, one process each. "
          f"SMOKE={SMOKE} OUTDIR={OUTDIR}", flush=True)
    for (c, d) in order:
        rid = f"cnnsvm_c{c}_d{d}"
        led = json.load(open(LEDGER)) if LEDGER.exists() else {}
        if led.get(rid, {}).get("done"):
            print(f"[driver][skip] {rid} already done", flush=True)
            continue
        print(f"\n########## [driver] {rid} ##########", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
                            "--one", str(c), str(d)])
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            # child hard-aborted (e.g. OOM SIGABRT) before it could record itself
            print(f"[driver][child-died] {rid} rc={r.returncode} after {dt:.0f}s", flush=True)
            led = json.load(open(LEDGER)) if LEDGER.exists() else {}
            if not led.get(rid, {}).get("done"):
                led[rid] = {"done": False, "n_conv": c, "n_dense": d,
                            "error": f"child process rc={r.returncode}"}
                json.dump(led, open(LEDGER, "w"), indent=2)

    led = json.load(open(LEDGER)) if LEDGER.exists() else {}
    write_summary(led)
    print("\n===== ABLATION SUMMARY (test_acc grid) =====", flush=True)
    for d in N_DENSE_LIST:
        cells = [str(led.get(f"cnnsvm_c{c}_d{d}", {}).get("test_acc", "FAIL"))
                 for c in N_CONV_LIST]
        print("  %d dense: " % d + "  ".join(f"{c}c={v}"
              for c, v in zip(N_CONV_LIST, cells)), flush=True)
    print("[driver] done.", flush=True)


if __name__ == "__main__":
    if "--one" in sys.argv:
        i = sys.argv.index("--one")
        run_single(int(sys.argv[i + 1]), int(sys.argv[i + 2]))
    elif "--summary" in sys.argv:
        write_summary(json.load(open(LEDGER)) if LEDGER.exists() else {})
    else:
        driver()
