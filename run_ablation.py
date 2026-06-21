"""
Architecture sweep for the CNN-SVM: every combination of 2-6 convolutional and
2-4 dense layers (15 configurations, single seed) on the leakage-free split, to
show how depth affects accuracy. The 5-conv / 4-dense cell is the proposed model.

Only the layer counts change between cells -- the optimizer, loss, augmentation
and other settings are held fixed. Each configuration runs in its own process so
the GPU memory is freed between cells, and the run is resumable via a ledger.
Results go to ablation_summary.csv and ablation_flat.csv.

Usage:
    python run_ablation.py                    # full sweep (set DATA_DIR below first)
    ABLATION_SMOKE=1 python run_ablation.py   # quick check
"""
import os
import gc
import csv
import sys
import json
import time
import subprocess
import traceback
import numpy as np
from pathlib import Path

DATA_DIR   = Path("path/to/dataset")    # leakage-free split (with Training/Validation/Testing)
SMOKE      = os.environ.get("ABLATION_SMOKE") == "1"
OUTPUT_DIR = Path("experiments_ablation_smoke" if SMOKE else "experiments_ablation")
SEED       = 42                       # single seed (the table reports single values)
MAX_EPOCHS = 1 if SMOKE else 100
BATCH_SIZE = 36
IMG_HEIGHT, IMG_WIDTH = 360, 640
N_CONV_LIST  = [2, 3, 4, 5, 6]
N_DENSE_LIST = [2, 3, 4]
CONV_FILTERS = [32, 64, 128, 256, 512, 1024]   # take the first n_conv; conv 1 uses stride 2
DENSE_UNITS  = [1024, 128, 16]                  # take the first (n_dense - 1); + SVM head
USE_CLASS_WEIGHTS = True
CLASS_NAMES = ["Defect", "No_Defect"]           # Defect = class 0 = positive

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_FILE = OUTPUT_DIR / "ledger.json"

# TensorFlow is imported lazily (inside each child process) so the parent driver
# stays light and every child starts with a clean GPU context.
tf = keras = layers = optimizers = regularizers = None
AUTOTUNE = None


def init_tensorflow():
    global tf, keras, layers, optimizers, regularizers, AUTOTUNE
    if tf is not None:
        return
    import tensorflow as _tf
    import keras as _keras
    from keras import layers as _layers, optimizers as _optimizers, regularizers as _regularizers
    tf, keras = _tf, _keras
    layers, optimizers, regularizers = _layers, _optimizers, _regularizers
    AUTOTUNE = tf.data.AUTOTUNE
    available_gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}, GPUs: "
          f"{available_gpus if available_gpus else 'none (running on CPU, this will be slow)'}")
    print(f"Output folder: {OUTPUT_DIR}, max epochs: {MAX_EPOCHS}, seed: {SEED}, smoke test: {SMOKE}", flush=True)
    for gpu in available_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as error:
            print("Could not set memory growth:", error)


def load_split(split_name, shuffle):
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / split_name, image_size=(IMG_HEIGHT, IMG_WIDTH), shuffle=shuffle, seed=42,
        color_mode="rgb", batch_size=BATCH_SIZE, label_mode="binary",
        class_names=CLASS_NAMES, verbose=False)


def squared_hinge_loss(y_true, y_pred):
    signed_labels = 2.0 * y_true - 1.0
    return tf.reduce_mean(tf.square(tf.maximum(0.0, 1.0 - signed_labels * y_pred)), axis=-1)


def svm_accuracy(y_true, y_pred):
    predicted = tf.cast(y_pred > 0.0, y_true.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(predicted, y_true), tf.float32))


def compute_class_weights():
    n_defect    = len(list((DATA_DIR / "Training" / "Defect").glob("*.png")))
    n_no_defect = len(list((DATA_DIR / "Training" / "No_Defect").glob("*.png")))
    if n_defect == 0 or n_no_defect == 0:
        raise SystemExit(f"ERROR: no training images under {DATA_DIR / 'Training'}.")
    total = n_defect + n_no_defect
    weights = {0: total / (2.0 * n_defect), 1: total / (2.0 * n_no_defect)}
    print(f"Training set: {n_defect} defect, {n_no_defect} no-defect tiles "
          f"(class weights {weights[0]:.3f} / {weights[1]:.3f})")
    return weights


def build_variant(n_conv, n_dense):
    """CNN-SVM with n_conv conv layers and n_dense dense layers (last = SVM head).
    At (5, 4) this is identical to the proposed model."""
    model_layers = [layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                    layers.Resizing(180, 320),
                    layers.RandomRotation(0.1),
                    layers.RandomBrightness(0.1)]
    for i in range(n_conv):
        model_layers.append(layers.Conv2D(CONV_FILTERS[i], 3, padding="same",
                                          activation="relu", strides=2 if i == 0 else 1))
        if i % 2 == 0:                       # pool after conv 1, 3, 5
            model_layers.append(layers.MaxPool2D(2, 2))
    model_layers.append(layers.Flatten())
    for units in DENSE_UNITS[:n_dense - 1]:  # hidden dense layers
        model_layers.append(layers.Dense(units, activation="relu",
                                        kernel_regularizer=regularizers.L2(0.01)))
    model_layers.append(layers.Dense(1, kernel_regularizer=regularizers.L2(0.01)))  # linear SVM head
    return keras.Sequential(model_layers, name=f"cnnsvm_c{n_conv}_d{n_dense}")


def confusion_metrics(y_true, y_pred):
    true_labels, predicted_labels = np.asarray(y_true), np.asarray(y_pred)
    TP = int(np.sum((true_labels == 0) & (predicted_labels == 0)))
    FN = int(np.sum((true_labels == 0) & (predicted_labels == 1)))
    FP = int(np.sum((true_labels == 1) & (predicted_labels == 0)))
    TN = int(np.sum((true_labels == 1) & (predicted_labels == 1)))
    total = TP + FN + FP + TN
    ratio = lambda a, b: (a / b) if b else 0.0
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, n=total, accuracy=ratio(TP + TN, total),
                recall=ratio(TP, TP + FN), precision=ratio(TP, TP + FP))


def run_one(n_conv, n_dense):
    init_tensorflow()
    run_id = f"cnnsvm_c{n_conv}_d{n_dense}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    class_weights = compute_class_weights() if USE_CLASS_WEIGHTS else None

    train_data = load_split("Training", shuffle=True).prefetch(AUTOTUNE)
    val_data = load_split("Validation", shuffle=False).prefetch(AUTOTUNE)
    if SMOKE:
        train_data, val_data = train_data.take(6), val_data.take(2)

    model = build_variant(n_conv, n_dense)
    n_params = int(model.count_params())
    model.compile(optimizer=optimizers.SGD(learning_rate=0.005, momentum=0.2,
                                           clipnorm=1.0),
                  loss=squared_hinge_loss, metrics=[svm_accuracy])

    callbacks = [
        keras.callbacks.BackupAndRestore(backup_dir=str(run_dir / "backup")),
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0005,
                                      patience=4, mode="min", verbose=1,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(run_dir / "best.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
    ]
    start = time.perf_counter()
    history = model.fit(train_data, validation_data=val_data, epochs=MAX_EPOCHS,
                        callbacks=callbacks, class_weight=class_weights, verbose=2)
    elapsed = time.perf_counter() - start

    hist = history.history
    epochs_run = len(hist.get("loss", []))
    val_loss = hist.get("val_loss", [])
    best_epoch = int(np.nanargmin(val_loss)) if val_loss and not all(np.isnan(val_loss)) else (len(val_loss) - 1)

    def value_at_best(key):
        values = hist.get(key, [])
        return round(float(values[best_epoch]), 4) if values and 0 <= best_epoch < len(values) else None

    train_acc = value_at_best("svm_accuracy")
    val_acc = value_at_best("val_svm_accuracy")

    # ---- test set, single pass, threshold 0 (Defect = class 0 = positive) ----
    test_data = load_split("Testing", shuffle=False)
    if SMOKE:
        test_data = test_data.take(2)
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in test_data]).astype(int)
    scores = model.predict(test_data, verbose=0).ravel()
    cm = confusion_metrics(y_true, (scores > 0.0).astype(int))

    hours, minutes, seconds = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    result = dict(done=True, n_conv=n_conv, n_dense=n_dense, params=n_params,
                  epochs_run=epochs_run, train_time_sec=round(elapsed, 1),
                  train_time_hms=f"{hours}h {minutes}m {seconds}s",
                  train_acc=train_acc, val_acc=val_acc,
                  test_acc=round(cm["accuracy"], 4),
                  test_recall=round(cm["recall"], 4),
                  test_precision=round(cm["precision"], 4),
                  TP=cm["TP"], FN=cm["FN"], FP=cm["FP"], TN=cm["TN"])

    with open(run_dir / "test_predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["y_true", "y_score"])
        for label, score in zip(y_true, scores):
            writer.writerow([int(label), float(score)])
    json.dump(result, open(run_dir / "metrics.json", "w"), indent=2)
    print(f"  {run_id}: test accuracy {result['test_acc']} (train {train_acc}, val {val_acc}), "
          f"{epochs_run} epochs in {result['train_time_hms']}, {n_params:,} parameters", flush=True)
    return result


def run_single(n_conv, n_dense):
    """Train exactly one config and record it in the ledger (child-process entry)."""
    ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
    run_id = f"cnnsvm_c{n_conv}_d{n_dense}"
    if ledger.get(run_id, {}).get("done"):
        print(f"Skipping {run_id} (already done).")
        return
    try:
        ledger[run_id] = run_one(n_conv, n_dense)
    except Exception as error:
        print(f"{run_id} failed: {error}")
        traceback.print_exc()
        ledger[run_id] = {"done": False, "n_conv": n_conv, "n_dense": n_dense,
                          "error": str(error)[:300]}
    json.dump(ledger, open(LEDGER_FILE, "w"), indent=2)


def write_summary(ledger):
    # grid CSV mirroring the paper table: one block per metric, rows = dense, cols = conv
    rows = [["metric \\ conv"] + [f"{n_conv}conv" for n_conv in N_CONV_LIST]]
    for metric in ("test_acc", "train_acc", "train_time_hms", "epochs_run"):
        rows.append([f"--- {metric} ---"] + [""] * len(N_CONV_LIST))
        for n_dense in N_DENSE_LIST:
            row = [f"{n_dense} dense"]
            for n_conv in N_CONV_LIST:
                record = ledger.get(f"cnnsvm_c{n_conv}_d{n_dense}", {})
                row.append(record.get(metric, "FAIL" if record else "-"))
            rows.append(row)
    with open(OUTPUT_DIR / "ablation_summary.csv", "w", newline="") as file:
        csv.writer(file).writerows(rows)

    flat = [["config", "n_conv", "n_dense", "params", "epochs_run",
             "train_time_hms", "train_time_sec", "train_acc", "val_acc",
             "test_acc", "test_recall", "test_precision", "done"]]
    for n_dense in N_DENSE_LIST:
        for n_conv in N_CONV_LIST:
            record = ledger.get(f"cnnsvm_c{n_conv}_d{n_dense}", {})
            flat.append([f"c{n_conv}_d{n_dense}", n_conv, n_dense, record.get("params"),
                         record.get("epochs_run"), record.get("train_time_hms"),
                         record.get("train_time_sec"), record.get("train_acc"),
                         record.get("val_acc"), record.get("test_acc"),
                         record.get("test_recall"), record.get("test_precision"),
                         record.get("done", False)])
    with open(OUTPUT_DIR / "ablation_flat.csv", "w", newline="") as file:
        csv.writer(file).writerows(flat)
    print("\nWrote ablation_summary.csv and ablation_flat.csv.", flush=True)


def driver():
    """Run each config in its own child process (GPU memory isolation)."""
    # 2-dense row first (it collapses fast, so it's a quick check of every conv
    # count), then the heavier 3- and 4-dense rows.
    order = [(n_conv, n_dense) for n_dense in N_DENSE_LIST for n_conv in N_CONV_LIST]
    print(f"Running {len(order)} configurations, one process each "
          f"(smoke test: {SMOKE}, output folder: {OUTPUT_DIR}).", flush=True)
    for (n_conv, n_dense) in order:
        run_id = f"cnnsvm_c{n_conv}_d{n_dense}"
        ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
        if ledger.get(run_id, {}).get("done"):
            print(f"Skipping {run_id} (already done).", flush=True)
            continue
        print(f"\n----- {run_id} -----", flush=True)
        start = time.perf_counter()
        child = subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
                                "--one", str(n_conv), str(n_dense)])
        elapsed = time.perf_counter() - start
        if child.returncode != 0:
            # child hard-aborted (e.g. out of memory) before it could record itself
            print(f"{run_id}: process exited with code {child.returncode} after {elapsed:.0f}s", flush=True)
            ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
            if not ledger.get(run_id, {}).get("done"):
                ledger[run_id] = {"done": False, "n_conv": n_conv, "n_dense": n_dense,
                                  "error": f"child process rc={child.returncode}"}
                json.dump(ledger, open(LEDGER_FILE, "w"), indent=2)

    ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
    write_summary(ledger)
    print("\nTest-accuracy grid (rows = dense layers, columns = conv layers):", flush=True)
    for n_dense in N_DENSE_LIST:
        cells = [str(ledger.get(f"cnnsvm_c{n_conv}_d{n_dense}", {}).get("test_acc", "FAIL"))
                 for n_conv in N_CONV_LIST]
        print(f"  {n_dense} dense: " + ", ".join(f"{n_conv} conv = {value}"
              for n_conv, value in zip(N_CONV_LIST, cells)), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    if "--one" in sys.argv:
        index = sys.argv.index("--one")
        run_single(int(sys.argv[index + 1]), int(sys.argv[index + 2]))
    elif "--summary" in sys.argv:
        write_summary(json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {})
    else:
        driver()
