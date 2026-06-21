"""
Leave-one-out ablation of the CNN-SVM training recipe on the leakage-free split.

Starting from the proposed model (SGD + augmentation + class weights), each run
drops one ingredient and retrains to convergence, so you can see what each one
contributes:

  full     SGD + augmentation + class weights   (the proposed model)
  no_aug   the same, without augmentation
  no_cw    the same, without class weights
  adam     the same, but Adam instead of SGD

The five-conv / four-dense architecture, squared-hinge (SVM) loss, seed 42 and
the rest are held fixed. Each variant runs in its own process so the GPU memory
is freed between runs, and the run is resumable via a small ledger. Results go to
component_ablation_summary.csv and component_ablation_flat.csv.

Usage:
    DATA_DIR="path/to/dataset" python run_component_ablation.py
    CA_SMOKE=1 DATA_DIR="path/to/dataset" python run_component_ablation.py   # quick check
"""
import os
import csv
import sys
import json
import time
import subprocess
import traceback
import numpy as np
from pathlib import Path


# Point DATA_DIR at your dataset (with Training/Validation/Testing).
DATA_DIR   = Path(os.environ.get("DATA_DIR", "path/to/dataset"))
SMOKE      = os.environ.get("CA_SMOKE") == "1"
OUTPUT_DIR = Path("experiments_component_ablation_smoke" if SMOKE
                  else "experiments_component_ablation")
SEED       = 42
MAX_EPOCHS = 1 if SMOKE else 100             # trained to convergence
BATCH_SIZE = 36
IMG_HEIGHT, IMG_WIDTH = 360, 640
CONV_FILTERS = [32, 64, 128, 256, 512]       # 5 conv
DENSE_UNITS  = [1024, 128, 16]               # 3 hidden dense + linear SVM head = 4 dense
CLASS_NAMES  = ["Defect", "No_Defect"]       # Defect = class 0 = positive

# Each variant drops one ingredient from the final recipe (leave-one-out).
VARIANTS = [
    {"key": "full",   "col": "Full (proposed)",  "augment": True,  "class_weight": True,  "opt": "sgd"},
    {"key": "no_aug", "col": "No augmentation",  "augment": False, "class_weight": True,  "opt": "sgd"},
    {"key": "no_cw",  "col": "No class weights", "augment": True,  "class_weight": False, "opt": "sgd"},
    {"key": "adam",   "col": "Adam (not SGD)",   "augment": True,  "class_weight": True,  "opt": "adam"},
]
KEY2VAR = {variant["key"]: variant for variant in VARIANTS}

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
        raise SystemExit(f"ERROR: no training images under {DATA_DIR / 'Training'}. "
                         f"Set DATA_DIR to your leakage-free split path.")
    total = n_defect + n_no_defect
    weights = {0: total / (2.0 * n_defect), 1: total / (2.0 * n_no_defect)}
    print(f"Training set: {n_defect} defect, {n_no_defect} no-defect tiles "
          f"(class weights {weights[0]:.3f} / {weights[1]:.3f})")
    return weights


def build_cnnsvm(augment):
    """The proposed 5-conv / 4-dense CNN-SVM; `augment` toggles the augmentation stem."""
    model_layers = [layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)), layers.Resizing(180, 320)]
    if augment:
        model_layers += [layers.RandomRotation(0.1), layers.RandomBrightness(0.1)]
    for i in range(5):
        model_layers.append(layers.Conv2D(CONV_FILTERS[i], 3, padding="same",
                                          activation="relu", strides=2 if i == 0 else 1))
        if i % 2 == 0:
            model_layers.append(layers.MaxPool2D(2, 2))
    model_layers.append(layers.Flatten())
    for units in DENSE_UNITS:
        model_layers.append(layers.Dense(units, activation="relu",
                                        kernel_regularizer=regularizers.L2(0.01)))
    model_layers.append(layers.Dense(1, kernel_regularizer=regularizers.L2(0.01)))
    return keras.Sequential(model_layers, name="cnnsvm_5c_4d")


def make_optimizer(opt):
    if opt == "sgd":
        return optimizers.SGD(learning_rate=0.005, momentum=0.2, clipnorm=1.0)
    return optimizers.Adam()                             # default learning rate 0.001


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


def run_one(variant):
    init_tensorflow()
    run_id = variant["key"]
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    class_weights = compute_class_weights() if variant["class_weight"] else None

    train_data = load_split("Training", shuffle=True).prefetch(AUTOTUNE)
    val_data = load_split("Validation", shuffle=False).prefetch(AUTOTUNE)
    if SMOKE:
        train_data, val_data = train_data.take(6), val_data.take(2)

    model = build_cnnsvm(variant["augment"])
    model.compile(optimizer=make_optimizer(variant["opt"]),
                  loss=squared_hinge_loss, metrics=[svm_accuracy])

    callbacks = [keras.callbacks.BackupAndRestore(backup_dir=str(run_dir / "backup")),
                 keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
                 keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0005,
                                               patience=4, mode="min", verbose=1,
                                               restore_best_weights=True)]

    start = time.perf_counter()
    history = model.fit(train_data, validation_data=val_data, epochs=MAX_EPOCHS,
                        callbacks=callbacks, class_weight=class_weights, verbose=2)
    elapsed = time.perf_counter() - start

    hist = history.history
    epochs_run = len(hist.get("loss", []))
    val_loss = hist.get("val_loss", [])
    best_epoch = int(np.nanargmin(val_loss)) if val_loss and not all(np.isnan(val_loss)) else epochs_run - 1

    def value_at_best(key):
        values = hist.get(key, [])
        return round(float(values[best_epoch]), 4) if values and 0 <= best_epoch < len(values) else None

    train_acc, val_acc = value_at_best("svm_accuracy"), value_at_best("val_svm_accuracy")

    test_data = load_split("Testing", shuffle=False)
    if SMOKE:
        test_data = test_data.take(2)
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in test_data]).astype(int)
    scores = model.predict(test_data, verbose=0).ravel()
    cm = confusion_metrics(y_true, (scores > 0.0).astype(int))

    hours, minutes, seconds = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    result = dict(done=True, variant=run_id, col=variant["col"], augment=variant["augment"],
                  class_weight=variant["class_weight"], optimizer=variant["opt"],
                  epochs_run=epochs_run, train_time_hms=f"{hours}h {minutes}m {seconds}s",
                  train_time_sec=round(elapsed, 1), train_acc=train_acc, val_acc=val_acc,
                  test_acc=round(cm["accuracy"], 4), test_recall=round(cm["recall"], 4),
                  test_precision=round(cm["precision"], 4),
                  TP=cm["TP"], FN=cm["FN"], FP=cm["FP"], TN=cm["TN"])

    with open(run_dir / "test_predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["y_true", "y_score"])
        for label, score in zip(y_true, scores):
            writer.writerow([int(label), float(score)])
    json.dump(result, open(run_dir / "metrics.json", "w"), indent=2)
    print(f"  {run_id}: test accuracy {result['test_acc']}, recall {result['test_recall']} "
          f"(train {train_acc}, val {val_acc}), {epochs_run} epochs in {result['train_time_hms']}", flush=True)
    return result


def run_single(key):
    ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
    if ledger.get(key, {}).get("done"):
        print(f"Skipping {key} (already done).")
        return
    try:
        ledger[key] = run_one(KEY2VAR[key])
    except Exception as error:
        print(f"{key} failed: {error}")
        traceback.print_exc()
        ledger[key] = {"done": False, "variant": key, "error": str(error)[:300]}
    json.dump(ledger, open(LEDGER_FILE, "w"), indent=2)


def write_summary(ledger):
    header = ["metric"] + [variant["col"] for variant in VARIANTS]
    rows = [header]
    specs = [("test_acc", "Test accuracy", True), ("test_recall", "Recall", False),
             ("test_precision", "Precision", False), ("train_acc", "Train accuracy", True)]
    for metric, label, as_pct in specs:
        row = [label]
        for variant in VARIANTS:
            record = ledger.get(variant["key"], {})
            val = record.get(metric)
            if isinstance(val, (int, float)):
                row.append(f"{val * 100:.1f}%" if as_pct else f"{val:.3f}")
            else:
                row.append("FAIL" if record else "-")
        rows.append(row)
    with open(OUTPUT_DIR / "component_ablation_summary.csv", "w", newline="") as file:
        csv.writer(file).writerows(rows)

    flat = [["variant", "col", "augment", "class_weight", "optimizer", "epochs_run",
             "train_time_hms", "train_acc", "val_acc", "test_acc", "test_recall",
             "test_precision", "TP", "FN", "FP", "TN", "done"]]
    for variant in VARIANTS:
        record = ledger.get(variant["key"], {})
        flat.append([variant["key"], variant["col"], variant["augment"],
                     variant["class_weight"], variant["opt"], record.get("epochs_run"),
                     record.get("train_time_hms"), record.get("train_acc"),
                     record.get("val_acc"), record.get("test_acc"), record.get("test_recall"),
                     record.get("test_precision"), record.get("TP"), record.get("FN"),
                     record.get("FP"), record.get("TN"), record.get("done", False)])
    with open(OUTPUT_DIR / "component_ablation_flat.csv", "w", newline="") as file:
        csv.writer(file).writerows(flat)
    print("\nWrote component_ablation_summary.csv and component_ablation_flat.csv.", flush=True)


def driver():
    print(f"Running {len(VARIANTS)} variants, one process each "
          f"(smoke test: {SMOKE}, output folder: {OUTPUT_DIR}).", flush=True)
    for variant in VARIANTS:
        key = variant["key"]
        ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
        if ledger.get(key, {}).get("done"):
            print(f"Skipping {key} (already done).", flush=True)
            continue
        print(f"\n----- {key} -----", flush=True)
        child = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--one", key])
        if child.returncode != 0:
            ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
            if not ledger.get(key, {}).get("done"):
                ledger[key] = {"done": False, "variant": key, "error": f"child rc={child.returncode}"}
                json.dump(ledger, open(LEDGER_FILE, "w"), indent=2)
            print(f"{key}: process exited with code {child.returncode}", flush=True)

    ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
    write_summary(ledger)
    print("\nResults (test accuracy / recall) for each variant:", flush=True)
    for variant in VARIANTS:
        record = ledger.get(variant["key"], {})
        print(f"  {variant['col']:18s}: test accuracy {record.get('test_acc', 'FAIL')}, "
              f"recall {record.get('test_recall', '-')}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    if "--one" in sys.argv:
        run_single(sys.argv[sys.argv.index("--one") + 1])
    elif "--summary" in sys.argv:
        write_summary(json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {})
    else:
        driver()
