"""
Train and evaluate the three roof-defect models (CNN-SVM, AlexNet, GoogLeNet) on
the leakage-free split, with repeated seeds for confidence intervals.

The run is resumable: if it stops part way, just start it again and it carries on
(a ledger skips finished runs and per-epoch state is saved to disk). Each run also
exports its test predictions, used later for the ROC/PR and confusion-matrix
figures. Needs TensorFlow with GPU support (on Windows, run it under WSL2).

Usage:
    # set SMOKE_TEST = True below and run once to check the GPU + pipeline,
    # then set it False and run again for the full set:
    python run_experiments.py
"""

import gc
import csv
import json
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers, losses, metrics, optimizers, regularizers


# Path to the split dataset (with Training/Validation/Testing subfolders).
DATA_DIR   = Path("path/to/dataset")
OUTPUT_DIR = Path("experiments")

SMOKE_TEST = True               # True = quick check (1 epoch, 1 model); set False for the real run
SEEDS      = [42, 43, 44]       # one run per seed, so we can report a confidence interval
MAX_EPOCHS = 200                # early stopping usually halts well before this
BATCH_SIZE = 36
IMG_HEIGHT, IMG_WIDTH = 360, 640
USE_CLASS_WEIGHTS = True        # weight the loss to offset the class imbalance (same for all models)
MODELS      = ["cnnsvm", "alexnet", "googlenet"]
CLASS_NAMES = ["Defect", "No_Defect"]   # "Defect" is class 0 and the positive class

if SMOKE_TEST:
    SEEDS, MAX_EPOCHS, MODELS = [42], 1, ["cnnsvm"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_FILE = OUTPUT_DIR / "ledger.json"


# --- GPU check ---
available_gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow {tf.__version__}")
print(f"GPUs detected: {available_gpus if available_gpus else 'none -- training will be slow on CPU'}")
if not available_gpus:
    print("No GPU detected. On Windows, run under WSL2 with")
    print("'pip install tensorflow[and-cuda]' (see requirements.txt).")
for gpu in available_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as error:
        print("Could not set memory growth:", error)

AUTOTUNE = tf.data.AUTOTUNE


# --- data loading ---
def load_split(split_name, shuffle):
    """Load one split (Training / Validation / Testing) as a batched image dataset."""
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / split_name, image_size=(IMG_HEIGHT, IMG_WIDTH), shuffle=shuffle,
        seed=42, color_mode="rgb", batch_size=BATCH_SIZE, label_mode="binary",
        class_names=CLASS_NAMES, verbose=False)


def squared_hinge_loss(y_true, y_pred):
    # Squared-hinge (L2-SVM) loss. Labels stay as {0, 1} on the outside and are
    # mapped to {-1, +1} here, so Keras class weights (keyed 0/1) still apply.
    signed_labels = 2.0 * y_true - 1.0
    return tf.reduce_mean(tf.square(tf.maximum(0.0, 1.0 - signed_labels * y_pred)), axis=-1)


def compute_class_weights():
    """Inverse-frequency weights so the smaller class is not under-counted in the loss."""
    n_defect    = len(list((DATA_DIR / "Training" / "Defect").glob("*.png")))
    n_no_defect = len(list((DATA_DIR / "Training" / "No_Defect").glob("*.png")))
    total = n_defect + n_no_defect
    if n_defect == 0 or n_no_defect == 0:
        raise SystemExit(
            f"ERROR: no training images found under {DATA_DIR / 'Training'}.\n"
            f"Set DATA_DIR (top of file) to your dataset path.")
    weights = {0: total / (2.0 * n_defect), 1: total / (2.0 * n_no_defect)}
    print(f"Training set: {n_defect} defect, {n_no_defect} no-defect tiles "
          f"(class weights {weights[0]:.3f} / {weights[1]:.3f})")
    return weights


CLASS_WEIGHTS = compute_class_weights() if USE_CLASS_WEIGHTS else None


# --- model architectures ---
def build_cnnsvm():
    # Proposed model: a resize + augmentation stem, five convolution blocks, three
    # dense layers, and a single linear output (the SVM decision score).
    return keras.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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
        layers.Dense(1, kernel_regularizer=regularizers.L2(0.01)),   # linear SVM head
    ], name="cnnsvm")


def build_alexnet():
    from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                              BatchNormalization)
    return keras.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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


def inception_block(x, filters_1x1, filters_3x3_reduce, filters_3x3,
                    filters_5x5_reduce, filters_5x5, filters_pool):
    # One GoogLeNet inception block: four parallel branches joined back together.
    from keras.layers import Conv2D, MaxPooling2D, Concatenate
    branch_1x1 = Conv2D(filters_1x1, 1, padding="same", activation="relu")(x)
    branch_3x3 = Conv2D(filters_3x3_reduce, 1, padding="same", activation="relu")(x)
    branch_3x3 = Conv2D(filters_3x3, 3, padding="same", activation="relu")(branch_3x3)
    branch_5x5 = Conv2D(filters_5x5_reduce, 1, padding="same", activation="relu")(x)
    branch_5x5 = Conv2D(filters_5x5, 5, padding="same", activation="relu")(branch_5x5)
    branch_pool = MaxPooling2D(3, strides=1, padding="same")(x)
    branch_pool = Conv2D(filters_pool, 1, padding="same", activation="relu")(branch_pool)
    return Concatenate(axis=-1)([branch_1x1, branch_3x3, branch_5x5, branch_pool])


def build_googlenet():
    from keras.layers import (Input, Conv2D, MaxPooling2D, AveragePooling2D,
                              Dropout, Flatten, Dense)
    from keras.models import Model
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(64, 7, strides=2, padding="same", activation="relu")(inputs)
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = Conv2D(64, 1, padding="same", activation="relu")(x)
    x = Conv2D(192, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = inception_block(x, 64, 96, 128, 16, 32, 32)     # block 3a
    x = inception_block(x, 128, 128, 192, 32, 96, 64)   # block 3b
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = AveragePooling2D(7, padding="valid")(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="googlenet")


def svm_accuracy(y_true, y_pred):
    # Accuracy for the SVM head: a raw score above 0 means class 1 (No_Defect).
    predicted = tf.cast(y_pred > 0.0, y_true.dtype)
    return tf.reduce_mean(tf.cast(tf.equal(predicted, y_true), tf.float32))


# Per-model settings. The CNN-SVM uses the squared-hinge loss; the two baselines
# use binary cross-entropy. Each keeps its own learning rate and early-stopping
# sensitivity (min_delta), chosen during development.
MODEL_CONFIG = {
    "cnnsvm":    dict(build=build_cnnsvm, is_svm=True,  learning_rate=0.005, momentum=0.2,
                      clipnorm=1.0, min_delta=0.0005,
                      loss=squared_hinge_loss, metric=svm_accuracy),
    "alexnet":   dict(build=build_alexnet, is_svm=False, learning_rate=0.002, momentum=0.2,
                      clipnorm=None, min_delta=0.001,
                      loss=losses.BinaryCrossentropy(from_logits=False),
                      metric=metrics.BinaryAccuracy(name="binary_accuracy")),
    "googlenet": dict(build=build_googlenet, is_svm=False, learning_rate=0.002, momentum=0.2,
                      clipnorm=None, min_delta=0.002,
                      loss=losses.BinaryCrossentropy(from_logits=False),
                      metric=metrics.BinaryAccuracy(name="binary_accuracy")),
}


# --- test-set metrics (Defect = the positive class) ---
def confusion_metrics(true_labels, predicted_labels):
    # Labels are {0, 1} with class 0 = Defect = positive.
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    true_positives  = int(np.sum((true_labels == 0) & (predicted_labels == 0)))
    false_negatives = int(np.sum((true_labels == 0) & (predicted_labels == 1)))
    false_positives = int(np.sum((true_labels == 1) & (predicted_labels == 0)))
    true_negatives  = int(np.sum((true_labels == 1) & (predicted_labels == 1)))
    total = true_positives + false_negatives + false_positives + true_negatives

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return dict(
        TP=true_positives, FN=false_negatives, FP=false_positives, TN=true_negatives, n=total,
        accuracy=ratio(true_positives + true_negatives, total),
        recall=ratio(true_positives, true_positives + false_negatives),
        precision=ratio(true_positives, true_positives + false_positives),
        FPR=ratio(false_positives, false_positives + true_negatives),
        FNR=ratio(false_negatives, true_positives + false_negatives),
        TNR=ratio(true_negatives, false_positives + true_negatives))


# --- train and evaluate one (model, seed) ---
def run_one(model_name, seed):
    run_id = f"{model_name}_seed{seed}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    keras.backend.clear_session()   # release the previous run's GPU memory
    gc.collect()
    keras.utils.set_random_seed(seed)
    config = MODEL_CONFIG[model_name]

    train_data = load_split("Training", shuffle=True).prefetch(AUTOTUNE)
    val_data   = load_split("Validation", shuffle=False).prefetch(AUTOTUNE)
    if SMOKE_TEST:                  # smoke test: a few batches just to confirm it runs
        train_data, val_data = train_data.take(6), val_data.take(2)

    optimizer_kwargs = {}
    if config["clipnorm"]:
        optimizer_kwargs["clipnorm"] = config["clipnorm"]
    model = config["build"]()
    model.compile(
        optimizer=optimizers.SGD(learning_rate=config["learning_rate"],
                                 momentum=config["momentum"], **optimizer_kwargs),
        loss=config["loss"], metrics=[config["metric"]])

    callbacks = [
        # save progress so an interrupted run can resume from where it stopped
        keras.callbacks.BackupAndRestore(backup_dir=str(run_dir / "backup")),
        # stop once the validation loss stops improving, and keep the best weights
        keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=config["min_delta"],
                                      patience=4, mode="min", verbose=1,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(run_dir / "best.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
    ]
    model.fit(train_data, validation_data=val_data, epochs=MAX_EPOCHS,
              callbacks=callbacks, class_weight=CLASS_WEIGHTS, verbose=2)
    model.save(run_dir / "final.keras")

    # Evaluate on the held-out test set in a single pass. We do not shuffle, so
    # the saved predictions stay in the same order as the image files.
    test_data = load_split("Testing", shuffle=False)
    true_labels = np.concatenate([labels.numpy().ravel() for _, labels in test_data]).astype(int)
    scores = model.predict(test_data, verbose=0).ravel()
    threshold = 0.0 if config["is_svm"] else 0.5
    predicted_labels = (scores > threshold).astype(int)   # 1 = No_Defect
    result = confusion_metrics(true_labels, predicted_labels)

    # Save the raw scores so the ROC/PR and confusion-matrix scripts can reuse them.
    with open(run_dir / "test_predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["y_true", "y_score"])    # y_true: 0 = Defect; y_score: raw model output
        for label, score in zip(true_labels, scores):
            writer.writerow([int(label), float(score)])
    json.dump(result, open(run_dir / "metrics.json", "w"), indent=2)
    print(f"  {run_id}: test accuracy {result['accuracy']:.4f}, "
          f"recall {result['recall']:.4f}, precision {result['precision']:.4f}")
    return result


# --- average over seeds (mean and 95% confidence interval) ---
# Student's t critical values (two-sided, 95%) indexed by degrees of freedom.
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_and_ci(values):
    """Return (mean, half-width of the 95% confidence interval) for a list of numbers."""
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    std = (sum((value - mean) ** 2 for value in values) / (count - 1)) ** 0.5
    half_width = T_CRITICAL_95.get(count - 1, 1.96) * std / (count ** 0.5)
    return mean, half_width


def write_summary(ledger):
    """Average accuracy/recall/precision over each model's seeds -> results_summary.csv."""
    from collections import defaultdict
    runs_by_model = defaultdict(list)
    for run_id, record in ledger.items():
        if record.get("done"):
            model_name = run_id.rsplit("_seed", 1)[0]
            runs_by_model[model_name].append(record)

    rows = [("model", "runs", "acc_mean", "acc_ci95", "rec_mean", "rec_ci95",
             "prec_mean", "prec_ci95")]
    for model_name, runs in sorted(runs_by_model.items()):
        acc_mean,  acc_ci  = mean_and_ci([r["accuracy"]  for r in runs])
        rec_mean,  rec_ci  = mean_and_ci([r["recall"]    for r in runs])
        prec_mean, prec_ci = mean_and_ci([r["precision"] for r in runs])
        rows.append((model_name, len(runs), f"{acc_mean:.4f}", f"{acc_ci:.4f}",
                     f"{rec_mean:.4f}", f"{rec_ci:.4f}", f"{prec_mean:.4f}", f"{prec_ci:.4f}"))
        print(f"  {model_name:10s}  {len(runs)} run(s)  "
              f"accuracy={acc_mean:.4f}+/-{acc_ci:.4f}  "
              f"recall={rec_mean:.4f}+/-{rec_ci:.4f}  "
              f"precision={prec_mean:.4f}+/-{prec_ci:.4f}")
    with open(OUTPUT_DIR / "results_summary.csv", "w", newline="") as file:
        csv.writer(file).writerows(rows)


# --- main loop ---
def main():
    ledger = json.load(open(LEDGER_FILE)) if LEDGER_FILE.exists() else {}
    for model_name in MODELS:
        for seed in SEEDS:
            run_id = f"{model_name}_seed{seed}"
            if ledger.get(run_id, {}).get("done"):
                print(f"Skipping {run_id} (already finished).")
                continue
            print(f"\n----- Training {run_id} -----")
            try:
                result = run_one(model_name, seed)
                ledger[run_id] = {"done": True, **result}
            except Exception as error:
                print(f"Run {run_id} failed: {error}")
                traceback.print_exc()
                ledger[run_id] = {"done": False, "error": str(error)}
            # save the ledger after every run, so progress is never lost
            json.dump(ledger, open(LEDGER_FILE, "w"), indent=2)

    print("\nSummary (mean +/- 95% confidence interval):")
    write_summary(ledger)
    print(f"\nFinished. Results are in the '{OUTPUT_DIR}' folder: ledger.json,")
    print("results_summary.csv, and per-run history.csv / metrics.json / test_predictions.csv.")


if __name__ == "__main__":
    main()
