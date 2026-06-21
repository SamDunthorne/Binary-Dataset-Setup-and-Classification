"""
Measure inference latency vs. batch size for the three models (CNN-SVM, AlexNet,
GoogLeNet) and write the results to timing.csv.

Latency doesn't depend on the trained weights, so this just pushes random inputs
through the model architectures. No dataset needed, only a GPU. Each batch size
is timed over several forward passes after a warm-up, so the one-time first-call
compilation cost is excluded from the reported numbers.

Usage:
    python run_timing.py
    TIMING_SMOKE=1 python run_timing.py        # fewer sizes / reps, for a quick check
"""
import os
import gc
import csv
import time
import tensorflow as tf
import keras
from keras import layers, regularizers
from keras.models import Model

IMG_HEIGHT, IMG_WIDTH = 360, 640
SMOKE = os.environ.get("TIMING_SMOKE") == "1"
BATCH_SIZES = [1, 12, 24] if SMOKE else [1, 12, 24, 36, 72, 144, 288, 576]
REPS   = 5 if SMOKE else 30          # timed forward passes per batch size
WARMUP = 3 if SMOKE else 8           # untimed warm-up passes before each measurement
OUT    = "timing.csv"

# ------------------------------ GPU setup -------------------------------
available_gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow {tf.__version__}, GPUs: {available_gpus if available_gpus else 'none (CPU -- timings not representative)'}")
for gpu in available_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as error:
        print("Could not set memory growth:", error)


# ----- model builders (same architectures as in run_experiments.py) -----
def build_cnnsvm():
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
        layers.Dense(1, kernel_regularizer=regularizers.L2(0.01)),
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
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Conv2D(64, 7, strides=2, padding="same", activation="relu")(inputs)
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = Conv2D(64, 1, padding="same", activation="relu")(x)
    x = Conv2D(192, 3, padding="same", activation="relu")(x)
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = inception_block(x, 64, 96, 128, 16, 32, 32)
    x = inception_block(x, 128, 128, 192, 32, 96, 64)
    x = MaxPooling2D(3, strides=2, padding="same")(x)
    x = AveragePooling2D(7, padding="valid")(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="googlenet")


MODEL_BUILDERS = [("cnnsvm", build_cnnsvm), ("alexnet", build_alexnet),
                  ("googlenet", build_googlenet)]


def time_model(name, build):
    model = build()

    @tf.function
    def forward_pass(batch):
        return model(batch, training=False)

    latency_by_batch = {}
    for batch_size in BATCH_SIZES:
        random_batch = tf.random.normal((batch_size, IMG_HEIGHT, IMG_WIDTH, 3))
        try:
            for _ in range(WARMUP):                  # warm-up: skip the one-time graph build / JIT cost
                forward_pass(random_batch)
            forward_pass(random_batch).numpy()       # make sure the warm-up has finished
            start = time.perf_counter()
            for _ in range(REPS):
                result = forward_pass(random_batch)
            result.numpy()                           # wait for all timed passes to finish on the GPU
            latency_ms = (time.perf_counter() - start) / REPS * 1000.0
            latency_by_batch[batch_size] = round(latency_ms, 1)
            print(f"  batch {batch_size:4d}: {latency_ms:8.1f} ms", flush=True)
        except Exception as error:                   # e.g. out of memory at the largest batch size
            latency_by_batch[batch_size] = ""
            print(f"  batch {batch_size:4d}: failed ({str(error)[:90]})", flush=True)
    return latency_by_batch


def main():
    results = {}
    for name, build in MODEL_BUILDERS:
        print(f"\n{name}:", flush=True)
        results[name] = time_model(name, build)
        keras.backend.clear_session()
        gc.collect()

    with open(OUT, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["batch_size", "cnnsvm_ms", "alexnet_ms", "googlenet_ms"])
        for batch_size in BATCH_SIZES:
            writer.writerow([batch_size, results["cnnsvm"].get(batch_size, ""),
                             results["alexnet"].get(batch_size, ""),
                             results["googlenet"].get(batch_size, "")])
    print(f"\nWrote {OUT}")
    print("Each value is milliseconds per batch, averaged over the timed passes.")


if __name__ == "__main__":
    main()
