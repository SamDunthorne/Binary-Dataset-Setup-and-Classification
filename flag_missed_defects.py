"""
Sort a run's test-set errors into folders so they're easy to browse:

  missed_defects/   defects the model predicted as non-defect (false negatives)
  false_alarms/     non-defects predicted as defects        (false positives)

plus a CSV for each, sorted by the model's confidence.

It reuses the exported test_predictions.csv and lines each score up with its
image file using the same (shuffle=False) order Keras loads them in, so no
retraining or GPU is needed. A per-image label check aborts if the predictions
don't line up with the folders, so you never get mismatched filenames.

Defect = class 0 = positive; a score > 0 means the tile was predicted No_Defect.

Usage:
    DATA_DIR="path/to/dataset" PRED="path/to/test_predictions.csv" python flag_missed_defects.py
"""
import os
import csv
import shutil
from pathlib import Path

import keras

DATA = Path(os.environ.get("DATA_DIR", "path/to/dataset"))           # split with Training/ Validation/ Testing/
PRED = Path(os.environ.get("PRED", "path/to/test_predictions.csv"))  # a run's exported predictions
OUT  = Path(os.environ.get("OUT", "error_analysis"))
IMG_H, IMG_W = 360, 640
CLASS_NAMES = ["Defect", "No_Defect"]            # Defect = class 0 = positive

# 1) Load the test images in the exact Keras order (shuffle=False). This only
#    reads filenames -- no model and no GPU are needed.
test_dataset = keras.utils.image_dataset_from_directory(
    DATA / "Testing", image_size=(IMG_H, IMG_W), shuffle=False,
    color_mode="rgb", batch_size=36, label_mode="binary",
    class_names=CLASS_NAMES, verbose=False)
image_paths = list(test_dataset.file_paths)

# 2) Read the exported predictions (same order as the images above).
if not PRED.is_file():
    raise SystemExit(f"ERROR: predictions file not found: {PRED} (set PRED=...).")
rows = list(csv.reader(open(PRED)))[1:]           # skip header
y_true = [int(float(row[0])) for row in rows]
y_score = [float(row[1]) for row in rows]
if len(image_paths) != len(y_true):
    raise SystemExit(f"ERROR: {len(image_paths)} test images but {len(y_true)} predictions -- "
                     f"is PRED from THIS Testing folder?")

# 3) Classify each tile and collect the two kinds of error.
missed, alarms = [], []                           # false negatives, false positives
for image_path, true_label, score in zip(image_paths, y_true, y_score):
    folder_label = 0 if Path(image_path).parent.name == "Defect" else 1
    if folder_label != true_label:
        raise SystemExit(f"ERROR: order mismatch at {image_path} "
                         f"(folder={folder_label}, csv={true_label}). Aborting.")
    predicted_label = 1 if score > 0.0 else 0     # score > 0 -> predicted No_Defect
    if true_label == 0 and predicted_label == 1:
        missed.append((image_path, score))        # missed defect (false negative)
    elif true_label == 1 and predicted_label == 0:
        alarms.append((image_path, score))        # false alarm (false positive)

# 4) Copy the error images into a folder and write a sorted list for each.
def save_bucket(items, subfolder, most_confident_first):
    out_dir = OUT / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(items, key=lambda item: item[1], reverse=most_confident_first)   # by score
    with open(OUT / f"{subfolder}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "score"])
        for image_path, score in items:
            shutil.copy2(image_path, out_dir / Path(image_path).name)
            writer.writerow([Path(image_path).name, round(score, 4)])
    print(f"  {subfolder}: {len(items)} tiles -> {out_dir}")

OUT.mkdir(parents=True, exist_ok=True)
print(f"missed defects (false negatives): {len(missed)}")
print(f"false alarms   (false positives): {len(alarms)}")
save_bucket(missed, "missed_defects", most_confident_first=True)    # most confident misses first
save_bucket(alarms, "false_alarms", most_confident_first=False)     # most confident false alarms first
print(f"\nDone. Browse {OUT / 'missed_defects'} (and {OUT / 'false_alarms'}).")
