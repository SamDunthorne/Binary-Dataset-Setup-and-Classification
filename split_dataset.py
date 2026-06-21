"""
Split the roof-defect dataset at the PHOTO level (leakage-free).
Groups every tile/augmentation by its 8-digit source-photo ID and assigns WHOLE
photos to train/val/test (75:15:10, seed 42), augmentations in Training only,
Validation/Testing originals only. Resumable + retry-resilient. Writes manifest.csv.
"""
import csv
import random
import shutil
import time
from pathlib import Path
from collections import Counter

# Configuration
SRC = Path("path/to/source_dataset")   # source: Training/ + Testing/, each with Defect/ and No_Defect/
DST = Path("path/to/output_split")     # where the leakage-free split is written
SEED = 42
TRAIN, VAL, TEST = 0.75, 0.15, 0.10
AUG_SPLITS = {"Training"}              # augmented images are only kept in these splits
LABELS = ["Defect", "No_Defect"]
SRC_SPLITS = ["Training", "Testing"]

def photo_id(stem):  return stem.split("_")[0]   # "00000123_05_aug" -> "00000123"
def is_aug(stem):    return stem.endswith("_aug")

# Collect every file in the source dataset, tagged with its photo id and label.
records = []
for source_split in SRC_SPLITS:
    for label in LABELS:
        folder = SRC / source_split / label
        if not folder.is_dir():
            continue
        for file_path in folder.glob("*.png"):
            records.append((file_path, photo_id(file_path.stem), label, is_aug(file_path.stem)))
print(f"Found {len(records)} files in the source dataset.")

# Assign each WHOLE photo to one split, so no photo is shared across train/val/test.
photos = sorted({record[1] for record in records})
random.seed(SEED)
random.shuffle(photos)
n_photos = len(photos)
n_train = int(TRAIN * n_photos)
n_val = int(VAL * n_photos)
split_of = {}
for i, pid in enumerate(photos):
    if i < n_train:
        split_of[pid] = "Training"
    elif i < n_train + n_val:
        split_of[pid] = "Validation"
    else:
        split_of[pid] = "Testing"

# Copy each file into its split folder and record it in the manifest.
DST.mkdir(parents=True, exist_ok=True)
manifest, counts = [], Counter()
for index, (file_path, pid, label, is_augmented) in enumerate(records):
    split_name = split_of[pid]
    if is_augmented and split_name not in AUG_SPLITS:   # drop augmented images from Val/Test
        continue
    out_dir = DST / split_name / label
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_path = out_dir / file_path.name
    # skip the copy if an identical-size file is already there (makes reruns cheap)
    if not (dest_path.exists() and dest_path.stat().st_size == file_path.stat().st_size):
        for attempt in range(6):                        # retry: the network drive can briefly drop out
            try:
                shutil.copy2(file_path, dest_path)
                break
            except OSError as error:
                if attempt == 5:
                    raise
                print(f"  retry {attempt + 1}/5 on {file_path.name}: {error}")
                time.sleep(3)
    manifest.append((split_name, label, pid, file_path.name, int(is_augmented)))
    counts[(split_name, label)] += 1
    if index % 5000 == 0:
        print(f"  copied {index}/{len(records)} ...")

with open(DST / "manifest.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["split", "label", "photo_id", "filename", "is_aug"])
    writer.writerows(manifest)

# Summary: photos per split, then images per split after the augmentation rule.
photo_counts = Counter(split_of.values())
print("\nPhotos per split:")
for split_name in ["Training", "Validation", "Testing"]:
    print(f"  {split_name:11s}: {photo_counts[split_name]:5d} photos")
print("\nImages per split (after the augmentation rule):")
total = 0
for split_name in ["Training", "Validation", "Testing"]:
    n_defect = counts[(split_name, "Defect")]
    n_no_defect = counts[(split_name, "No_Defect")]
    total += n_defect + n_no_defect
    print(f"  {split_name:11s}: {n_defect + n_no_defect:6d}   (Defect {n_defect}, No_Defect {n_no_defect})")
print(f"  {'TOTAL':11s}: {total:6d}")
print(f"\nNew dataset : {DST}\nManifest    : {DST / 'manifest.csv'}")
