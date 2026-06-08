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

# ---------------------------- configuration ----------------------------
SRC = Path("Binary_Blended - Final")      # source: Training/ + Testing/, each with Defect/No_Defect
DST = Path("Binary_Blended - Split")    # output split (use a local disk for speed)
SEED = 42
TRAIN, VAL, TEST = 0.75, 0.15, 0.10
AUG_SPLITS = {"Training"}
LABELS = ["Defect", "No_Defect"]
SRC_SPLITS = ["Training", "Testing"]
# -----------------------------------------------------------------------

def photo_id(stem): return stem.split("_")[0]
def is_aug(stem):  return stem.endswith("_aug")

records = []
for sp in SRC_SPLITS:
    for label in LABELS:
        folder = SRC / sp / label
        if not folder.is_dir():
            continue
        for f in folder.glob("*.png"):
            records.append((f, photo_id(f.stem), label, is_aug(f.stem)))
print(f"found {len(records)} files in the source dataset")

photos = sorted({r[1] for r in records})
random.seed(SEED)
random.shuffle(photos)
n = len(photos); n_tr = int(TRAIN * n); n_va = int(VAL * n)
split_of = {}
for i, pid in enumerate(photos):
    split_of[pid] = "Training" if i < n_tr else "Validation" if i < n_tr + n_va else "Testing"

DST.mkdir(parents=True, exist_ok=True)
manifest, counts = [], Counter()
for k, (path, pid, label, aug) in enumerate(records):
    sp = split_of[pid]
    if aug and sp not in AUG_SPLITS:
        continue
    out_dir = DST / sp / label
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / path.name
    if not (dest.exists() and dest.stat().st_size == path.stat().st_size):
        for attempt in range(6):
            try:
                shutil.copy2(path, dest); break
            except OSError as e:
                if attempt == 5: raise
                print(f"  retry {attempt + 1}/5 on {path.name}: {e}")
                time.sleep(3)
    manifest.append((sp, label, pid, path.name, int(aug)))
    counts[(sp, label)] += 1
    if k % 5000 == 0:
        print(f"  copied {k}/{len(records)} ...")

with open(DST / "manifest.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["split", "label", "photo_id", "filename", "is_aug"])
    w.writerows(manifest)

photo_counts = Counter(split_of.values())
print("\n=== PHOTOS per split ===")
for s in ["Training", "Validation", "Testing"]:
    print(f"  {s:11s}: {photo_counts[s]:5d} photos")
print("=== IMAGES per split (after the augmentation rule) ===")
total = 0
for s in ["Training", "Validation", "Testing"]:
    d, nd = counts[(s, "Defect")], counts[(s, "No_Defect")]
    total += d + nd
    print(f"  {s:11s}: {d + nd:6d}   (Defect {d}, No_Defect {nd})")
print(f"  {'TOTAL':11s}: {total:6d}")
print(f"\nnew dataset : {DST}\nmanifest    : {DST / 'manifest.csv'}")
