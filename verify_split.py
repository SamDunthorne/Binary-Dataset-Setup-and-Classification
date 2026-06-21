"""Check that the split is leakage-free and matches its manifest (read-only)."""
import csv
import collections
from pathlib import Path

DST = Path("path/to/split")    # the split to verify (must contain manifest.csv)
manifest_path = DST / "manifest.csv"

rows = list(csv.DictReader(open(manifest_path, newline="")))
print(f"Manifest rows (kept images): {len(rows)}")

# 1) Leakage check: each photo_id should appear in exactly one split.
splits_by_photo = collections.defaultdict(set)
for row in rows:
    splits_by_photo[row["photo_id"]].add(row["split"])
leaked_photos = {photo: splits for photo, splits in splits_by_photo.items() if len(splits) > 1}
print(f"Distinct photos: {len(splits_by_photo)}")
print(f"Photos appearing in more than one split: {len(leaked_photos)} (should be 0)")
for photo, splits in list(leaked_photos.items())[:10]:
    print("   leaked:", photo, sorted(splits))

# 2) Same check at the filename level (no identical tile in two splits).
splits_by_filename = collections.defaultdict(set)
for row in rows:
    splits_by_filename[row["filename"]].add(row["split"])
duplicate_filenames = {name: splits for name, splits in splits_by_filename.items() if len(splits) > 1}
print(f"Filenames appearing in more than one split: {len(duplicate_filenames)} (should be 0)")

# 3) How many photos ended up in each split.
photos_per_split = collections.defaultdict(set)
for row in rows:
    photos_per_split[row["split"]].add(row["photo_id"])
for split_name in ["Training", "Validation", "Testing"]:
    print(f"Photos in {split_name:11s}: {len(photos_per_split[split_name])}")

# 4) Augmented images should only be in Training, never in Validation/Testing.
aug_in_val_test = [row for row in rows
                   if row["split"] in ("Validation", "Testing") and row["is_aug"] == "1"]
print(f"Augmented images in Validation/Testing: {len(aug_in_val_test)} (should be 0)")

# 5) Compare the PNG counts on disk against the manifest (catches a partial copy).
manifest_counts = collections.Counter((row["split"], row["label"]) for row in rows)
all_match = True
for split_name in ["Training", "Validation", "Testing"]:
    for label in ["Defect", "No_Defect"]:
        folder = DST / split_name / label
        disk_count = sum(1 for _ in folder.glob("*.png")) if folder.is_dir() else 0
        manifest_count = manifest_counts[(split_name, label)]
        matches = (disk_count == manifest_count)
        all_match &= matches
        print(f"  {split_name:11s} / {label:9s}: {disk_count:6d} on disk, "
              f"{manifest_count:6d} in manifest  {'OK' if matches else 'MISMATCH'}")
print(f"\nAll disk counts match the manifest: {all_match}")
print(f"Split is leakage-free: {len(leaked_photos) == 0 and len(duplicate_filenames) == 0}")
