"""Integrity + leakage verification for the local resplit. Read-only."""
import csv, collections
from pathlib import Path

DST = Path("Binary_Blended - Resplit")    # the split to verify (relative to CWD)
man = DST / "manifest.csv"

rows = list(csv.DictReader(open(man, newline="")))
print(f"manifest rows (kept images): {len(rows)}")

# 1) LEAKAGE: each photo_id must live in exactly one split
pid_splits = collections.defaultdict(set)
for r in rows:
    pid_splits[r["photo_id"]].add(r["split"])
leaked = {p: s for p, s in pid_splits.items() if len(s) > 1}
print(f"distinct photo_ids: {len(pid_splits)}")
print(f"LEAKED photo_ids (in >1 split): {len(leaked)}  <-- must be 0")
for p, s in list(leaked.items())[:10]:
    print("   LEAK", p, sorted(s))

# 2) filename-level cross-split check (no identical tile in two splits)
fn_splits = collections.defaultdict(set)
for r in rows:
    fn_splits[r["filename"]].add(r["split"])
dupfn = {f: s for f, s in fn_splits.items() if len(s) > 1}
print(f"filenames in >1 split: {len(dupfn)}  <-- must be 0")

# 3) photos per split
pps = collections.defaultdict(set)
for r in rows:
    pps[r["split"]].add(r["photo_id"])
for sp in ["Training", "Validation", "Testing"]:
    print(f"photos {sp:11s}: {len(pps[sp])}")

# 4) no augmented images in Validation/Testing
augbad = [r for r in rows if r["split"] in ("Validation", "Testing") and r["is_aug"] == "1"]
print(f"augmented imgs in Val/Test: {len(augbad)}  <-- must be 0")

# 5) on-disk PNG counts vs manifest counts (detect partial copy)
img = collections.Counter((r["split"], r["label"]) for r in rows)
allok = True
for sp in ["Training", "Validation", "Testing"]:
    for lb in ["Defect", "No_Defect"]:
        p = DST / sp / lb
        n = sum(1 for _ in p.glob("*.png")) if p.is_dir() else 0
        m = img[(sp, lb)]
        ok = (n == m)
        allok &= ok
        print(f"disk {sp:11s}/{lb:9s}: disk={n:6d} manifest={m:6d} {'OK' if ok else 'MISMATCH'}")
print(f"\nALL DISK COUNTS MATCH MANIFEST: {allok}")
print(f"LEAKAGE-FREE: {len(leaked) == 0 and len(dupfn) == 0}")
