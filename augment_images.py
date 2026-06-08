"""
augment_images.py -- create one augmented copy of each tile.

Reproduces the dataset's augmentation step: a random horizontal flip, a random
rotation (+/- 30 deg), a random brightness shift (+/- 50), and Gaussian noise are
applied to each input tile, and the result is saved as <name>_aug.png.
"""
import random
from pathlib import Path

import cv2
import numpy as np

# ----------------------------- configuration -----------------------------
SRC  = Path(r"path/to/tiles")          # tiles to augment (e.g. a class folder)
DST  = Path(r"path/to/augmented")
SEED = 42
EXT  = ".png"
# -------------------------------------------------------------------------


def augment(img):
    if random.choice([True, False]):                 # random horizontal flip
        img = cv2.flip(img, 1)
    angle = random.uniform(-30, 30)                  # random rotation +/- 30 deg
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, m, (w, h))
    img = cv2.add(img, random.randint(-50, 50))      # random brightness +/- 50
    noise = np.random.normal(loc=0, scale=0.5, size=img.shape)   # Gaussian noise
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    DST.mkdir(parents=True, exist_ok=True)
    tiles = sorted(SRC.glob(f"*{EXT}"))
    print(f"augmenting {len(tiles)} tiles -> {DST}")
    for i, p in enumerate(tiles):
        im = cv2.imread(str(p))
        if im is None:
            print(f"  skip (unreadable): {p.name}")
            continue
        cv2.imwrite(str(DST / f"{p.stem}_aug{EXT}"), augment(im))
        if i % 1000 == 0:
            print(f"  {i}/{len(tiles)} ...")
    print("done")


if __name__ == "__main__":
    main()
