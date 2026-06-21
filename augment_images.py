"""
Make one augmented copy of each tile -- a random flip, a +/- 30 deg rotation, a
+/- 50 brightness shift, and a little Gaussian noise. File is saved as <name>_aug.png.
"""

import random
import cv2
import numpy as np
from pathlib import Path

SRC  = Path("path/to/tiles")          # tiles to augment (e.g. a class folder)
DST  = Path("path/to/augmented")
SEED = 42
EXT  = ".png"


def augment(image):
    if random.choice([True, False]):                 # maybe flip left-to-right
        image = cv2.flip(image, 1)
    angle = random.uniform(-30, 30)                  # rotate by up to +/- 30 degrees
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (width, height))
    image = cv2.add(image, random.randint(-50, 50))  # shift brightness by +/- 50
    noise = np.random.normal(loc=0, scale=0.5, size=image.shape)   # add a little Gaussian noise
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    DST.mkdir(parents=True, exist_ok=True)
    tiles = sorted(SRC.glob(f"*{EXT}"))
    print(f"Augmenting {len(tiles)} tiles -> {DST}")
    for index, tile_path in enumerate(tiles):
        image = cv2.imread(str(tile_path))
        if image is None:
            print(f"  skipping (could not read): {tile_path.name}")
            continue
        cv2.imwrite(str(DST / f"{tile_path.stem}_aug{EXT}"), augment(image))
        if index % 1000 == 0:
            print(f"  {index}/{len(tiles)} ...")
    print("Done.")


if __name__ == "__main__":
    main()
