"""
tile_images.py -- split full UAS photographs into an N x N grid of tiles.

Reproduces the dataset's tiling ("sectorization") step. Each source photograph
(named e.g. 00000123.png) is cropped into GRID x GRID equal tiles, saved as
00000123_01.png ... 00000123_<NN>.png so that the 8-digit photo id is preserved
in every tile name (this is what makes the leakage-free, photo-level split
possible downstream).
"""
from pathlib import Path
from PIL import Image

# ----------------------------- configuration -----------------------------
SRC  = Path(r"path/to/original_photos")   # full photos, named <8-digit>.png
DST  = Path(r"path/to/tiled_output")
GRID = 6                                   # 6 -> 6x6 = 36 tiles (3 -> 9, etc.)
EXT  = ".png"
# -------------------------------------------------------------------------


def tile_photo(img_path: Path, dst: Path, grid: int) -> int:
    img = Image.open(img_path)
    w, h = img.size
    tw, th = w // grid, h // grid
    n = 0
    for row in range(grid):
        for col in range(grid):
            n += 1
            box = (col * tw, row * th, (col + 1) * tw, (row + 1) * th)
            img.crop(box).save(dst / f"{img_path.stem}_{n:02d}{EXT}")
    return n


def main():
    DST.mkdir(parents=True, exist_ok=True)
    photos = sorted(SRC.glob(f"*{EXT}"))
    print(f"tiling {len(photos)} photos into {GRID}x{GRID} = {GRID * GRID} tiles each")
    for i, p in enumerate(photos):
        tile_photo(p, DST, GRID)
        if i % 200 == 0:
            print(f"  {i}/{len(photos)} ...")
    print(f"done -> {DST}")


if __name__ == "__main__":
    main()
