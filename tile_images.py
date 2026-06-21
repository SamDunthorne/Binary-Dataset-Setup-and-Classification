"""
Split each full UAS photo into a GRID x GRID set of tiles.

Each photo (e.g. 00000123.png) is cropped into GRID*GRID equal tiles saved as
00000123_01.png ... 00000123_NN.png, keeping the 8-digit photo id in every tile
name -- which is what lets the dataset be split by whole photo later.
"""
from pathlib import Path
from PIL import Image

# Configuration
SRC  = Path("path/to/original_photos")   # full photos, named <8-digit>.png
DST  = Path("path/to/tiled_output")
GRID = 6                                   # 6 -> 6x6 = 36 tiles (3 -> 9, etc.)
EXT  = ".png"


def tile_photo(photo_path: Path, output_dir: Path, grid: int) -> int:
    """Crop one photo into grid x grid tiles and save them. Returns the tile count."""
    image = Image.open(photo_path)
    width, height = image.size
    tile_width, tile_height = width // grid, height // grid
    tile_number = 0
    for row in range(grid):
        for col in range(grid):
            tile_number += 1
            left, top = col * tile_width, row * tile_height
            crop_box = (left, top, left + tile_width, top + tile_height)
            image.crop(crop_box).save(output_dir / f"{photo_path.stem}_{tile_number:02d}{EXT}")
    return tile_number


def main():
    DST.mkdir(parents=True, exist_ok=True)
    photos = sorted(SRC.glob(f"*{EXT}"))
    print(f"Tiling {len(photos)} photos into {GRID}x{GRID} = {GRID * GRID} tiles each.")
    for index, photo_path in enumerate(photos):
        tile_photo(photo_path, DST, GRID)
        if index % 200 == 0:
            print(f"  {index}/{len(photos)} ...")
    print(f"Done. Tiles written to {DST}")


if __name__ == "__main__":
    main()
