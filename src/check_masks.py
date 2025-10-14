from pathlib import Path
from tifffile import imread

mask_dir = Path("data/training_data")
bad_shapes = []

for f in mask_dir.rglob("*_masks.tif"):
    m = imread(f)
    if m.ndim != 2:
        print(f"{f.name}: shape = {m.shape}")
        bad_shapes.append(f)

print(f"\n{len(bad_shapes)} problematic mask(s) found.")
