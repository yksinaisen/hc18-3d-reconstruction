import cv2
import numpy as np
import trimesh
import pandas as pd
import os
import glob
from tqdm import tqdm

MASK_DIR = "data/masks"
CSV_PATH = "data/training_set_pixel_size_and_HC.csv"
OUTPUT_DIR = "output/stl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV -> dict: full name (без .png) -> pixel_size
df = pd.read_csv(CSV_PATH).rename(columns={"pixel size(mm)": "pixel_size"})
csv_dict = {row["filename"].replace(".png", ""): row["pixel_size"] for _, row in df.iterrows()}
print(f"CSV entries: {len(csv_dict)}")

# Все маски (и _HC, и _2HC, и _3HC)
mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*_mask.png")))
print(f"Masks found: {len(mask_files)}")

errors = []
skipped = 0
created = 0

for mask_path in tqdm(mask_files, desc="Building 3D"):
    fname = os.path.basename(mask_path)              # 010_2HC_mask.png
    name = fname.replace("_mask.png", "")             # 010_2HC
    out_path = os.path.join(OUTPUT_DIR, f"{name}_3d.stl")

    if os.path.exists(out_path):
        skipped += 1
        continue

    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("mask not loaded")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("no contours")
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(largest)

        pixel_size = csv_dict.get(name)
        if pixel_size is None:
            raise ValueError(f"no pixel_size for {name}")

        a_mm = MA / 2 * pixel_size
        b_mm = ma / 2 * pixel_size
        c_mm = (a_mm + b_mm) / 2

        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale([a_mm, b_mm, c_mm])
        mesh.export(out_path)
        created += 1

    except Exception as e:
        errors.append((fname, str(e)))

print(f"\nSTL created: {created}")
print(f"STL skipped (already exist): {skipped}")
print(f"Errors: {len(errors)}")
print(f"Total STL in output/stl: {len(os.listdir(OUTPUT_DIR))}")
for n, e in errors[:10]:
    print(f"  {n}: {e}")