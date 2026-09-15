import trimesh
import pandas as pd
import glob
import os

# Пути
STL_DIR = "output/stl"
OUT_CSV = "output/model_stats.csv"

# Все STL-файлы
stl_files = sorted(glob.glob(os.path.join(STL_DIR, "*_3d.stl")))
print(f"Found STL files: {len(stl_files)}")

rows = []
errors = []
for stl_path in stl_files:
    name = os.path.basename(stl_path).replace("_3d.stl", "")
    try:
        mesh = trimesh.load(stl_path)
        bbox = mesh.bounding_box.extents
        rows.append({
            "filename": name,
            "volume_mm3": float(mesh.volume),
            "bbox_x_mm": float(bbox[0]),
            "bbox_y_mm": float(bbox[1]),
            "bbox_z_mm": float(bbox[2]),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
        })
    except Exception as e:
        errors.append((name, str(e)))

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"\n=== Summary ===")
print(f"Total models: {len(df)}")
print(f"Errors: {len(errors)}")
print()
print(f"Volume (cubic mm):")
print(f"  min:  {df['volume_mm3'].min():.1f}")
print(f"  mean: {df['volume_mm3'].mean():.1f}")
print(f"  max:  {df['volume_mm3'].max():.1f}")
print()
print(f"BBox X (mm): min={df['bbox_x_mm'].min():.2f}, mean={df['bbox_x_mm'].mean():.2f}, max={df['bbox_x_mm'].max():.2f}")
print(f"BBox Y (mm): min={df['bbox_y_mm'].min():.2f}, mean={df['bbox_y_mm'].mean():.2f}, max={df['bbox_y_mm'].max():.2f}")
print(f"BBox Z (mm): min={df['bbox_z_mm'].min():.2f}, mean={df['bbox_z_mm'].mean():.2f}, max={df['bbox_z_mm'].max():.2f}")
print()
print(f"Saved: {OUT_CSV}")

if errors:
    print(f"\nFirst 5 errors:")
    for n, e in errors[:5]:
        print(f"  {n}: {e}")