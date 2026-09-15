import cv2
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

SAMPLES = ["000_HC", "100_HC", "300_HC", "500_HC"]  # 4 примера
IMG_DIR = "data/images"
MASK_DIR = "data/masks"
STL_DIR = "output/stl"
OUTPUT_PNG = "output/comparison.png"

os.makedirs("output", exist_ok=True)

# Фильтруем только те, для которых есть все три файла
valid_samples = []
for base in SAMPLES:
    img_path = os.path.join(IMG_DIR, f"{base}.png")
    mask_path = os.path.join(MASK_DIR, f"{base}_mask.png")
    stl_path = os.path.join(STL_DIR, f"{base}_3d.stl")
    if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(stl_path):
        valid_samples.append(base)
    else:
        print(f"Skipping {base}: missing files")
        print(f"  img:  {os.path.exists(img_path)}")
        print(f"  mask: {os.path.exists(mask_path)}")
        print(f"  stl:  {os.path.exists(stl_path)}")

if not valid_samples:
    raise RuntimeError("No valid samples found. Check paths.")

print(f"Plotting {len(valid_samples)} samples: {valid_samples}")

n = len(valid_samples)
fig = plt.figure(figsize=(16, 4 * n))

for i, base in enumerate(valid_samples):
    img_path = os.path.join(IMG_DIR, f"{base}.png")
    mask_path = os.path.join(MASK_DIR, f"{base}_mask.png")
    stl_path = os.path.join(STL_DIR, f"{base}_3d.stl")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mesh = trimesh.load(stl_path)

    ax1 = fig.add_subplot(n, 3, i * 3 + 1)
    ax1.imshow(img, cmap="gray")
    ax1.set_title(f"{base} - Ultrasound", fontsize=12)
    ax1.axis("off")

    ax2 = fig.add_subplot(n, 3, i * 3 + 2)
    ax2.imshow(mask, cmap="gray")
    ax2.set_title(f"{base} - Mask", fontsize=12)
    ax2.axis("off")

    ax3 = fig.add_subplot(n, 3, i * 3 + 3, projection="3d")

    verts = mesh.vertices
    faces = mesh.faces

    # Ограничиваем количество граней для скорости (если очень много)
    if len(faces) > 5000:
        idx = np.random.choice(len(faces), 5000, replace=False)
        faces = faces[idx]

    poly = Poly3DCollection(
        verts[faces],
        alpha=0.75,
        facecolor="lightblue",
        edgecolor="gray",
        linewidth=0.1,
    )
    ax3.add_collection3d(poly)

    # Масштаб по осям
    ax3.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax3.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax3.set_zlim(verts[:, 2].min(), verts[:, 2].max())

    # Одинаковый масштаб по осям (чтобы эллипсоид не выглядел сплюснутым)
    max_range = max(
        verts[:, 0].max() - verts[:, 0].min(),
        verts[:, 1].max() - verts[:, 1].min(),
        verts[:, 2].max() - verts[:, 2].min(),
    ) / 2
    mid = [
        (verts[:, 0].max() + verts[:, 0].min()) / 2,
        (verts[:, 1].max() + verts[:, 1].min()) / 2,
        (verts[:, 2].max() + verts[:, 2].min()) / 2,
    ]
    ax3.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax3.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax3.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax3.set_title(f"{base} - 3D ({mesh.volume:.0f} cubic mm)", fontsize=12)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
print(f"Saved: {OUTPUT_PNG}")
plt.show()