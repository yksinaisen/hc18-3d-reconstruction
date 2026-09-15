import cv2
import numpy as np
import trimesh
import pandas as pd
from scipy.ndimage import gaussian_filter
import os

# ===== НАСТРОЙКИ =====
BASE = "414_HC"
IMG_PATH = f"data/images/{BASE}.png"
MASK_PATH = f"data/masks/{BASE}_mask.png"
CSV_PATH = "data/training_set_pixel_size_and_HC.csv"
OUTPUT_STL = f"output/anatomical_{BASE}.stl"

HEIGHT_MAX_MM = 3.0      # максимальное смещение бугров (мм) — небольшое!
SMOOTH_SIGMA = 8.0       # сглаживание карты высот
SUBDIVISIONS = 4         # детальность эллипсоида (4 = ~5000 вершин)
# ====================

os.makedirs("output", exist_ok=True)

# 1. Загружаем УЗИ, маску, CSV
img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
df = pd.read_csv(CSV_PATH).rename(columns={"pixel size(mm)": "pixel_size"})
row = df[df["filename"] == f"{BASE}.png"].iloc[0]
pixel_size = row["pixel_size"]

print(f"Image shape: {img.shape}")
print(f"pixel_size: {pixel_size:.6f} mm/px")

# 2. Карта высот из яркости
mask_bool = mask > 127
img_masked = img.copy()
img_masked[~mask_bool] = 0
img_smooth = gaussian_filter(img_masked.astype(float), sigma=SMOOTH_SIGMA)

# Нормируем в диапазон [-1, +1] относительно среднего
# Центрируем: средняя яркость внутри маски = 0
mean_intensity = img_smooth[mask_bool].mean()
height_map = (img_smooth - mean_intensity) / 255.0 * 2  # -1..+1 примерно

# Ограничиваем в [-1, 1]
height_map = np.clip(height_map, -1, 1)
print(f"Height map range: {height_map.min():.2f} .. {height_map.max():.2f}")

# 3. Строим эллипсоид из маски
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
(cx_px, cy_px), (MA, ma), angle = cv2.fitEllipse(largest)

a_mm = MA / 2 * pixel_size
b_mm = ma / 2 * pixel_size
c_mm = (a_mm + b_mm) / 2
print(f"Semi-axes: a={a_mm:.2f}, b={b_mm:.2f}, c={c_mm:.2f} mm")

# Базовый эллипсоид
mesh = trimesh.creation.icosphere(subdivisions=SUBDIVISIONS, radius=1.0)
mesh.apply_scale([a_mm, b_mm, c_mm])
# Смещаем центр в центр эллипса
mesh.apply_translation([cx_px * pixel_size, cy_px * pixel_size, 0])

print(f"Ellipsoid: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# 4. Для каждой вершины эллипсоида находим яркость в соответствующей точке УЗИ
#    Координаты вершины (x, y, z) в мм -> в пиксели -> берём яркость
vertices = mesh.vertices.copy()

# Проекция вершины на плоскость XY (в пиксели)
vx_px = (vertices[:, 0] / pixel_size).astype(int)
vy_px = (vertices[:, 1] / pixel_size).astype(int)

# Ограничиваем индексы
vx_px = np.clip(vx_px, 0, img.shape[1] - 1)
vy_px = np.clip(vy_px, 0, img.shape[0] - 1)

# Берём яркость из карты высот
brightness = height_map[vy_px, vx_px]  # от -1 до +1

# 5. Смещаем каждую вершину по нормали
#    Нормаль — от центра эллипсоида наружу
center = np.array([cx_px * pixel_size, cy_px * pixel_size, 0])
normals = vertices - center
normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
normals_norm[normals_norm == 0] = 1  # защита от деления на 0
unit_normals = normals / normals_norm

# Смещение = яркость * масштаб
displacement = brightness[:, None] * HEIGHT_MAX_MM * unit_normals

# Новые вершины
new_vertices = vertices + displacement

# 6. Обновляем меш
mesh_anatomical = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
mesh_anatomical.fix_normals()

print(f"Watertight: {mesh_anatomical.is_watertight}")
print(f"Volume: {mesh_anatomical.volume:.2f} cubic mm")

# 7. Экспорт
mesh_anatomical.export(OUTPUT_STL)
print(f"STL saved: {OUTPUT_STL}")

if os.path.exists(OUTPUT_STL):
    size = os.path.getsize(OUTPUT_STL)
    print(f"File size: {size} bytes")