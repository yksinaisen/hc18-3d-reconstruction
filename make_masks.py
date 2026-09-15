import cv2
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm

ANNOT_DIR = "data/annotations"
MASK_DIR  = "data/masks"
CSV_PATH  = "data/training_set_pixel_size_and_HC.csv"

os.makedirs(MASK_DIR, exist_ok=True)

# Загружаем CSV
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={
    "pixel size(mm)": "pixel_size",
    "head circumference (mm)": "HC"
})

# Создаём словарь: base -> (pixel_size, HC)
# base — это имя без .png и без _HC / _Annotation
def get_base(filename):
    """000_HC.png -> 000 ; 010_2HC.png -> 010_2 ; 010_2HC_Annotation.png -> 010_2"""
    name = filename.replace(".png", "")
    name = name.replace("_Annotation", "")
    name = name.replace("_HC", "")
    return name

csv_dict = {}
for _, row in df.iterrows():
    base = get_base(row["filename"])
    csv_dict[base] = (row["pixel_size"], row["HC"])

print(f"CSV entries: {len(csv_dict)}")

# Все аннотации
annot_files = sorted(glob.glob(os.path.join(ANNOT_DIR, "*_Annotation.png")))
print(f"Annotations found: {len(annot_files)}")

results = []
errors = []

for annot_path in tqdm(annot_files, desc="Processing"):
    name = os.path.basename(annot_path)

    try:
        # 1. Загружаем
        annot = cv2.imread(annot_path, cv2.IMREAD_GRAYSCALE)
        if annot is None:
            raise ValueError("failed to load")

        # 2. Порог
        _, thresh = cv2.threshold(annot, 127, 255, cv2.THRESH_BINARY)

        # 3. Морфологическое закрытие
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4. Контуры
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("no contours found")

        # 5. Самый большой контур -> маска
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(annot)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

        # 6. Сохраняем маску с именем как у исходного УЗИ
        mask_name = name.replace("_Annotation.png", "_mask.png")
        mask_path = os.path.join(MASK_DIR, mask_name)
        cv2.imwrite(mask_path, mask)

        # 7. Достаём pixel_size и HC из CSV
        base = get_base(name)
        if base in csv_dict:
            pixel_size, HC_csv = csv_dict[base]

            # Эллипс из маски
            ellipse = cv2.fitEllipse(largest)
            (cx, cy), (MA, ma), angle = ellipse
            a_mm = MA / 2 * pixel_size
            b_mm = ma / 2 * pixel_size

            # Окружность эллипса (формула Рамануджана)
            h = ((a_mm - b_mm) ** 2) / ((a_mm + b_mm) ** 2)
            HC_calc = np.pi * (a_mm + b_mm) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

            results.append({
                "filename": name,
                "base": base,
                "HC_csv": HC_csv,
                "HC_mask": HC_calc,
                "diff": abs(HC_csv - HC_calc),
                "pixel_size": pixel_size
            })
        else:
            results.append({
                "filename": name,
                "base": base,
                "HC_csv": None,
                "HC_mask": None,
                "diff": None,
                "pixel_size": None
            })

    except Exception as e:
        errors.append((name, str(e)))


print(f"\nDone. Masks created: {len(os.listdir(MASK_DIR))}")
print(f"Errors: {len(errors)}")

if errors:
    print("\nFirst 10 errors:")
    for name, err in errors[:10]:
        print(f"  {name}: {err}")

# Метрики
res_df = pd.DataFrame(results)
res_df.to_csv("output/mask_validation.csv", index=False)

valid = res_df.dropna(subset=["diff"])
if len(valid) > 0:
    print(f"\n=== Stats over {len(valid)} files ===")
    print(f"Mean HC difference: {valid['diff'].mean():.2f} mm")
    print(f"Median difference:  {valid['diff'].median():.2f} mm")
    print(f"Max difference:     {valid['diff'].max():.2f} mm")
    print(f"Files with diff < 2 mm: {(valid['diff'] < 2).sum()} ({(valid['diff'] < 2).mean()*100:.1f}%)")
    print(f"Files with diff > 5 mm: {(valid['diff'] > 5).sum()}")

# Показать проблемные файлы
bad = valid[valid["diff"] > 5]
if len(bad) > 0:
    print(f"\nProblem files (diff > 5 mm):")
    print(bad.sort_values("diff", ascending=False).head(10))