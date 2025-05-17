import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm

# === Configuración ===
ROOT_FOLDER = "/home/gmanty/code/Workspace_prueba"
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "outputs_tracking")
IMG_FOLDER = os.path.join(ROOT_FOLDER, "imagenes_og_re_50")
BLOBS_INFO_PATH = os.path.join(OUTPUT_FOLDER, "blobs_info_wtshd.pkl")
VIS_WATERSHED_FOLDER = os.path.join(OUTPUT_FOLDER, "blobs_visual_watershed")
os.makedirs(VIS_WATERSHED_FOLDER, exist_ok=True)

# === Cargar blobs_info ===
with open(BLOBS_INFO_PATH, "rb") as f:
    blobs_info = pickle.load(f)

img_paths = sorted([f for f in os.listdir(IMG_FOLDER) if f.endswith('.jpg')])

# === Visualización de blobs ===
for frame_idx, img_name in tqdm(enumerate(img_paths), total=len(img_paths), desc="Visualizando blobs"):
    img_path = os.path.join(IMG_FOLDER, img_name)
    img = cv2.imread(img_path)
    overlay = img.copy()

    if frame_idx in blobs_info:
        for blob in blobs_info[frame_idx]:
            mask = blob["mask"].astype(bool)
            cx, cy = blob["centroid"]
            tipo = blob["type"]

            if tipo == "individual":
                color = (255, 0, 0)  # Azul
            elif tipo == "group":
                color = (0, 0, 255)  # Rojo
            elif tipo == "ruido":
                color = (128, 128, 128)  # Gris
            else:
                color = (255, 255, 255)  # Blanco

            for c in range(3):
                overlay[:, :, c][mask] = (0.6 * color[c] + 0.4 * overlay[:, :, c][mask]).astype("uint8")

            cv2.putText(overlay, tipo[0].upper(), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    output_path = os.path.join(VIS_WATERSHED_FOLDER, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(output_path, overlay)

print("✅ Visualización de blobs con separación Watershed completada.")
