import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ruta al directorio raíz donde están los subdirectorios con las máscaras
root_dir = "/home/gms/AnemoNAS/Workspace/masks_guardadas_2/"
# Ruta al directorio donde se guardarán las máscaras resultantes (una por subdirectorio)
out_dir = "/home/gms/AnemoNAS/Workspace/or_masks/"


os.makedirs(out_dir, exist_ok=True)

# Obtener todos los subdirectorios
subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Obtener lista de archivos desde el primer subdirectorio (asumiendo que todos tienen los mismos nombres y orden)
frame_names = sorted([f for f in os.listdir(subdirs[0]) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])

for frame_name in tqdm(frame_names, desc="Procesando frames"):
    frame_or = None

    for subdir in subdirs:
        frame_path = os.path.join(subdir, frame_name)
        mask = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"No se pudo leer la imagen: {frame_path}")
            continue

        mask = (mask > 0).astype(np.uint8)

        if frame_or is None:
            frame_or = mask.copy()
        else:
            frame_or = cv2.bitwise_or(frame_or, mask)

    if frame_or is not None:
        output_path = os.path.join(out_dir, frame_name)
        frame_or = (frame_or * 255).astype(np.uint8)
        cv2.imwrite(output_path, frame_or)
    else:
        print(f"No se pudieron procesar las máscaras para el frame: {frame_name}")