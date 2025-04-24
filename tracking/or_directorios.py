import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

root_dir = "/home/gmanty/code/USCL2-194221-194721/Workspace_bloque_1"
# Ruta al directorio raíz donde están los subdirectorios con las máscaras
mask_dir = os.path.join(root_dir, "masks_guardadas")
# Ruta al directorio donde se guardarán las máscaras resultantes (una por subdirectorio)
out_dir = os.path.join(root_dir, "or_masks")
# Crear el directorio de salida si no existe
os.makedirs(out_dir, exist_ok=True)



# Obtener todos los subdirectorios
subdirs = [os.path.join(mask_dir, d) for d in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, d))]

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