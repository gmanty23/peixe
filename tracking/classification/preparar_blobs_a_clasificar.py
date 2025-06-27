import os
import glob
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import random

# === CONFIGURACIÓN ===
ROOT_FOLDER = "/home/gmanty/code/USCL2-194221-194721/Workspace_bloque_1"
MASK_FOLDER = os.path.join(ROOT_FOLDER, "or_masks")
IMG_FOLDER = os.path.join(ROOT_FOLDER, "imagenes_og")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "outputs_tracking")
OUTPUT_BLOBS_FOLDER = os.path.join(OUTPUT_FOLDER, "blobs_sin_clasificar")
OUTPUT_FRAMES_FOLDER = os.path.join(OUTPUT_FOLDER, "frames")
FRAME_VISUALIZATION_FLAG = False
os.makedirs(OUTPUT_BLOBS_FOLDER, exist_ok=True)
if FRAME_VISUALIZATION_FLAG:
    os.makedirs(OUTPUT_FRAMES_FOLDER, exist_ok=True)


FRAME_STEP = 10  # Procesar 1 de cada N frames
EPS = 3        # Parámetro de DBSCAN
MIN_SAMPLES = 10

# === Cargar rutas ===
mask_paths = sorted(glob.glob(os.path.join(MASK_FOLDER, "*.png")))
img_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
assert len(mask_paths) == len(img_paths), "¡Mismatch entre máscaras e imágenes!"

# === Flujo óptico ===
flow_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=21,
    iterations=5,
    poly_n=7,
    poly_sigma=1.5,
    flags=0
)

# === Guardar blobs individuales ===
def exportar_blob(frame, mask, frame_idx, blob_idx, export_folder):
    if np.sum(mask) == 0:
        return
    masked = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))
    filename = f"frame_{frame_idx:04d}_blob_{blob_idx:02d}.png"
    path = os.path.join(export_folder, filename)
    cv2.imwrite(path, masked)

# === Visualización coloreada ===
def visualizar_blobs_coloreados(frame, blob_masks, frame_idx, output_folder):
    overlay = frame.copy()

    for i, mask in enumerate(blob_masks):
        color = [random.randint(0, 255) for _ in range(3)]  # color aleatorio
        for c in range(3):
            overlay[:, :, c][mask > 0] = (
                0.6 * color[c] + 0.4 * overlay[:, :, c][mask > 0]
            ).astype(np.uint8)

        # Opcional: número del blob encima del centroide
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(overlay, str(i), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
    # Guardar imagen
    out_path = os.path.join(output_folder, f"frame_{frame_idx:04d}_blobs.png")
    cv2.imwrite(out_path, overlay)

# === Bucle principal ===
for i in range(0, len(mask_paths) - 1, FRAME_STEP):
    print(f"[INFO] Procesando frame {i}")

    # Cargar imágenes y máscaras
    img1 = cv2.imread(img_paths[i])
    img2 = cv2.imread(img_paths[i+1])
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mask1 = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask_paths[i+1], cv2.IMREAD_GRAYSCALE)
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Aplicar máscaras para limitar el flujo óptico solo a zonas con pez
    gray1_masked = cv2.bitwise_and(gray1, gray1, mask=mask1)
    gray2_masked = cv2.bitwise_and(gray2, gray2, mask=mask2)

    # Calcular flujo óptico
    flow = cv2.calcOpticalFlowFarneback(
        gray1_masked.astype(np.float32),
        gray2_masked.astype(np.float32),
        None, **flow_params
    )

    # Obtener puntos con movimiento
    yx = np.column_stack(np.where(mask1 > 0))       # coordenadas [y, x]
    vectors = flow[mask1 > 0]                       # vectores de movimiento [dx, dy]

    # Obtener color (R, G, B) del frame original en los píxeles activos
    color_pixels = img1[yx[:, 0], yx[:, 1]]  # shape: (n, 3) → BGR

    # Convertir de BGR a RGB (opcional)
    color_pixels = color_pixels[:, ::-1] / 255 # ahora es RGB

    # (Opcional) normalizar colores (escala 0-1 o amplificar impacto)
    color_weight = 1 #para ponerlo en la misma escala que el movimiento
    color_scaled = color_pixels * color_weight

    # Unir todo: posición + movimiento + color
    features = np.hstack([yx, vectors, color_scaled])

    # Clustering con DBSCAN
    clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(features)
    labels = clustering.labels_
    num_blobs = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"    → {num_blobs} blobs detectados")

    # Guardar blobs
    blob_masks = []
    for label in range(num_blobs):
        blob_mask = np.zeros_like(mask1, dtype=np.uint8)
        coords = yx[labels == label]
        blob_mask[coords[:, 0], coords[:, 1]] = 1
        exportar_blob(img1, blob_mask, i, label, OUTPUT_BLOBS_FOLDER)
        blob_masks.append(blob_mask)

    # Visualización de blobs sobre el frame
    if FRAME_VISUALIZATION_FLAG:
        visualizar_blobs_coloreados(img1, blob_masks, i, OUTPUT_FRAMES_FOLDER)