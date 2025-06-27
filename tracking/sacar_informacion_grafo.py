import os
import glob
import cv2
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# === Configuración ===
ROOT_FOLDER = "/home/gmanty/code/Workspace_prueba"
MASK_FOLDER = os.path.join(ROOT_FOLDER, "or_masks_50")
IMG_FOLDER = os.path.join(ROOT_FOLDER, "imagenes_og_re_50")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "outputs_tracking")
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "blobs_info_wtshd.pkl")
VIS_FOLDER = os.path.join(OUTPUT_FOLDER, "blobs_visual")
os.makedirs(VIS_FOLDER, exist_ok=True)

INPUT_SIZE = 256
AREA_MIN = 50

# === Parámetros de flujo óptico para clasificación ===
flow_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=21,
    iterations=5,
    poly_n=7,
    poly_sigma=1.5,
    flags=0
)

# === Funciones Auxiliares ===
def clasificar_blob(mask, flow, area_min=200, area_max=3000, umbral_varianza=2.0):
    area = np.sum(mask)
    if area < area_min:
        return "ruido"
    if area > area_max:
        return "group"

    if flow is not None:
        movimiento = flow[mask > 0]
        if movimiento.shape[0] > 10:
            var_total = np.var(movimiento[:, 0]) + np.var(movimiento[:, 1])
            if var_total > umbral_varianza:
                return "group"
    return "individual"

def separar_por_watershed(blob_mask):
    dist_transform = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(blob_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    fake_rgb = np.stack([blob_mask]*3, axis=-1)
    markers = cv2.watershed(fake_rgb, markers)

    masks = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        new_mask = (markers == label).astype(np.uint8)
        if np.sum(new_mask) > 10:
            masks.append(new_mask)
    return masks

mask_paths = sorted(glob.glob(os.path.join(MASK_FOLDER, "*.png")))
img_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
assert len(mask_paths) == len(img_paths)

# === Estructura de salida ===
blobs_info = defaultdict(list)

for frame_idx in tqdm(range(len(mask_paths)-1), desc="Extrayendo blobs"):
    mask1 = cv2.imread(mask_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask_paths[frame_idx+1], cv2.IMREAD_GRAYSCALE)
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    frame = cv2.imread(img_paths[frame_idx])
    overlay = frame.copy()

    flow = cv2.calcOpticalFlowFarneback(mask1.astype(np.float32), mask2.astype(np.float32), None, **flow_params)
    yx = np.column_stack(np.where(mask1 > 0))
    vectors = flow[mask1 > 0]
    features = np.hstack([yx, vectors])

    clustering = DBSCAN(eps=3, min_samples=10).fit(features)
    labels = clustering.labels_
    num_blobs = len(set(labels)) - (1 if -1 in labels else 0)

    for label in range(num_blobs):
        blob_mask = np.zeros_like(mask1, dtype=np.uint8)
        coords = yx[labels == label]
        blob_mask[coords[:, 0], coords[:, 1]] = 1

        area = np.sum(blob_mask)
        if area < AREA_MIN:
            continue

        M = cv2.moments(blob_mask)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        tipo = clasificar_blob(blob_mask, flow=flow)

        if tipo == "group":
            submasks = separar_por_watershed(blob_mask)
            for sub_mask in submasks:
                area_sub = np.sum(sub_mask)
                if area_sub < AREA_MIN:
                    continue
                M_sub = cv2.moments(sub_mask)
                if M_sub["m00"] == 0:
                    continue
                cx_sub = int(M_sub["m10"] / M_sub["m00"])
                cy_sub = int(M_sub["m01"] / M_sub["m00"])
                y_coords, x_coords = np.where(sub_mask > 0)
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)

                tipo_sub = clasificar_blob(sub_mask, flow=flow)

                blobs_info[frame_idx].append({
                    "blob_id": len(blobs_info[frame_idx]),
                    "centroid": (cx_sub, cy_sub),
                    "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
                    "mask": sub_mask,
                    "type": tipo_sub
                })
        else:
            y_coords, x_coords = np.where(blob_mask > 0)
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            blobs_info[frame_idx].append({
                "blob_id": len(blobs_info[frame_idx]),
                "centroid": (cx, cy),
                "bbox": (int(x_min), int(y_min), int(x_max), int(y_max)),
                "mask": blob_mask,
                "type": tipo
            })

# === Guardar en pickle ===
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(blobs_info, f)

print(f"✅ Información de blobs guardada en {OUTPUT_PATH}")
