import os
import glob
import cv2
import numpy as np
import pickle
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm  # Indicador de progreso

# === Configuración ===
ROOT_FOLDER = "/home/gmanty/code/Workspace_prueba"
MASK_FOLDER = os.path.join(ROOT_FOLDER, "or_masks_50")
IMG_FOLDER = os.path.join(ROOT_FOLDER, "imagenes_og_re_50")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "outputs_tracking")
OUTPUT_VIS_FOLDER = os.path.join(OUTPUT_FOLDER, "frames_con_ids")
os.makedirs(OUTPUT_VIS_FOLDER, exist_ok=True)

AREA_MIN = 500
MAX_DISTANCE = 50  # Distancia máxima para asociar blobs entre frames

mask_paths = sorted(glob.glob(os.path.join(MASK_FOLDER, "*.png")))
img_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
assert len(mask_paths) == len(img_paths)

# === Inicialización del grafo temporal y asignación de IDs ===
tracking_graph = defaultdict(list)
next_id = 0
prev_centroids = []
prev_ids = []

for i in tqdm(range(len(mask_paths)), desc="Tracking"):
    mask_img = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
    mask = (mask_img > 0).astype(np.uint8)
    frame = cv2.imread(img_paths[i])

    num_labels, labels = cv2.connectedComponents(mask)
    blobs = []

    for label in range(1, num_labels):
        blob_mask = (labels == label).astype(np.uint8)
        area = np.sum(blob_mask)
        if area < AREA_MIN:
            continue  # Descartar como ruido

        M = cv2.moments(blob_mask)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)

        blobs.append({
            "mask": blob_mask,
            "centroid": centroid,
            "ids": []  # Se asignarán más abajo
        })

    # Asociación de blobs con los del frame anterior
    if prev_centroids:
        cost_matrix = np.zeros((len(prev_centroids), len(blobs)), dtype=np.float32)
        for j, pc in enumerate(prev_centroids):
            for k, blob in enumerate(blobs):
                bc = blob["centroid"]
                dist = np.linalg.norm(np.array(pc) - np.array(bc))
                cost_matrix[j, k] = dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        used_blobs = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < MAX_DISTANCE:
                blobs[c]["ids"].append(prev_ids[r])
                used_blobs.add(c)

        # Blobs no asignados reciben nuevo ID
        for idx, blob in enumerate(blobs):
            if idx not in used_blobs:
                blob["ids"].append(next_id)
                next_id += 1
    else:
        # Primer frame: todos los blobs reciben nuevo ID
        for blob in blobs:
            blob["ids"].append(next_id)
            next_id += 1

    # Guardar en el grafo
    for blob in blobs:
        tracking_graph[i].append({
            "mask": blob["mask"],
            "centroid": blob["centroid"],
            "ids": blob["ids"]
        })

    # Visualizar y guardar imagen con IDs
    overlay = frame.copy()
    for blob in blobs:
        mask = blob["mask"].astype(bool)
        cx, cy = blob["centroid"]
        for c in range(3):
            overlay[:, :, c][mask] = (0.6 * 255 + 0.4 * overlay[:, :, c][mask]).astype(np.uint8)
        for id_ in blob["ids"]:
            cv2.putText(overlay, str(id_), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    filename = os.path.join(OUTPUT_VIS_FOLDER, f"frame_{i:04d}_ids.png")
    cv2.imwrite(filename, overlay)

    # Actualizar para el siguiente frame
    prev_centroids = [blob["centroid"] for blob in blobs]
    prev_ids = [blob["ids"][0] for blob in blobs]

# Guardar el grafo temporal para la fase de resolución posterior
with open(os.path.join(OUTPUT_FOLDER, "tracking_graph.pkl"), "wb") as f:
    pickle.dump(tracking_graph, f)

print("✅ Tracking básico completado. Grafo y visualizaciones guardados.")
