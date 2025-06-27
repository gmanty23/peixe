import os
import pickle
from collections import defaultdict
from tqdm import tqdm

IOU_THRESHOLD = 0.75

# === Utilidad para calcular IoU entre dos bounding boxes ===
def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# === Cargar blobs_info.pkl ===
with open("/home/gmanty/code/Workspace_prueba/outputs_tracking/blobs_info.pkl", "rb") as f:
    blobs_info = pickle.load(f)

# === Inicializar grafo ===
tracking_graph = {}
next_id = 0

# === Crear nodos ===
for frame, blobs in blobs_info.items():
    for blob in blobs:
        node_key = (frame, blob["blob_id"])
        tracking_graph[node_key] = {
            "centroid": blob["centroid"],
            "bbox": blob["bbox"],
            "mask": blob["mask"],
            "type": blob["type"],
            "ids": [],
            "vecinos_siguientes": [],
            "vecinos_anteriores": []
        }

# === Asignar IDs iniciales a blobs individuales del primer frame ===
for blob in blobs_info[0]:
    if blob["type"] == "individual":
        node = tracking_graph[(0, blob["blob_id"])]
        node["ids"] = [next_id]
        next_id += 1

# === Conectar nodos por IoU ===
frames = sorted(blobs_info.keys())
from tqdm import tqdm

for f in tqdm(frames[:-1], desc="Construyendo conexiones"):
    blobs_f = blobs_info[f]
    blobs_f1 = blobs_info[f+1]

    for blob_b in blobs_f1:
        mejor_iou = 0
        mejor_blob_a = None

        for blob_a in blobs_f:
            iou = calcular_iou(blob_a["bbox"], blob_b["bbox"])
            if iou > IOU_THRESHOLD and iou > mejor_iou:
                mejor_iou = iou
                mejor_blob_a = blob_a

        if mejor_blob_a:
            key_a = (f, mejor_blob_a["blob_id"])
            key_b = (f+1, blob_b["blob_id"])

            tracking_graph[key_a]["vecinos_siguientes"].append(key_b)
            tracking_graph[key_b]["vecinos_anteriores"].append(key_a)

            # Propagar IDs
            if tracking_graph[key_a]["ids"]:
                tracking_graph[key_b]["ids"].extend(tracking_graph[key_a]["ids"])
            else:
                # Si el anterior no tenía ID, generar nuevo si tipo es individual
                if blob_b["type"] == "individual" and not tracking_graph[key_b]["ids"]:
                    tracking_graph[key_b]["ids"].append(next_id)
                    next_id += 1

# === Guardar grafo ===
with open("/home/gmanty/code/Workspace_prueba/outputs_tracking/tracking_graph.pkl", "wb") as f:
    pickle.dump(tracking_graph, f)

print("✅ Grafo de tracking construido y guardado.")
