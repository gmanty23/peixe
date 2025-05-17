import os
import glob
import cv2
import pickle
from tqdm import tqdm

# === Configuración ===
ROOT_FOLDER = "/home/gmanty/code/Workspace_prueba"
IMG_FOLDER = os.path.join(ROOT_FOLDER, "imagenes_og_re_50")
GRAPH_PATH = os.path.join(ROOT_FOLDER, "outputs_tracking", "tracking_graph.pkl")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "outputs_tracking", "frames_tracking_ids")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Cargar imágenes y grafo ===
img_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
with open(GRAPH_PATH, "rb") as f:
    tracking_graph = pickle.load(f)

# === Agrupar nodos por frame ===
nodos_por_frame = {}
for (frame, blob_id), data in tracking_graph.items():
    if frame not in nodos_por_frame:
        nodos_por_frame[frame] = []
    nodos_por_frame[frame].append((blob_id, data))

# === Visualizar cada frame ===
for frame_idx in tqdm(range(len(img_paths)), desc="Generando visualizaciones"):
    img = cv2.imread(img_paths[frame_idx])
    overlay = img.copy()

    if frame_idx in nodos_por_frame:
        for blob_id, data in nodos_por_frame[frame_idx]:
            mask = data["mask"].astype(bool)
            cx, cy = data["centroid"]
            ids = data["ids"]
            tipo = data["type"]

            # Asignar color según tipo
            if tipo == "individual":
                color = (255, 0, 0)  # Azul
            elif tipo == "group":
                color = (0, 0, 255)  # Rojo
            elif tipo == "ruido":
                color = (128, 128, 128)  # Gris
            else:
                color = (255, 255, 255)  # Blanco por defecto

            for c in range(3):
                overlay[:, :, c][mask] = (0.6 * color[c] + 0.4 * overlay[:, :, c][mask]).astype("uint8")

            for id_ in ids:
                cv2.putText(overlay, str(id_), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    output_path = os.path.join(OUTPUT_FOLDER, f"frame_{frame_idx:04d}_ids.png")
    cv2.imwrite(output_path, overlay)

print("✅ Visualización por frame con IDs completada.")
