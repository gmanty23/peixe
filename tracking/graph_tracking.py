import os
import glob
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import matplotlib.pyplot as plt


# === Configuración del usuario ===
ROOT_FOLDER = "/home/gmanty/code/Workspace_prueba"
MASK_FOLDER = os.path.join(ROOT_FOLDER, "or_masks_50")
IMG_FOLDER = os.path.join(ROOT_FOLDER, "imagenes_og_re_50")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "outputs_tracking")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
EXPORT_BLOBS = True
BLOBS_EXPORT_FOLDER = os.path.join(OUTPUT_FOLDER, "blobs_sin_clasificar")
os.makedirs(BLOBS_EXPORT_FOLDER, exist_ok=True)
for f in os.listdir(BLOBS_EXPORT_FOLDER):
    os.remove(os.path.join(BLOBS_EXPORT_FOLDER, f))


mask_paths = sorted(glob.glob(os.path.join(MASK_FOLDER, "*.png"))) 
img_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))

# Comprobar que las listas de máscaras e imágenes tienen la misma longitud
assert len(mask_paths) == len(img_paths)

# Parámetros de flujo óptico
flow_params = dict(
    pyr_scale=0.5,     # resolución en pirámide
    levels=3,          # número de escalas de imagen
    winsize=21,        # tamaño de ventana de búsqueda
    iterations=5,      # refinamientos por nivel
    poly_n=7,          # vecindario de polinomio (para suavizar)
    poly_sigma=1.5,    # desviación de suavizado
    flags=0
)


# === Funciones Auxiliares ===
def clasificar_blob(mask, flow=None, area_min=200, area_max=3000, usar_flow=False, umbral_varianza=2.0):
    area = np.sum(mask)
    if area < area_min:
        return "ruido"
    if area > area_max:
        return "group"

    if usar_flow and flow is not None:
        movimiento = flow[mask > 0]
        if movimiento.shape[0] > 10:  # hay suficientes vectores
            var_total = np.var(movimiento[:, 0]) + np.var(movimiento[:, 1])
            if var_total > umbral_varianza:
                return "group"
    return "individual"
    
def visualizar_blobs_coloreados(frame_idx, tracking_graph, frame_img, output_folder):
    blobs = tracking_graph.get(frame_idx, [])
    overlay = frame_img.copy()

    for idx, blob in enumerate(blobs):
        tipo = blob["type"]
        mask = blob["mask"].astype(bool)
        cx, cy = blob["centroid"]

        if tipo == "individual":
            color = (255, 0, 0)  # Azul
            label = "I"
        elif tipo == "group":
            color = (0, 0, 255)  # Rojo
            label = "G"
        elif tipo == "ruido":
            color = (128, 128, 128)  # Gris
            label = "R"
        else:
            color = (255, 255, 255)
            label = "?"

        # Pintar el blob con transparencia
        for c in range(3):
            overlay[:, :, c][mask] = (0.6 * color[c] + 0.4 * overlay[:, :, c][mask]).astype(np.uint8)

        # Escribir tipo sobre el centroide
        cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Guardar imagen
    filename = os.path.join(output_folder, f"frame_{frame_idx:04d}_tipos.png")
    cv2.imwrite(filename, overlay)

def mostrar_histograma_areas(tracking_graph):
    areas = []

    for frame_blobs in tracking_graph.values():
        for blob in frame_blobs:
            mask = blob["mask"]
            area = np.sum(mask)
            areas.append(area)

    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=30, color="skyblue", edgecolor="black")
    plt.axvline(x=50, color='gray', linestyle='--', label='min sugerido')
    plt.axvline(x=250, color='gray', linestyle='--', label='max sugerido')
    plt.title("Histograma de áreas de blobs")
    plt.xlabel("Área (número de píxeles)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def resumen_por_tipo(tracking_graph):
    frames = sorted(tracking_graph.keys())
    individuales = []
    grupos = []
    ruidos = []

    for f in frames:
        indiv = sum(1 for blob in tracking_graph[f] if blob["type"] == "individual")
        grup = sum(1 for blob in tracking_graph[f] if blob["type"] == "group")
        ruid = sum(1 for blob in tracking_graph[f] if blob["type"] == "ruido")

        individuales.append(indiv)
        grupos.append(grup)
        ruidos.append(ruid)

    # Plot
    x = np.arange(len(frames))
    plt.figure(figsize=(12, 6))
    plt.bar(x, ruidos, label="Ruido", color="gray")
    plt.bar(x, individuales, bottom=ruidos, label="Individual", color="blue")
    plt.bar(x, grupos, bottom=np.array(ruidos)+np.array(individuales), label="Grupo", color="red")

    plt.xlabel("Frame")
    plt.ylabel("Número de blobs")
    plt.title("Resumen por tipo de blob en cada frame")
    plt.legend()
    plt.tight_layout()
    plt.show()

def exportar_blob_recortado(frame, mask, frame_idx, blob_idx, export_folder, padding=5):
    os.makedirs(export_folder, exist_ok=True)

    if np.sum(mask) == 0:
        return  # máscara vacía

    masked_frame = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))
    filename = f"frame_{frame_idx:04d}_blob_{blob_idx:02d}.png"
    path = os.path.join(export_folder, filename)
    cv2.imwrite(path, masked_frame)
  
# === Inicialización del grafo temporal ===
tracking_graph = defaultdict(list)

# === Estado del tracking ===
prev_centroids = []     # lista de centroides de la última imagen
prev_ids = []           # lista de IDs de la última imagen
next_id = 0             # ID para el siguiente blob
MAX_IDS = 50            # máximo número de IDs a asignar

# === Loop frame a frame ===
for i in range(len(mask_paths) - 1):
    print(f"[INFO] Frame {i} → {i+1}")

    # Cargar máscaras y frame para recortar las partes de la imagen que se corresponden con las mascaras (peces)
    mask1 = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask_paths[i+1], cv2.IMREAD_GRAYSCALE)
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    frame = cv2.imread(img_paths[i])

    # Optical Flow (solo para clustering)
    flow = cv2.calcOpticalFlowFarneback(mask1.astype(np.float32), mask2.astype(np.float32), None, **flow_params)
    yx = np.column_stack(np.where(mask1 > 0))
    vectors = flow[mask1 > 0]
    features = np.hstack([yx, vectors])

    # Clustering dentro de la máscara global
    clustering = DBSCAN(eps=3, min_samples=10).fit(features)
    labels = clustering.labels_
    num_blobs = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"    → Detectados {num_blobs} blobs de movimiento")

    # Procesar cada blob (pez individual o cruce)
    for label in range(num_blobs):
        blob_mask = np.zeros_like(mask1, dtype=np.uint8)
        coords = yx[labels == label]
        blob_mask[coords[:, 0], coords[:, 1]] = 1

        # Centroide
        M = cv2.moments(blob_mask)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)

        # Guardar blob en el grafo temporal
        tracking_graph[i].append({
            "mask": blob_mask,
            "centroid": centroid,
            "type": None,           # aún no sabemos si es individual o grupo
            "assigned_ids": [],     # se asignarán luego
        })
        
        # Clasificar los blobs en individuales o de grupo
        blob_type = clasificar_blob(blob_mask, flow=flow, usar_flow=True)
        tracking_graph[i][-1]["type"] = blob_type  #tracking_graph[i][-1] accede al último blob añadido al grafo en ese frame (el que acabamos de guardar)
        if EXPORT_BLOBS:
            exportar_blob_recortado(frame, blob_mask, i, label, BLOBS_EXPORT_FOLDER)

        
    # Visualizar los blobs en la imagen  
    BLOB_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "blob_type")
    os.makedirs(BLOB_OUTPUT_FOLDER, exist_ok=True)
    visualizar_blobs_coloreados(i, tracking_graph, frame, BLOB_OUTPUT_FOLDER)
        
# Estadisticas finales
resumen_por_tipo(tracking_graph)
mostrar_histograma_areas(tracking_graph)

        
        
                        
                    
                    
                    


