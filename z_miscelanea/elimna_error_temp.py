# import os
# import numpy as np
# import cv2
# from skimage.measure import label, regionprops
# from scipy.spatial.distance import cdist
# from collections import defaultdict
# from tqdm import tqdm

# # Parámetros
# mask_folder = "/home/gms/AnemoNAS/Workspace/imagenes_diferencias/"
# output_folder = "/home/gms/AnemoNAS/Workspace/prueba_blobs/"
# min_life = 3  # número mínimo de frames que debe vivir un blob
# max_dist = 30  # distancia máxima entre centroides para considerar continuidad

# # Cargar máscaras
# mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.jpg')])
# masks = [cv2.imread(os.path.join(mask_folder, f), cv2.IMREAD_GRAYSCALE) > 0 for f in mask_files]

# # Seguimiento de blobs
# next_id = 1
# track_history = defaultdict(list)  # id -> [(frame_idx, region)]

# active_tracks = dict()  # blob_id -> (centroid, frame_idx)

# for frame_idx, mask in tqdm(enumerate(masks), total=len(masks)):
#     labeled = label(mask)
#     regions = regionprops(labeled)

#     current_centroids = np.array([r.centroid for r in regions])
#     matched_ids = set()

#     # Intentar asociar con blobs activos
#     if len(current_centroids) > 0 and active_tracks:
#         prev_centroids = np.array([v[0] for v in active_tracks.values()])
#         prev_ids = list(active_tracks.keys())
#         distances = cdist(current_centroids, prev_centroids)

#         for i, region in enumerate(regions):
#             j = np.argmin(distances[i])
#             if distances[i][j] < max_dist:
#                 blob_id = prev_ids[j]
#                 track_history[blob_id].append((frame_idx, region))
#                 active_tracks[blob_id] = (region.centroid, frame_idx)
#                 matched_ids.add(blob_id)
#             else:
#                 # Nuevo blob
#                 blob_id = next_id
#                 next_id += 1
#                 track_history[blob_id].append((frame_idx, region))
#                 active_tracks[blob_id] = (region.centroid, frame_idx)

#     else:
#         # Primer frame o sin blobs previos
#         for region in regions:
#             blob_id = next_id
#             next_id += 1
#             track_history[blob_id].append((frame_idx, region))
#             active_tracks[blob_id] = (region.centroid, frame_idx)

#     # Eliminar tracks viejos (si han desaparecido)
#     to_remove = [bid for bid, (_, last_frame) in active_tracks.items() if last_frame < frame_idx - 1]
#     for bid in to_remove:
#         del active_tracks[bid]

# # Crear nuevas máscaras sin blobs "fugaces"
# final_masks = [np.zeros_like(m, dtype=np.uint8) for m in masks]

# for blob_id, history in track_history.items():
#     if len(history) >= min_life:
#         for frame_idx, region in history:
#             coords = region.coords
#             final_masks[frame_idx][coords[:, 0], coords[:, 1]] = 255

# # Guardar
# os.makedirs(output_folder, exist_ok=True)
# for f, out in zip(mask_files, final_masks):
#     cv2.imwrite(os.path.join(output_folder, f), out)



import os
import numpy as np
import cv2
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- Parámetros ---
mask_folder = "/home/gms/AnemoNAS/Workspace/imagenes_diferencias/"
output_folder = "/home/gms/AnemoNAS/Workspace/prueba_blobs/"
min_life = 100       # número mínimo de frames que debe vivir un blob
max_dist = 50        # distancia máxima entre centroides para considerar continuidad
num_threads = 8      # núcleos/hilos para lectura y escritura

# --- Cargar máscaras (paralelizado) ---
mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.jpg')])

def read_mask(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 0

print("📥 Leyendo máscaras...")
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    masks = list(tqdm(executor.map(read_mask, [os.path.join(mask_folder, f) for f in mask_files]),
                      total=len(mask_files)))

# --- Seguimiento de blobs ---
next_id = 1
track_history = defaultdict(list)  # id -> [(frame_idx, region)]
active_tracks = dict()             # blob_id -> (centroid, frame_idx)

print("🧠 Procesando blobs...")
for frame_idx, mask in tqdm(enumerate(masks), total=len(masks)):
    labeled = label(mask)
    regions = regionprops(labeled)
    current_centroids = np.array([r.centroid for r in regions])
    matched_ids = set()

    if len(current_centroids) > 0 and active_tracks:
        prev_centroids = np.array([v[0] for v in active_tracks.values()])
        prev_ids = list(active_tracks.keys())
        distances = cdist(current_centroids, prev_centroids)

        for i, region in enumerate(regions):
            j = np.argmin(distances[i])
            if distances[i][j] < max_dist:
                blob_id = prev_ids[j]
                track_history[blob_id].append((frame_idx, region))
                active_tracks[blob_id] = (region.centroid, frame_idx)
                matched_ids.add(blob_id)
            else:
                blob_id = next_id
                next_id += 1
                track_history[blob_id].append((frame_idx, region))
                active_tracks[blob_id] = (region.centroid, frame_idx)
    else:
        for region in regions:
            blob_id = next_id
            next_id += 1
            track_history[blob_id].append((frame_idx, region))
            active_tracks[blob_id] = (region.centroid, frame_idx)

    # Limpieza de blobs inactivos
    to_remove = [bid for bid, (_, last_frame) in active_tracks.items() if last_frame < frame_idx - 1]
    for bid in to_remove:
        del active_tracks[bid]

# --- Construcción de máscaras finales ---
print("🎨 Generando máscaras finales...")
final_masks = [np.zeros_like(m, dtype=np.uint8) for m in masks]

for blob_id, history in track_history.items():
    if len(history) >= min_life:
        for frame_idx, region in history:
            coords = region.coords
            final_masks[frame_idx][coords[:, 0], coords[:, 1]] = 255

# --- Guardado de resultados (paralelizado) ---
os.makedirs(output_folder, exist_ok=True)

def write_mask(args):
    path, mask = args
    cv2.imwrite(path, mask)

print("💾 Guardando resultados...")
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(write_mask, [(os.path.join(output_folder, f), m)
                               for f, m in zip(mask_files, final_masks)])

print("✅ Proceso completado.")
