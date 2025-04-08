import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
input_path = "/home/gms/AnemoNAS/Workspace/or_masks/fondo_at_0000.png"  # Cambia esto si es necesario
min_area = 5  # Para ignorar blobs muy pequeños
blob_id_to_test = 4  # Blob al que aplicar watershed + kmeans

# --- Cargar y binarizar imagen ---
mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# --- Obtener blobs ---
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Seleccionar un blob concreto por ID
x, y, w, h, area = stats[blob_id_to_test]
if area < min_area:
    raise ValueError("El blob seleccionado es demasiado pequeño.")
blob_mask = (labels[y:y+h, x:x+w] == blob_id_to_test).astype(np.uint8) * 255

### ---------------------------
###     1. WATERSHED
### ---------------------------
blob_color = cv2.cvtColor(blob_mask, cv2.COLOR_GRAY2BGR)
dist_transform = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
sure_bg = cv2.dilate(blob_mask, None, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
ws_markers = cv2.watershed(blob_color.copy(), markers)

# Convertir resultado a máscara
watershed_result = np.zeros_like(blob_mask)
watershed_result[ws_markers == 2] = 128
watershed_result[ws_markers == 3] = 255

### ---------------------------
###     2. K-MEANS
### ---------------------------
coords = np.column_stack(np.where(blob_mask > 0))
kmeans = KMeans(n_clusters=2, random_state=0).fit(coords)
labels_kmeans = kmeans.labels_

cluster_mask_1 = np.zeros_like(blob_mask)
cluster_mask_2 = np.zeros_like(blob_mask)
for idx, (y0, x0) in enumerate(coords):
    if labels_kmeans[idx] == 0:
        cluster_mask_1[y0, x0] = 255
    else:
        cluster_mask_2[y0, x0] = 255

### ---------------------------
###     VISUALIZACIÓN
### ---------------------------
fig, axs = plt.subplots(1, 4, figsize=(18, 5))
axs[0].imshow(blob_mask, cmap="gray")
axs[0].set_title(f"Blob original #{blob_id_to_test}")
axs[1].imshow(watershed_result, cmap="gray")
axs[1].set_title("Watershed")
axs[2].imshow(cluster_mask_1, cmap="gray")
axs[2].imshow(cluster_mask_2, cmap="jet", alpha=0.5)
axs[2].set_title("K-means (K=2)")
axs[3].imshow(dist_transform, cmap="magma")
axs[3].set_title("Distance Transform")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()
