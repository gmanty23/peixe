import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# === CONFIGURACIÓN ===
image_path = "/home/gmanty/code/pruebas_cluster/blobs/frame_0000_blob_04.png"  
n_peces = 8  # Número de peces que quieres separar

# === CARGAR Y PREPARAR ===
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Crear una máscara binaria: todo lo que no es negro
mask = np.any(img != [0, 0, 0], axis=-1).astype(np.uint8)

# Extraer coordenadas de píxeles y sus colores
yx = np.column_stack(np.where(mask > 0))          # shape: (n, 2) → [y, x]
colors = img_rgb[mask > 0]                         # shape: (n, 3) → [R, G, B]
color_rgb = colors / 255.0                         # normalizar colores
color_scaled = color_rgb * 3.0                     # opcional: aumentar peso del color

# === CREAR FEATURES Y APLICAR K-MEANS ===
features = np.hstack([yx, color_scaled])           # [y, x, R, G, B]
kmeans = KMeans(n_clusters=n_peces, random_state=0).fit(features)
labels = kmeans.labels_

# === VISUALIZAR RESULTADO ===
output_img = np.zeros_like(img_rgb)
cluster_colors = np.random.randint(0, 255, (n_peces, 3))  # colores aleatorios

for idx, label in enumerate(labels):
    y, x = yx[idx]
    output_img[y, x] = cluster_colors[label]

# Mostrar imagen combinada
combined = np.hstack([output_img, img_rgb])
plt.figure(figsize=(14, 6))
plt.imshow(combined)
plt.title(f"K-Means ({n_peces} clusters) | Blob original")
plt.axis("off")
plt.tight_layout()
plt.show()
