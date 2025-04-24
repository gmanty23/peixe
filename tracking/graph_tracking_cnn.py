import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict
from tracking.classification.model import BlobCNN 

# === Configuración del usuario ===
ROOT_FOLDER = "/home/gmanty/code/Workspace_prueba"
MASK_FOLDER = os.path.join(ROOT_FOLDER, "or_masks_50")
IMG_FOLDER = os.path.join(ROOT_FOLDER, "imagenes_og_re_50")
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "outputs_tracking_cnn")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
CHECKPOINT_PATH = "tracking/classification/model/modelo_estable.pt" 

# === Parámetros del modelo y normalización ===
INPUT_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["individual", "group", "ruido"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === Cargar el modelo entrenado ===
model = BlobCNN(num_classes=3).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
# model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

def clasificar_blob_con_modelo(frame, mask, input_size=256):
    """Recorta una ventana centrada en el blob (como en el dataset) y clasifica con el modelo."""

    if np.sum(mask) == 0:
        return "ruido"

    # Calcular centroide
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return "ruido"
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    h, w = frame.shape[:2]
    half = input_size // 2
    left = max(cx - half, 0)
    top = max(cy - half, 0)
    right = min(left + input_size, w)
    bottom = min(top + input_size, h)

    # Corregir si el recorte no es exactamente 256×256 (bordes)
    if right - left < input_size:
        left = max(right - input_size, 0)
    if bottom - top < input_size:
        top = max(bottom - input_size, 0)

    crop = frame[top:bottom, left:right]
    crop_mask = mask[top:bottom, left:right]

    # Aplicar la máscara sobre el recorte
    masked_crop = cv2.bitwise_and(crop, crop, mask=crop_mask.astype(np.uint8))

    # Preprocesado como en el dataset
    img_tensor = transform(masked_crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return CLASSES[pred]

def visualizar_blobs_coloreados(frame_idx, blobs, frame_img, output_folder):
    overlay = frame_img.copy()

    for blob in blobs:
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

        for c in range(3):
            overlay[:, :, c][mask] = (0.6 * color[c] + 0.4 * overlay[:, :, c][mask]).astype(np.uint8)

        cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"frame_{frame_idx:04d}_tipos.png")
    cv2.imwrite(filename, overlay)

# === Procesamiento principal ===

mask_paths = sorted(glob.glob(os.path.join(MASK_FOLDER, "*.png")))
img_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
assert len(mask_paths) == len(img_paths)

tracking_graph = defaultdict(list)
output_vis_folder = os.path.join(OUTPUT_FOLDER, "blob_type_nn")
os.makedirs(output_vis_folder, exist_ok=True)

for i in range(len(mask_paths)):
    print(f"[INFO] Frame {i}")

    mask_img = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
    mask = (mask_img > 0).astype(np.uint8)
    frame = cv2.imread(img_paths[i])

    # Extraer blobs con connected components
    num_labels, labels = cv2.connectedComponents(mask)

    for label in range(1, num_labels):
        blob_mask = (labels == label).astype(np.uint8)

        # Calcular centroide
        M = cv2.moments(blob_mask)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)

        # Clasificar el blob usando la red neuronal
        tipo = clasificar_blob_con_modelo(frame, blob_mask)

        tracking_graph[i].append({
            "mask": blob_mask,
            "centroid": centroid,
            "type": tipo
        })

    # Visualización
    visualizar_blobs_coloreados(i, tracking_graph[i], frame, output_vis_folder)

print("✅ Clasificación y visualización completadas con la red neuronal.")
