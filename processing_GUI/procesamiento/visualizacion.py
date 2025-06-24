import cv2
import os
import numpy as np
import json
from PySide6.QtGui import QImage, QPixmap

def cargar_frame_de_video(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None

# def cargar_mascara(carpeta_base, frame_idx):
#     ruta = os.path.join(carpeta_base, "masks", f"frame_{frame_idx:05d}.jpg")
#     if os.path.exists(ruta):
#         return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
#     return None

def cargar_mascara(carpeta_base, frame_idx):
    ruta = os.path.join(carpeta_base, "masks_rle", f"frame_{frame_idx:05d}.npz")
    if os.path.exists(ruta):
        data = np.load(ruta)
        shape = tuple(data['shape'])
        rle = data['rle']
        return decode_rle(shape, rle)
    return None

def decode_rle(shape, rle):
    starts = rle[::2] - 1
    lengths = rle[1::2]
    ends = starts + lengths
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        flat[s:e] = 255
    return flat.reshape(shape)


# # Cargar recorte desde archivos JSON, si existen
def cargar_recorte(carpeta_base):
    json_path_mask = os.path.join(carpeta_base, "masks_rle", "recorte_morphology.json")
    json_path_bbox = os.path.join(carpeta_base, "bbox", "recorte_yolo.json")
    print(f"Buscando recortes en: {json_path_mask} y {json_path_bbox}")
    recorte_mask = None
    recorte_bbox = None
    if os.path.exists(json_path_mask):
        with open(json_path_mask, 'r') as f:
            recorte_mask = json.load(f)
    if os.path.exists(json_path_bbox):
        with open(json_path_bbox, 'r') as f:
            recorte_bbox = json.load(f)
    return recorte_mask, recorte_bbox
    

def aplicar_recorte(frame, recorte):
    if not recorte:
        return frame
    if isinstance(recorte, dict):
        x, y, w, h = recorte["x"], recorte["y"], recorte["w"], recorte["h"]
    elif isinstance(recorte, (tuple, list)) and len(recorte) == 4:
        x, y, w, h = recorte
    else:
        raise ValueError(f"Formato de recorte no soportado: {recorte}")
    return frame[y:y+h, x:x+w]

def superponer_mascara(frame, mask, alpha=0.5, color=(0, 255, 0)):
    if mask.shape[:2] != frame.shape[:2]:
        print("[AVISO] Dimensiones de la máscara no coinciden con el frame. Se omite superposición.")
        print(f"Frame shape: {frame.shape}, Mask shape: {mask.shape}")
        return frame
    overlay = frame.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + alpha * np.array(color)).astype('uint8')
    return overlay

def cargar_bboxes_yolo(carpeta_base, frame_idx):
    label_path = os.path.join(carpeta_base, "bbox", "labels", f"frame_{frame_idx:05d}.txt")
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x1, y1, x2, y2 = map(float, parts)
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    return bboxes

def dibujar_bboxes(frame, bboxes, color=(0, 255, 0), thickness=2):
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def cargar_output_dims(carpeta_subdir):
    json_path = os.path.join(carpeta_subdir, "output_dims.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            return tuple(data.get("output_dims", [1920, 1080]))
    return (1920, 1080)
