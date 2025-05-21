import cv2
import os
import numpy as np

def cargar_frame_de_video(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None

def cargar_mascara(carpeta_base, tipo, frame_idx):
    nombre_carpeta = {
        "cutie": "masks_cutie",
        "morfologia": "mask_morfologia"
    }.get(tipo.lower())
    if not nombre_carpeta:
        return None

    path_mask = os.path.join(carpeta_base, nombre_carpeta, f"frame_{frame_idx:05d}.jpg")
    if os.path.exists(path_mask):
        return cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)
    return None

def superponer_mascara(frame, mask, color=(0, 255, 0), alpha=0.5):
    if mask is None:
        return frame
    overlay = frame.copy()
    overlay[mask > 0] = (overlay[mask > 0] * (1 - alpha) + alpha * np.array(color)).astype('uint8')
    return overlay

def cargar_bboxes_yolo(carpeta_base, frame_idx):
    label_path = os.path.join(carpeta_base, "bbox_yolo", "labels", f"frame_{frame_idx:05d}.txt")
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x1, y1, x2, y2 = map(float, parts)
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    return bboxes

def dibujar_bboxes(frame, bboxes, color=(0, 0, 255), thickness=2):
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame
