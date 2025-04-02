import os
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops

# ---------------- CONFIG ------------------
DIR_INPUT = "/home/gms/AnemoNAS/Workspace/masks_guardadas_2"
DIR_OUTPUT_IMG = "/home/gms/AnemoNAS/Workspace/salida_seguimiento/ids_png"
OUTPUT_JSON = "/home/gms/AnemoNAS/Workspace/salida_seguimiento/ids_por_frame.json"
EXT = ".png"
NUM_PECES = 50

# Colores fijos por ID
COLORS = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) for _ in range(NUM_PECES)]

# ------------- UTILS --------------
def binarizar_mascaras_en_directorios(dir_raiz, extension=".png"):
    subdirs = [os.path.join(dir_raiz, d) for d in os.listdir(dir_raiz) if os.path.isdir(os.path.join(dir_raiz, d))]
    for subdir in subdirs:
        print(f"[INFO] Binarizando {subdir}")
        archivos = glob(os.path.join(subdir, f"*{extension}"))
        for archivo in tqdm(archivos):
            img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            binaria = (img > 0).astype("uint8") * 255
            cv2.imwrite(archivo, binaria)

def listar_frames_comunes(directorios, extension=".png"):
    conjuntos = []
    for d in directorios:
        archivos = sorted([f for f in os.listdir(d) if f.lower().endswith(extension.lower())])
        conjuntos.append(set(archivos))
    return sorted(set.intersection(*conjuntos)) if conjuntos else []

def fusionar_masks(nombre_archivo, subdirs):
    masks = []
    for carpeta in subdirs:
        ruta = os.path.join(carpeta, nombre_archivo)
        mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask > 0)
    if not masks:
        return None
    return (np.logical_or.reduce(masks).astype(np.uint8)) * 255

def obtener_centroides(mascara):
    etiquetas = label(mascara)
    regiones = regionprops(etiquetas)
    return etiquetas, [r.centroid for r in regiones], regiones

def calcular_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0

def dibujar_ids(mascara_labels, id_por_label):
    img = np.zeros((*mascara_labels.shape, 3), dtype=np.uint8)
    for label_val, id_list in id_por_label.items():
        color = COLORS[id_list[0] % NUM_PECES] if len(id_list) == 1 else (255, 255, 255)  # blanco para múltiples peces
        img[mascara_labels == label_val] = color
    return img

# ------------- MAIN --------------
def main():
    os.makedirs(DIR_OUTPUT_IMG, exist_ok=True)
    binarizar_mascaras_en_directorios(DIR_INPUT)

    subdirs = [os.path.join(DIR_INPUT, d) for d in os.listdir(DIR_INPUT) if os.path.isdir(os.path.join(DIR_INPUT, d))]
    frames = listar_frames_comunes(subdirs)

    historial_ids = {}
    id_mascara_previas = [None for _ in range(NUM_PECES)]  # lista de máscaras anteriores por pez
    id_centroides_previos = [None for _ in range(NUM_PECES)]

    id_asignados = False
    siguiente_id = 0

    for frame_name in tqdm(frames):
        frame_path = os.path.join(DIR_OUTPUT_IMG, frame_name)
        mask = fusionar_masks(frame_name, subdirs)
        if mask is None:
            continue

        labels, centroides, regiones = obtener_centroides(mask)
        id_por_label = {}

        if not id_asignados:
            # Asignación inicial
            for i, region in enumerate(regiones):
                if siguiente_id < NUM_PECES:
                    id_por_label[i + 1] = [siguiente_id]
                    id_mascara_previas[siguiente_id] = (labels == (i + 1)).astype(np.uint8)
                    id_centroides_previos[siguiente_id] = centroides[i]
                    siguiente_id += 1
            if siguiente_id >= NUM_PECES:
                id_asignados = True

        else:
            asignados = set()
            for i, region in enumerate(regiones):
                blob_mask = (labels == (i + 1)).astype(np.uint8)
                centro = centroides[i]
                candidatos = []
                for pid in range(NUM_PECES):
                    if id_mascara_previas[pid] is None:
                        continue
                    dist = np.linalg.norm(np.array(centro) - np.array(id_centroides_previos[pid]))
                    iou = calcular_iou(blob_mask, id_mascara_previas[pid])
                    score = (1 / (1 + dist)) + iou
                    candidatos.append((score, pid))
                candidatos.sort(reverse=True)
                usados = set()
                id_detectados = []
                for score, pid in candidatos:
                    if pid not in usados and len(id_detectados) < 3:  # permitir que un blob tenga hasta 3 ids
                        id_detectados.append(pid)
                        usados.add(pid)
                id_por_label[i + 1] = id_detectados

            # Actualizamos máscaras anteriores
            for label_val, ids in id_por_label.items():
                for pid in ids:
                    id_mascara_previas[pid] = (labels == label_val).astype(np.uint8)
                    props = regionprops((labels == label_val).astype(np.uint8))
                    if props:
                        id_centroides_previos[pid] = props[0].centroid

        # Guardar imagen coloreada
        imagen_coloreada = dibujar_ids(labels, id_por_label)
        cv2.imwrite(frame_path, cv2.cvtColor(imagen_coloreada, cv2.COLOR_RGB2BGR))

        # Guardar IDs por frame
        historial_ids[frame_name] = id_por_label

    # Guardar JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(historial_ids, f, indent=2)

    print("[HECHO] IDs asignados y guardados.")

if __name__ == "__main__":
    main()