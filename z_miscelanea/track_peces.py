import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# ---------- CONFIGURACIÓN ----------
EXT = ".png" # extensión de las máscaras
DIR_RAIZ = "/home/gms/AnemoNAS/Workspace/masks_guardadas_2/" 
NUM_PROCESOS = max(cpu_count() - 1, 1)
VISUALIZAR = False  # Muestra los resultados por frame
GUARDAR = True  # Guarda imágenes etiquetadas
DIR_SALIDA = "/home/gms/AnemoNAS/Workspace/salida_etiquetada/"
# -----------------------------------

# def debug_directorios(directorios):
#     print("\n[DEBUG] Subdirectorios detectados:")
#     for d in directorios:
#         print(f"  - {d}")
#         if not os.path.isdir(d):
#             print(f"    ❌ No es un directorio")
#         else:
#             archivos = os.listdir(d)
#             print(f"    Contiene {len(archivos)} archivos. Ejemplos:")
#             print(f"    {archivos[:5]}")

#     print("-" * 50)
def binarizar_mascaras_en_directorios(dir_raiz, extension=".png"):
    subdirs = [os.path.join(dir_raiz, d) for d in os.listdir(dir_raiz) if os.path.isdir(os.path.join(dir_raiz, d))]

    for subdir in subdirs:
        print(f"[INFO] Procesando {subdir}")
        archivos = glob(os.path.join(subdir, f"*{extension}"))
        for archivo in tqdm(archivos):
            img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[AVISO] No se pudo leer {archivo}")
                continue

            # Binariza: valores > 0 se convierten en 255
            binaria = (img > 0).astype("uint8") * 255
            cv2.imwrite(archivo, binaria)

    print("[HECHO] Preprocesamiento completo.")
    
def listar_frames_comunes(directorios, extension=".png"):
    # Tomamos los nombres de los archivos comunes en todos los subdirectorios
    conjuntos = []
    for d in directorios:
        archivos = sorted([
            f for f in os.listdir(d)
            if f.lower().endswith(extension.lower())
        ])
        conjuntos.append(set(archivos))

    # Intersección de todos los conjuntos de nombres de archivo
    frames_comunes = sorted(set.intersection(*conjuntos)) if conjuntos else []
    return frames_comunes

def fusionar_masks(nombre_archivo, subdirs):
    masks = []
    for carpeta in subdirs:
        ruta = os.path.join(carpeta, nombre_archivo)
        mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask > 0)
    if not masks:
        return None
    return np.logical_or.reduce(masks).astype(np.uint8) * 255

def etiquetar_mascara(mascara_binaria):
    num_labels, labels = cv2.connectedComponents(mascara_binaria)
    return labels, num_labels

def colorear_labels(labels):
    num_labels = labels.max() + 1
    color_map = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    output_color = color_map[labels]
    return output_color

def procesar_frame(nombre_archivo, subdirs):
    mascara_combinada = fusionar_masks(nombre_archivo, subdirs)
    if mascara_combinada is None:
        return

    labels, num_labels = etiquetar_mascara(mascara_combinada)
    imagen_color = colorear_labels(labels)

    if VISUALIZAR:
        plt.figure(figsize=(6, 6))
        plt.imshow(imagen_color)
        plt.title(f"{nombre_archivo} - {num_labels - 1} blobs")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    if GUARDAR:
        os.makedirs(DIR_SALIDA, exist_ok=True)
        salida_path = os.path.join(DIR_SALIDA, nombre_archivo)
        cv2.imwrite(salida_path, cv2.cvtColor(imagen_color, cv2.COLOR_RGB2BGR))

def main():
    binarizar_mascaras_en_directorios(DIR_RAIZ, extension=".png")
    subdirs = [os.path.join(DIR_RAIZ, d) for d in os.listdir(DIR_RAIZ) if os.path.isdir(os.path.join(DIR_RAIZ, d))]
    # debug_directorios(subdirs)

    frames = listar_frames_comunes(subdirs)

    print(f"[INFO] Procesando {len(frames)} frames con {NUM_PROCESOS} procesos...")

    with Pool(NUM_PROCESOS) as pool:
        pool.map(partial(procesar_frame, subdirs=subdirs), frames)

    print("[HECHO] Procesamiento completado.")

if __name__ == "__main__":
    main()
