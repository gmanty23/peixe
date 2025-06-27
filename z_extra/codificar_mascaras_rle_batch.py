import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process

# ==== CONFIGURA AQUÍ LA CARPETA DE TRABAJO Y NÚCLEOS ====
carpeta_trabajo = "/home/gms/AnemoNAS/prueba_GUI/"
num_procesos = 3
# =========================================================

def encode_rle(mask_bin):
    pixels = mask_bin.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes[1::2] - changes[::2]
    rle = np.empty(changes.size, dtype=int)
    rle[::2] = changes[::2] + 1  # 1-based indexing
    rle[1::2] = runs
    return rle

def convertir_mascaras_a_rle_npz(carpeta_masks, carpeta_rle):
    os.makedirs(carpeta_rle, exist_ok=True)
    archivos = [f for f in os.listdir(carpeta_masks) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for nombre_archivo in archivos:
        ruta = os.path.join(carpeta_masks, nombre_archivo)
        mascara = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if mascara is None:
            continue

        _, binaria = cv2.threshold(mascara, 127, 1, cv2.THRESH_BINARY)
        rle = encode_rle(binaria)
        shape = np.array(binaria.shape, dtype=int)

        nombre_salida = os.path.splitext(nombre_archivo)[0] + ".npz"
        ruta_salida = os.path.join(carpeta_rle, nombre_salida)
        np.savez_compressed(ruta_salida, shape=shape, rle=rle)

def procesar_subdirectorio(subdir):
    from tqdm import tqdm
    carpeta_masks = os.path.join(subdir, "masks")
    carpeta_rle = os.path.join(subdir, "masks_rle")

    if not os.path.isdir(carpeta_masks):
        return f"⏭️  No hay carpeta 'masks' en {subdir}"

    tqdm.write(f"🔄 Procesando: {subdir}")

    convertir_mascaras_a_rle_npz(carpeta_masks, carpeta_rle)

    try:
        shutil.rmtree(carpeta_masks)
        return f"🗑️  'masks' eliminada en {subdir}"
    except Exception as e:
        return f"❌ Error al eliminar 'masks' en {subdir}: {e}"

def procesar_todos(carpeta_trabajo):
    subdirs = [os.path.join(carpeta_trabajo, d) for d in os.listdir(carpeta_trabajo)
               if os.path.isdir(os.path.join(carpeta_trabajo, d))]

    with Pool(processes=num_procesos) as pool:
        resultados = list(tqdm(pool.imap_unordered(procesar_subdirectorio, subdirs), total=len(subdirs), desc="Procesando carpetas"))

    print("\nResumen:")
    for r in resultados:
        print(r)

if __name__ == "__main__":
    procesar_todos(carpeta_trabajo)
