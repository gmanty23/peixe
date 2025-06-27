import os
import cv2
import numpy as np
from tqdm import tqdm

# ==== CONFIGURA AQUÍ ====
directorio_entrada = "/home/gmanty/code/calculos_memoria/"
directorio_salida = "/home/gmanty/code/calculos_memoria/USCL2-061045-061545_p1_rle"
os.makedirs(directorio_salida, exist_ok=True)
# =========================

def encode_rle(mask_bin):
    pixels = mask_bin.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes[1::2] - changes[::2]
    rle = np.empty(changes.size, dtype=int)
    rle[::2] = changes[::2] + 1  # 1-based indexing
    rle[1::2] = runs
    return rle

def convertir_mascaras_a_rle_npz(directorio_entrada, directorio_salida):
    os.makedirs(directorio_salida, exist_ok=True)
    archivos = [f for f in os.listdir(directorio_entrada) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for nombre_archivo in tqdm(archivos, desc="Codificando a RLE"):
        ruta = os.path.join(directorio_entrada, nombre_archivo)
        mascara = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if mascara is None:
            print(f"⚠️ No se pudo leer: {ruta}")
            continue

        # Convertir a binaria [0,1]
        _, binaria = cv2.threshold(mascara, 127, 1, cv2.THRESH_BINARY)

        rle = encode_rle(binaria)
        shape = np.array(binaria.shape, dtype=int)

        nombre_salida = os.path.splitext(nombre_archivo)[0] + ".npz"
        ruta_salida = os.path.join(directorio_salida, nombre_salida)
        np.savez_compressed(ruta_salida, shape=shape, rle=rle)

    print(f"✅ Codificación completada en: {directorio_salida}")

if __name__ == "__main__":
    convertir_mascaras_a_rle_npz(directorio_entrada, directorio_salida)
