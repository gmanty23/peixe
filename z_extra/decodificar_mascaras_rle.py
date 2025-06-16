import os
import numpy as np
import cv2
from tqdm import tqdm

# ==== CONFIGURA AQUÍ ====
directorio_entrada = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/masks_rle/"
directorio_salida = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/masks_rle_decode/"
os.makedirs(directorio_salida, exist_ok=True)
# =========================

def decode_rle(shape, rle):
    starts = rle[::2] - 1
    lengths = rle[1::2]
    ends = starts + lengths
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        flat[s:e] = 255  # reconstruir con valores 0 / 255
    return flat.reshape(shape)

def decodificar_rle_npz_a_png(directorio_entrada, directorio_salida):
    os.makedirs(directorio_salida, exist_ok=True)
    archivos = [f for f in os.listdir(directorio_entrada) if f.lower().endswith('.npz')]

    for nombre_archivo in tqdm(archivos, desc="Decodificando RLE"):
        ruta = os.path.join(directorio_entrada, nombre_archivo)
        data = np.load(ruta)
        shape = tuple(data['shape'])
        rle = data['rle']

        mascara = decode_rle(shape, rle)

        nombre_salida = os.path.splitext(nombre_archivo)[0] + ".png"
        ruta_salida = os.path.join(directorio_salida, nombre_salida)
        cv2.imwrite(ruta_salida, mascara)

    print(f"✅ Decodificación completada en: {directorio_salida}")

if __name__ == "__main__":
    decodificar_rle_npz_a_png(directorio_entrada, directorio_salida)
