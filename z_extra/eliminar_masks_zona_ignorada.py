import os
import numpy as np
import cv2
from tqdm import tqdm

def decode_rle(shape, rle):
    starts = rle[::2] - 1
    lengths = rle[1::2]
    ends = starts + lengths
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for s, e in zip(starts, ends):
        flat[s:e] = 1
    return flat.reshape(shape)

def encode_rle(mask_bin):
    pixels = mask_bin.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes[1::2] - changes[::2]
    rle = np.empty(changes.size, dtype=int)
    rle[::2] = changes[::2] + 1  # 1-based indexing
    rle[1::2] = runs
    return rle

def limpiar_rle_con_mascara(carpeta_trabajo, ruta_mascara):
    # Cargar máscara
    mascara_ignorar = cv2.imread(ruta_mascara, 0)
    if mascara_ignorar is None:
        print(f"❌ Error: No se pudo cargar la máscara: {ruta_mascara}")
        return
    
    binaria_ignorar = (mascara_ignorar > 0).astype(np.uint8)
    altura_masc, ancho_masc = binaria_ignorar.shape

    # Recorre subdirectorios
    npz_paths = []
    for root, dirs, files in os.walk(carpeta_trabajo):
        if "masks_rle" in root.replace("\\", "/"):
            npz_files = [f for f in files if f.endswith(".npz")]
            for npz_file in npz_files:
                npz_paths.append(os.path.join(root, npz_file))

    for npz_path in tqdm(npz_paths, desc="Procesando masks_rle"):
        data = np.load(npz_path)
        shape = tuple(data['shape'])
        rle = data['rle']

        mask = decode_rle(shape, rle)

        # Redimensiona la máscara de ignorar si es necesario
        if (shape[1], shape[0]) != (ancho_masc, altura_masc):
            zona_resized = cv2.resize(binaria_ignorar, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            zona_resized = binaria_ignorar

        # Elimina la zona ignorada
        mask_before = mask.copy()
        mask = mask * (zona_resized == 0).astype(np.uint8)

        #if np.any(mask_before != mask):
            #tqdm.write(f"🟠 Se eliminaron píxeles en: {os.path.basename(npz_path)}")

        # Re-codifica
        new_rle = encode_rle(mask)

        # Guarda sobrescribiendo
        np.savez_compressed(npz_path, shape=np.array(shape, dtype=int), rle=new_rle)

        #tqdm.write(f"✅ Limpieza aplicada: {os.path.basename(npz_path)}")

if __name__ == "__main__":
    carpeta_trabajo = "/home/gmanty/code/AnemoNAS/07-12-23/0926-1752" 
    ruta_mascara = "processing_GUI/procesamiento/zona_no_valida.png"
    limpiar_rle_con_mascara(carpeta_trabajo, ruta_mascara)
