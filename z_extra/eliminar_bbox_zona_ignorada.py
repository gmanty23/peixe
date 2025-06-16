import os
import cv2
import numpy as np
from tqdm import tqdm

def limpiar_bboxes_txt_con_mascara(carpeta_trabajo, ruta_mascara):
    # Cargar máscara
    mascara_ignorar = cv2.imread(ruta_mascara, 0)
    if mascara_ignorar is None:
        print(f"❌ Error: No se pudo cargar la máscara: {ruta_mascara}")
        return
    
    mascara_binaria = (mascara_ignorar > 0).astype(np.uint8)
    altura_masc, ancho_masc = mascara_binaria.shape

    # Localiza todos los archivos txt de bbox/labels
    txt_paths = []
    for root, dirs, files in os.walk(carpeta_trabajo):
        if "bbox/labels" in root.replace("\\", "/"):
            txt_files = [f for f in files if f.endswith(".txt")]
            for txt_file in txt_files:
                txt_paths.append(os.path.join(root, txt_file))

    # Progreso sobre archivos
    for txt_path in tqdm(txt_paths, desc="Procesando archivos bbox"):
        with open(txt_path, "r") as f:
            lines = f.readlines()

        nuevas_lineas = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x1, y1, x2, y2 = map(int, parts)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if 0 <= cx < ancho_masc and 0 <= cy < altura_masc:
                if mascara_binaria[cy, cx] == 0:
                    nuevas_lineas.append(line)
                else:
                    tqdm.write(f"[INFO] BBox eliminada en {os.path.basename(txt_path)} (centroide en zona ignorada).")
            else:
                tqdm.write(f"[INFO] BBox fuera de límites en {os.path.basename(txt_path)}. Eliminada por seguridad.")

        with open(txt_path, "w") as f:
            f.writelines(nuevas_lineas)

        #tqdm.write(f"✅ {os.path.basename(txt_path)}: {len(nuevas_lineas)} bboxes conservadas.")

if __name__ == "__main__":
    carpeta_trabajo = "/home/gms/AnemoNAS/POR_DIA/08-12-2023/" 
    ruta_mascara = "processing_GUI/procesamiento/zona_no_valida.png"
    limpiar_bboxes_txt_con_mascara(carpeta_trabajo, ruta_mascara)
