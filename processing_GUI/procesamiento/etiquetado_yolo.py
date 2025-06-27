import os
import cv2
import json
import shutil
from pathlib import Path
import numpy as np


# ---------------------- Señales de estado ----------------------
class EstadoProceso:
    def __init__(self):
        self.on_etapa = None  # callback(str)
        self.on_progreso = None  # callback(int)
        self.on_error = None  # callback(str)

    def emitir_etapa(self, mensaje):
        if self.on_etapa:
            self.on_etapa(mensaje)

    def emitir_progreso(self, porcentaje):
        if self.on_progreso:
            self.on_progreso(porcentaje)

    def emitir_error(self, mensaje):
        if self.on_error:
            self.on_error(mensaje)

# ---------------------- Utilidades ----------------------
def extraer_y_preprocesar_frames_yolo(video_path, output_folder, resize, bbox_recorte=None, estado=None):
    try:
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        # Recuperar tamaño original del frame
        ret, frame_original = cap.read()
        if not ret:
            raise Exception("No se pudo leer el primer frame del vídeo.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar lectura
        h_original, w_original = frame_original.shape[:2]

        if bbox_recorte:
            x, y, w, h = bbox_recorte
            recorte_margenes = {
                "left": x,
                "top": y,
                "right": w_original - (x + w),
                "bottom": h_original - (y + h)
            }
        else:
            recorte_margenes = {"left": 0, "top": 0, "right": 0, "bottom": 0}

        padding_info = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if bbox_recorte:
                x, y, w, h = bbox_recorte
                frame = frame[y:y+h, x:x+w]

            frame, padding_info = resize_con_padding(frame, resize, return_padding=True)

            frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1

            if estado:
                porcentaje = int((frame_idx / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        cap.release()
        return output_folder, recorte_margenes, padding_info

    except Exception as e:
        if estado:
            estado.emitir_error(f"Error al preprocesar los frames: {e}")
        else:
            print(f"Error: {e}")
        return None, None, None

    

# ---------------------- Padding y resize ----------------------
def resize_con_padding(imagen, tamaño_objetivo, return_padding=False):
    h, w = imagen.shape[:2]
    escala = tamaño_objetivo / max(h, w)
    new_w, new_h = int(w * escala), int(h * escala)
    imagen_redimensionada = cv2.resize(imagen, (new_w, new_h))

    delta_w = tamaño_objetivo - new_w
    delta_h = tamaño_objetivo - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    imagen_padded = cv2.copyMakeBorder(imagen_redimensionada, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    if return_padding:
        return imagen_padded, {"top": top, "bottom": bottom, "left": left, "right": right, "escala": escala}
    else:
        return imagen_padded


# ---------------------- Elimiar bbox de zona a ignorar ----------------------
def limpiar_bboxes_txt_con_mascara(bbox_path, estado=None):
    # Cargar máscara
    ruta_mascara = "processing_GUI/procesamiento/zona_no_valida.png"
    mascara_ignorar = cv2.imread(ruta_mascara, 0)
    if mascara_ignorar is None:
        if estado:
            estado.emitir_error(f"❌ Error: No se pudo cargar la máscara: {ruta_mascara}")
        else:
            print(f"❌ Error: No se pudo cargar la máscara: {ruta_mascara}")
        return
    
    mascara_binaria = (mascara_ignorar > 0).astype(np.uint8)
    altura_masc, ancho_masc = mascara_binaria.shape

    # Localiza todos los archivos txt de bbox/labels
    txt_paths = [f for f in Path(bbox_path).rglob("*.txt")]
    total = len(txt_paths)
    print(f"[INFO] Encontrados {total} archivos de etiquetas para procesar en {bbox_path}.")

    if estado:
        estado.emitir_etapa("Eliminando BBoxes en zona ignorada...")
        estado.emitir_progreso(0)

    for idx, txt_path in enumerate(txt_paths):
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
                    print(f"[INFO] BBox eliminada en {txt_path.name} (centroide en zona ignorada).")
            else:
                    print(f"[INFO] BBox fuera de límites en {txt_path.name}. Eliminada por seguridad.")

        with open(txt_path, "w") as f:
            f.writelines(nuevas_lineas)

        # Emitir progreso
        if estado:
            porcentaje = int((idx + 1) / total * 100)
            estado.emitir_progreso(porcentaje)

    if estado:
        estado.emitir_etapa("Eliminación completada.")


# ---------------------- Pipeline principal ----------------------
def procesar_yolo(video_path, output_path, resize_dim, bbox_recorte=None, estado=None, output_dims=(1920,1080)):
    try:
        estado.emitir_etapa("Preprocesando imágenes...")
        estado.emitir_progreso(0)

        carpeta_cache = os.path.join("processing_GUI/procesamiento/cache", "input_yolo")
        if os.path.exists(carpeta_cache):
            shutil.rmtree(carpeta_cache)

        imagenes_path, recorte_margenes, padding_info = extraer_y_preprocesar_frames_yolo(
            video_path,
            carpeta_cache,
            resize_dim,
            bbox_recorte,
            estado
        )

        if imagenes_path is None:
            return None, None, None

        if estado:
            estado.emitir_progreso(100)
            estado.emitir_etapa("Listo para ejecutar YOLO")

        return imagenes_path, recorte_margenes, padding_info

    except Exception as e:
        if estado:
            estado.emitir_error(f"Error en el procesamiento YOLO: {e}")
        else:
            print(f"Error: {e}")
        return None, None, None

