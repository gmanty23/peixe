import os
import cv2
import json
import shutil
from pathlib import Path


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

# ---------------------- Reproyección de BBoxes ----------------------
def reproyectar_bboxes_a_original(lista_bboxes, imgsz, padding_info, recorte_margenes):
    reproyectadas = []

    pad_top = padding_info["top"]
    pad_left = padding_info["left"]
    escala = padding_info["escala"]

    margen_x = recorte_margenes["left"]
    margen_y = recorte_margenes["top"]

    for bbox in lista_bboxes:
        clase, x_rel, y_rel, w_rel, h_rel = bbox

        x_c_abs = x_rel * imgsz
        y_c_abs = y_rel * imgsz
        w_abs = w_rel * imgsz
        h_abs = h_rel * imgsz

        x_c_sin_pad = x_c_abs - pad_left
        y_c_sin_pad = y_c_abs - pad_top

        x_c_orig = x_c_sin_pad / escala
        y_c_orig = y_c_sin_pad / escala
        w_orig = w_abs / escala
        h_orig = h_abs / escala

        x1 = x_c_orig - w_orig / 2 + margen_x
        y1 = y_c_orig - h_orig / 2 + margen_y
        x2 = x_c_orig + w_orig / 2 + margen_x
        y2 = y_c_orig + h_orig / 2 + margen_y

        reproyectadas.append([int(clase), x1, y1, x2, y2])

    return reproyectadas

# ---------------------- Procesar .txt reproyectando ----------------------
def reproyectar_txts_yolo(labels_dir, imgsz, padding_info, recorte_margenes, output_dir, estado=None):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    total = len(txt_files)
    if estado:
        estado.emitir_etapa("Reproyectando coordenadas...")
        estado.emitir_progreso(0)
    for i,archivo in enumerate(txt_files):
        if not archivo.endswith(".txt"):
            continue

        ruta_txt = os.path.join(labels_dir, archivo)
        with open(ruta_txt, "r") as f:
            lineas = f.readlines()

        bboxes = []
        for linea in lineas:
            partes = linea.strip().split()
            if len(partes) != 5:
                continue
            bbox = [int(partes[0])] + list(map(float, partes[1:]))
            bboxes.append(bbox)

        reproyectadas = reproyectar_bboxes_a_original(bboxes, imgsz, padding_info, recorte_margenes)
        if estado:
            porcentaje = int((i + 1) / total * 100)
            estado.emitir_progreso(porcentaje)

        with open(os.path.join(output_dir, archivo), "w") as f_out:
            for bbox in reproyectadas:
                clase, x1, y1, x2, y2 = bbox
                f_out.write(f"{clase} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

# ---------------------- Pipeline principal ----------------------
def procesar_yolo(video_path, output_path, resize_dim, bbox_recorte=None, estado=None):
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

