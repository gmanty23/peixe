import os
import cv2
import json
import time
import shutil
import random
import numpy as np
import multiprocessing
from pathlib import Path

# ---------------------- Señales de estado ----------------------
class EstadoProceso:
    def __init__(self):
        self.on_etapa = None  # callback(str)
        self.on_progreso = None  # callback(int)
        self.on_error = None  # callback(str)
        self.on_total_videos = None  # callback(int)
        self.on_video_progreso = None  # callback(int)

    def emitir_etapa(self, mensaje):
        if self.on_etapa:
            self.on_etapa(mensaje)

    def emitir_progreso(self, porcentaje):
        if self.on_progreso:
            self.on_progreso(porcentaje)

    def emitir_error(self, mensaje):
        if self.on_error:
            self.on_error(mensaje)

    def emitir_total_videos(self, total):
        if self.on_total_videos:
            self.on_total_videos(total)

    def emitir_video_progreso(self, index):
        if self.on_video_progreso:
            self.on_video_progreso(index)

# ---------------------- Utilidades ----------------------
def extraer_frames(video_path, output_folder, resize_enabled=False, resize_dims=(1920, 1080)):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_enabled:
            frame = cv2.resize(frame, resize_dims)
        frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()
    return sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])

# ---------------------- Fondo ----------------------
def calcular_fondo_percentil(input_path, imagenes, percentil, grupo_size, num_procesos, cache_path, estado=None):
    def calcular_mediana_segmento(lotes, id_proc, q):
        for idx, grupo in lotes:
            imgs = [cv2.imread(os.path.join(input_path, img)) for img in grupo]
            imgs = [im for im in imgs if im is not None]
            if imgs:
                arr = np.array(imgs)
                fondo = np.percentile(arr, percentil, axis=0).astype(np.uint8)
                out_path = os.path.join(cache_path, f"mediana_grupo_{idx:03d}.jpg")
                cv2.imwrite(out_path, fondo)
                q.put(1)

    os.makedirs(cache_path, exist_ok=True)
    random.shuffle(imagenes)
    grupos = [imagenes[i:i+grupo_size] for i in range(0, len(imagenes), grupo_size)]
    lotes_por_proc = [[] for _ in range(num_procesos)]
    for i, grupo in enumerate(grupos):
        lotes_por_proc[i % num_procesos].append((i, grupo))

    q = multiprocessing.Queue()
    procesos = [
        multiprocessing.Process(target=calcular_mediana_segmento, args=(lotes_por_proc[i], i, q))
        for i in range(num_procesos)
    ]
    for p in procesos: p.start()

    total = len(grupos)
    progreso = 0
    while progreso < total:
        q.get()
        progreso += 1
        porcentaje = int((progreso / total) * 100)
        if estado:
            estado.emitir_progreso(porcentaje)

    for p in procesos: p.join()

    medianas = [cv2.imread(os.path.join(cache_path, f))
                for f in os.listdir(cache_path) if f.endswith(".jpg")]
    medianas = [m for m in medianas if m is not None]
    while len(medianas) > grupo_size:
        medianas = [
            np.percentile(np.array(medianas[i:i+grupo_size]), percentil, axis=0).astype(np.uint8)
            for i in range(0, len(medianas), grupo_size)
        ]
    fondo_final = medianas[0]
    shutil.rmtree(cache_path)
    return fondo_final

# ---------------------- Procesado ----------------------
def aplicar_pipeline_morfologica(mask, pipeline):
    for paso in pipeline:
        op = paso["op"]
        kx, ky = paso["kernel"]
        kernel = np.ones((kx, ky), np.uint8)
        if op == "apertura":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif op == "cierre":
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif op == "dilatacion":
            mask = cv2.dilate(mask, kernel)
        elif op == "erosion":
            mask = cv2.erode(mask, kernel)
    return mask

def procesar_segmento(imagenes, input_path, output_path, fondo, umbral, pipeline, q):
    try:
        os.makedirs(output_path, exist_ok=True)
        for img_name in imagenes:
            #print(f"[Segmento] Procesando {img_name} en proceso PID {os.getpid()}")
            img = cv2.imread(os.path.join(input_path, img_name))
            if img is None:
                continue
            diff = cv2.absdiff(img, fondo)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)
            mask = aplicar_pipeline_morfologica(mask, pipeline)
            cv2.imwrite(os.path.join(output_path, img_name), mask)
            #print(f"[Segmento] Guardada máscara: {img_name}")
            q.put(1)
        #print(f"[Segmento] Proceso {os.getpid()} terminado")
    except Exception as e:
        print(f"[Segmento] Error en el proceso {os.getpid()}: {e}")

# ---------------------- Flujo principal ----------------------
def procesar_videos_con_morfologia(video_fondo, videos_dir, output_dir,
                                   percentil, grupo_size, umbral, nucleos,
                                   pipeline_ops, estado=None,
                                   usar_imagen_fondo=False, ruta_imagen_fondo=None,
                                   resize_enabled=False, resize_dims=(1920, 1080)):
    try:
        if usar_imagen_fondo and ruta_imagen_fondo:
            fondo = cv2.imread(ruta_imagen_fondo)
            if fondo is None:
                if estado:
                    estado.emitir_error("No se pudo leer la imagen de fondo proporcionada.")
                return
        else:
            if estado:
                estado.emitir_etapa("Calculando fondo...")
                estado.emitir_progreso(0)

            fondo_frames_dir = "__frames_fondo_tmp__"
            fondo_imagenes = extraer_frames(video_fondo, fondo_frames_dir, resize_enabled, resize_dims)
            fondo = calcular_fondo_percentil(
                fondo_frames_dir, fondo_imagenes,
                percentil, grupo_size, nucleos,
                "__cache_fondo__", estado
            )
            shutil.rmtree(fondo_frames_dir)

        if fondo is not None:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "fondo_calculado.jpg"), fondo)
        else:
            if estado:
                estado.emitir_error("No se pudo obtener el fondo (imagen o video).")
            return

        if estado:
            estado.emitir_etapa("Aplicando morfología...")
            estado.emitir_progreso(0)

        pipeline = pipeline_ops
        videos = [f for f in os.listdir(videos_dir) if f.endswith(('.avi', '.mp4'))]
        if estado:
            estado.emitir_total_videos(len(videos))

        for i, video in enumerate(videos):
            if estado:
                estado.emitir_video_progreso(i)

            video_path = os.path.join(videos_dir, video)
            frames_dir = os.path.join("__frames_tmp__", Path(video).stem)
            output_masks = os.path.join(output_dir, f"mascaras_{Path(video).stem}")
            imagenes = extraer_frames(video_path, frames_dir, resize_enabled, resize_dims)
            if not imagenes:
                if estado:
                    estado.emitir_error("No se encontraron imágenes para procesar.")
                return

            segment_size = len(imagenes) // nucleos
            q = multiprocessing.Queue()
            procesos = []
            for j in range(nucleos):
                ini = j * segment_size
                fin = (j + 1) * segment_size if j < nucleos - 1 else len(imagenes)
                segmento = imagenes[ini:fin]
                p = multiprocessing.Process(
                    target=procesar_segmento,
                    args=(segmento, frames_dir, output_masks, fondo, umbral, pipeline, q)
                )
                procesos.append(p)
                p.start()

            progreso = 0
            while progreso < len(imagenes):
                q.get()
                progreso += 1
                porcentaje = int((progreso / len(imagenes)) * 100)
                if estado:
                    estado.emitir_progreso(porcentaje)

            for p in procesos:
                p.join()
            shutil.rmtree(frames_dir)

    except Exception as e:
        if estado:
            estado.emitir_error(f"Error en el procesamiento: {e}")
        else:
            print(f"Error: {e}")
