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
def extraer_frames(video_path, output_folder, resize_enabled=False, resize_dims=(1920, 1080), bbox_recorte=None):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if bbox_recorte:
            x, y, w, h = map(int, bbox_recorte)
            frame = frame[y:y+h, x:x+w]
        if resize_enabled:
            frame = cv2.resize(frame, resize_dims)
        frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()
    return sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])

def escalar_bbox(bbox, tamaño_origen, tamaño_destino):
    """Escala bbox (x, y, w, h) desde tamaño_origen a tamaño_destino"""
    x, y, w, h = bbox
    alto_og, ancho_og = tamaño_origen
    alto_dst, ancho_dst = tamaño_destino

    scale_x = ancho_dst / ancho_og
    scale_y = alto_dst / alto_og

    x_nuevo = int(x * scale_x)
    y_nuevo = int(y * scale_y)
    w_nuevo = int(w * scale_x)
    h_nuevo = int(h * scale_y)

    return x_nuevo, y_nuevo, w_nuevo, h_nuevo

def reproyectar_mascara(mask_redimensionada, bbox, tamaño_original):
    """
    Inserta una máscara redimensionada en su posición original dentro de un canvas negro del tamaño original.

    Args:
        mask_redimensionada: np.array con shape (h_redim, w_redim)
        bbox: (x, y, w, h) original donde fue recortada
        tamaño_original: (alto, ancho) de la imagen original

    Returns:
        mask_final del tamaño original, con la máscara redimensionada colocada correctamente.
    """
    x, y, w, h = bbox
    alto_ori, ancho_ori = tamaño_original

    # Redimensionar la máscara al tamaño original del recorte (antes del resize)
    mask_escalada = cv2.resize(mask_redimensionada, (w, h), interpolation=cv2.INTER_NEAREST)

    # Crear lienzo negro
    mask_final = np.zeros((alto_ori, ancho_ori), dtype=np.uint8)

    # Insertar la máscara escalada en su lugar
    mask_final[y:y+h, x:x+w] = mask_escalada

    return mask_final


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

def procesar_segmento(imagenes, input_path, output_path, fondo, pipeline, q , bbox_recorte=None, tamaño_original=None):
    try:
        os.makedirs(output_path, exist_ok=True)
        for img_name in imagenes:
            #print(f"[Segmento] Procesando {img_name} en proceso PID {os.getpid()}")
            img = cv2.imread(os.path.join(input_path, img_name))
            if img is None:
                continue
            
            diff = cv2.absdiff(img, fondo)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Paso de umbralización como parte del pipeline
            if pipeline and pipeline[0]["op"] == "umbral":
                valor = pipeline[0].get("valor", 15)
                _, mask = cv2.threshold(gray, valor, 255, cv2.THRESH_BINARY)
                pipeline_proc = pipeline[1:]  # el resto del pipeline
            elif pipeline and pipeline[0]["op"] == "adaptativo":
                block_size = pipeline[0].get("block_size", 11)
                C = pipeline[0].get("C", 2)
                method = pipeline[0].get("method", "mean")
                adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C

                mask = cv2.adaptiveThreshold(gray, 255,
                                            adaptive_method,
                                            cv2.THRESH_BINARY,
                                            block_size, C)
                pipeline_proc = pipeline[1:]
            else:
                raise ValueError("El primer paso del pipeline debe ser 'umbral' o 'adaptativo'")

            # Resto del pipeline
            mask = aplicar_pipeline_morfologica(mask, pipeline_proc)

            if bbox_recorte and tamaño_original:
                mask = reproyectar_mascara(mask, bbox_recorte, tamaño_original)
            
            cv2.imwrite(os.path.join(output_path, img_name), mask)
            #print(f"[Segmento] Guardada máscara: {img_name}")
            q.put(1)
        #print(f"[Segmento] Proceso {os.getpid()} terminado")
    except Exception as e:
        print(f"[Segmento] Error en el proceso {os.getpid()}: {e}")

# ---------------------- Flujo principal ----------------------
def procesar_videos_con_morfologia(video_fondo, videos_dir, output_dir,
                                   percentil, grupo_size, nucleos,
                                   pipeline_ops, estado=None,
                                   usar_imagen_fondo=False, ruta_imagen_fondo=None,
                                   resize_enabled=False, resize_dims=(1920, 1080), bbox_recorte=None):
    try:
        # Obtener dimensiones reales del primer vídeo de videos_dir
        primer_video = next((f for f in os.listdir(videos_dir) if f.endswith(('.avi', '.mp4'))), None)
        if primer_video:
            video_path = os.path.join(videos_dir, primer_video)
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                alto_og, ancho_og = frame.shape[:2]

        if usar_imagen_fondo and ruta_imagen_fondo:
            fondo = cv2.imread(ruta_imagen_fondo)
            if fondo is None:
                if estado:
                    estado.emitir_error("No se pudo leer la imagen de fondo proporcionada.")
                return
            # Redimensionar el fondo si no coincide con el tamaño del vídeo
            alto_fondo, ancho_fondo = fondo.shape[:2]
            if (alto_fondo, ancho_fondo) != (alto_og, ancho_og):
                print(f"[DEBUG] Redimensionando fondo de ({ancho_fondo}, {alto_fondo}) a ({ancho_og}, {alto_og})")
                fondo = cv2.resize(fondo, (ancho_og, alto_og))
            if bbox_recorte:
                h_img, w_img = fondo.shape[:2]
                bbox_escalada = escalar_bbox(bbox_recorte, tamaño_origen=(alto_og, ancho_og), tamaño_destino=(h_img, w_img))
                x, y, w, h = bbox_escalada
                print(f"[DEBUG] bbox usada: x={x}, y={y}, w={w}, h={h}")
                fondo = fondo[y:y+h, x:x+w]
                print(f"[DEBUG] Fondo shape tras recorte: {fondo.shape}")
        else:
            if estado:
                estado.emitir_etapa("Calculando fondo...")
                estado.emitir_progreso(0)

            fondo_frames_dir = "__frames_fondo_tmp__"
            fondo_imagenes = extraer_frames(video_fondo, fondo_frames_dir, resize_enabled, resize_dims, bbox_recorte)
            fondo = calcular_fondo_percentil(
                fondo_frames_dir, fondo_imagenes,
                percentil, grupo_size, nucleos,
                "__cache_fondo__", estado
            )
            shutil.rmtree(fondo_frames_dir)

        if fondo is not None:
            print(f"[DEBUG] Fondo shape antes del recorte: {fondo.shape}")
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "fondo_calculado.jpg"), fondo)
            # os.makedirs(output_dir, exist_ok=True)
            # cv2.imwrite(os.path.join(output_dir, "fondo_calculado.jpg"), fondo)
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
            carpeta_video = os.path.join(output_dir, Path(video).stem)
            os.makedirs(carpeta_video, exist_ok=True)
            output_masks = os.path.join(carpeta_video, "mask_morfologia")
            if bbox_recorte:
                os.makedirs(output_masks, exist_ok=True)
                h_img, w_img = fondo.shape[:2]
                x, y, w, h = bbox_recorte
                recorte_info = {
                    "left": x,
                    "top": y,
                    "right": w_img - (x + w),
                    "bottom": h_img - (y + h)
                }
                with open(os.path.join(output_masks, "recorte_morphology.json"), "w") as f:
                    json.dump(recorte_info, f, indent=4)

            imagenes = extraer_frames(video_path, frames_dir, resize_enabled, resize_dims, bbox_recorte)
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
                    args=(segmento, frames_dir, output_masks, fondo, pipeline, q, bbox_recorte, (alto_og, ancho_og)) 
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

