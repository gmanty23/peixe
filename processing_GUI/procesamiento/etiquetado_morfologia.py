import os
import cv2
import json
import time
import shutil
import random
import numpy as np
import multiprocessing
from pathlib import Path
from multiprocessing import Pool
import queue

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
# def extraer_frames_segmento(video_path, output_folder, frame_inicial, frame_final, segment_id, resize_enabled=False, resize_dims=(1920, 1080), bbox_recorte=None, q=None, estado=None):
#     try:
#         print(f"[DEBUG] Procesando segmento {segment_id}: frames {frame_inicial} a {frame_final}")
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"No se pudo abrir el vídeo: {video_path}")
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)
#         frame_count = frame_inicial
#         total_frames = frame_final - frame_inicial + 1

#         while frame_count <= frame_final:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             if bbox_recorte:
#                 x, y, w, h = map(int, bbox_recorte)
#                 frame = frame[y:y+h, x:x+w]
            
#             if resize_enabled:
#                 frame = cv2.resize(frame, resize_dims)

#             frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
#             cv2.imwrite(frame_path, frame)
#             frame_count += 1

#             progreso = int(((frame_count - frame_inicial) / total_frames) * 100)
#             q.put((segment_id, progreso))

#         cap.release()
#         print(f"[DEBUG] Segmento {segment_id} procesado: {frame_count - frame_inicial} frames extraídos.")

        
#     except Exception as e:
#         print(f"[ERROR] Error al procesar segmento: {e}")



# def extraer_frames(video_path, output_folder, num_procesos, resize_enabled=False, resize_dims=(1920, 1080), bbox_recorte=None, estado=None):
#     try:
#         os.makedirs(output_folder, exist_ok=True)
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"No se pudo abrir el vídeo: {video_path}")
#             return None
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         print(f"[DEBUG] Total de frames en el vídeo: {total_frames}")
#         cap.release()

#         q = multiprocessing.Queue()
#         segment_size = total_frames // num_procesos
#         procesos = []
#         for i in range(num_procesos):
#             frame_inicial = i * segment_size
#             frame_final = (i + 1) * segment_size - 1 if i < num_procesos - 1 else total_frames -1
#             p = multiprocessing.Process(
#                 target=extraer_frames_segmento,
#                 args=(video_path, output_folder, frame_inicial, frame_final, i, resize_enabled, resize_dims, bbox_recorte, q, estado)
#             )
#             procesos.append(p)
#             p.start()

#         progreso = [0] * num_procesos
#         while any(p.is_alive() for p in procesos):
#             while not q.empty():
#                 segment_id, frames_procesados = q.get()
#                 progreso[segment_id] = frames_procesados
#                 progreso_promedio = int (sum(progreso) // num_procesos)
#                 if estado:
#                     estado.emitir_progreso(progreso_promedio)
#             while not q.empty():
#                 q.get()

#         for p in procesos:
#             p.join()

#         print(f"[DEBUG] Todos los procesos han terminado.")
#         return sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])

#     except Exception as e:
#         print (f"[ERROR] Error al extraer frames: {e}")

def extraer_frames(video_path, output_folder, resize_enabled=False, resize_dims=(1920, 1080), bbox_recorte=None, estado=None):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

        if estado and total_frames > 0:
            porcentaje = int((frame_idx / total_frames) * 100)
            estado.emitir_progreso(porcentaje)

    cap.release()
    return sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])

# ---------------------- Fondo ----------------------
def calcular_mediana_segmento(input_path, percentil, cache_path, lotes, id_proc, q):
    try:
        for idx, grupo in lotes:
            #print(f"[Proceso {id_proc}] Procesando grupo {idx} con {len(grupo)} imágenes")
            imgs = [cv2.imread(os.path.join(input_path, img)) for img in grupo]
            imgs = [im for im in imgs if im is not None]
            if imgs:
                arr = np.array(imgs)
                fondo = np.percentile(arr, percentil, axis=0).astype(np.uint8)
                os.makedirs(cache_path, exist_ok=True)
                out_path = os.path.join(cache_path, f"mediana_grupo_{idx:03d}.jpg")
                cv2.imwrite(out_path, fondo)
            for img in grupo:
                try:
                    os.remove(os.path.join(input_path, img))
                except Exception as e:
                    print(f"[WARNING] No se pudo eliminar {img}: {e}")
            #print(f"Eliminadas imágenes del grupo {idx} en proceso {id_proc}")
            q.put(1)
    except Exception as e:
        print(f"[Proceso {id_proc}] Error al procesar grupo: {e}")
        q.put(0)


def calcular_fondo_percentil(input_path, imagenes, percentil, grupo_size, num_procesos, cache_path, estado=None):
    os.makedirs(cache_path, exist_ok=True)
    random.shuffle(imagenes)
    grupos = [imagenes[i:i+grupo_size] for i in range(0, len(imagenes), grupo_size)]
    
    # Dividir los grupos en lotes para cada proceso
    lotes_por_proceso = [[] for _ in range(num_procesos)]
    for idx, grupo in enumerate(grupos):
        lotes_por_proceso[idx % num_procesos].append((idx, grupo))

    q = multiprocessing.Queue()
    procesos = []

    #Lanzar procesos para calcular la mediana de cada lote
    for i in range(num_procesos):
        p = multiprocessing.Process(
            target=calcular_mediana_segmento,
            args=(input_path, percentil, cache_path, lotes_por_proceso[i], i, q)
        )
        procesos.append(p)
        p.start()

    # Seguimiento del progreso con proteccion por timeout
    total = len(grupos)
    progreso = 0
    while progreso < total:
        try:
            q.get(timeout=100)  
            progreso += 1
            porcentaje = int((progreso / total) * 100)
            if estado:
                estado.emitir_progreso(porcentaje)
        except queue.Empty:
            print("⚠️ Advertencia: timeout esperando progreso. Puede que un proceso haya fallado.")
            break

    # Esperar a que todos los procesos terminen
    for p in procesos: p.join()

    #print(f"[DEBUG] Procesos terminados, calculando fondo final...")
    medianas = [cv2.imread(os.path.join(cache_path, f))
                for f in os.listdir(cache_path) if f.endswith(".jpg")]
    medianas = [m for m in medianas if m is not None]
    while len(medianas) > grupo_size:
        medianas = [
            np.percentile(np.array(medianas[i:i+grupo_size]), percentil, axis=0).astype(np.uint8)
            for i in range(0, len(medianas), grupo_size)
        ]
    #print(f"[DEBUG] Número de medianas calculadas: {len(medianas)}")
    fondo_final = np.percentile(np.array(medianas), percentil, axis=0).astype(np.uint8)
    #print(f"[DEBUG] Fondo final calculado")
    shutil.rmtree(cache_path)
    shutil.rmtree(input_path, ignore_errors=True)
    return fondo_final

# ---------------------- Procesado ----------------------
def aplicar_pipeline_morfologica(mask, pipeline):
    for paso in pipeline:
        op = paso["op"]
        kx, ky = paso["kernel"]
        kernel = np.ones((kx, ky), np.uint8)
        if op == "apertura":
            #print(f"[DEBUG] Aplicando apertura con kernel {kernel.shape}")
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif op == "cierre":
            #print(f"[DEBUG] Aplicando cierre con kernel {kernel.shape}")
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif op == "dilatacion":
            #print(f"[DEBUG] Aplicando dilatación con kernel {kernel.shape}")
            mask = cv2.dilate(mask, kernel)
        elif op == "erosion":
            #print(f"[DEBUG] Aplicando erosión con kernel {kernel.shape}")
            mask = cv2.erode(mask, kernel)
    return mask

def procesar_segmento(imagenes, input_path, output_path, fondo, pipeline, q , output_dims, zona_no_valida):
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

            #print(f"[Segmento] Procesando máscara para {img_name} en proceso PID {os.getpid()}: resize")
            mask = cv2.resize(mask, output_dims, interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite(os.path.join(output_path, img_name), mask)
            
            # Guardar la máscara codificada como RLE y guardada como .npz
            _, binaria = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

            # Si hay una zona no válida, aplicar la máscara
            if zona_no_valida is not None:
                binaria = binaria * (zona_no_valida == 0).astype(np.uint8)

            pixels = binaria.flatten()
            pixels = np.concatenate([[0], pixels, [0]])
            changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
            runs = changes[1::2] - changes[::2]
            rle = np.empty(changes.size, dtype=int)
            rle[::2] = changes[::2] + 1  # 1-based indexing
            rle[1::2] = runs
            shape = np.array(binaria.shape, dtype=int)

            nombre_salida = os.path.splitext(img_name)[0] + ".npz"
            ruta_salida = os.path.join(output_path, nombre_salida)
            np.savez_compressed(ruta_salida, shape=shape, rle=rle)
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
                                   resize_enabled=False, resize_dims=(1920, 1080), bbox_recorte=None, output_dims=(1920, 1080)):
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
            if resize_enabled:
                #print(f"[DEBUG] Redimensionando fondo a {resize_dims}")
                fondo = cv2.resize(fondo, resize_dims)
            # Redimensionar el fondo si no coincide con el tamaño del vídeo
            # alto_fondo, ancho_fondo = fondo.shape[:2]
            # if (alto_fondo, ancho_fondo) != (alto_og, ancho_og):
            #     print(f"[DEBUG] Redimensionando fondo de ({ancho_fondo}, {alto_fondo}) a ({ancho_og}, {alto_og})")
            #     fondo = cv2.resize(fondo, (ancho_og, alto_og))
            # if bbox_recorte:
            #     h_img, w_img = fondo.shape[:2]
            #     x, y, w, h = map(int, bbox_recorte)
            #     fondo = fondo[y:y+h, x:x+w]
        else:
            if estado:
                estado.emitir_etapa("Calculando fondo...")
                estado.emitir_progreso(0)

            fondo_frames_dir = "processing_GUI/procesamiento/cache/__frames_fondo_tmp__"
            os.makedirs(fondo_frames_dir, exist_ok=True)
            #print(f"[DEBUG] Extrayendo frames del vídeo de fondo: {video_fondo}")
            if estado:
                estado.emitir_etapa("Extrayendo frames del vídeo de fondo...")
                estado.emitir_progreso(0)
            fondo_imagenes = extraer_frames(video_fondo, fondo_frames_dir, resize_enabled, resize_dims, bbox_recorte, estado)
            #print(f"[DEBUG] Número de imágenes extraídas del fondo: {len(fondo_imagenes)}")
            #print(f"[DEBUG] Calculando fondo usando percentil {percentil} con grupo size {grupo_size} y {nucleos} núcleos")
            if estado:
                estado.emitir_etapa("Calculando fondo usando percentil...")
                estado.emitir_progreso(0)
            fondo = calcular_fondo_percentil(
                fondo_frames_dir, fondo_imagenes,
                percentil, grupo_size, nucleos,
                "processing_GUI/procesamiento/cache/__cache_fondo__", estado
            )

        if fondo is not None:
            #print(f"[DEBUG] Fondo shape antes del recorte: {fondo.shape}")
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
            frames_dir = os.path.join("processing_GUI/procesamiento/cache/__frames_tmp__", Path(video).stem)
            carpeta_video = os.path.join(output_dir, Path(video).stem)
            zona_no_valida = cv2.imread("processing_GUI/procesamiento/zona_no_valida.png", 0)
            if zona_no_valida is None:
                print("⚠️ Advertencia: zona_no_valida.png no encontrada. Continuando sin máscara de ignorar.")

            os.makedirs(carpeta_video, exist_ok=True)
            output_masks = os.path.join(carpeta_video, "masks_rle")
            if bbox_recorte:
                os.makedirs(output_masks, exist_ok=True)
                h_img, w_img = fondo.shape[:2]
                x, y, w, h = bbox_recorte
                recorte_info = {
                    "left": x,
                    "top": y,
                    "right": w_img - (x + w),
                    "bottom": h_img - (y + h),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "shape": [h_img, w_img]
                }
                with open(os.path.join(output_masks, "recorte_morphology.json"), "w") as f:
                    json.dump(recorte_info, f, indent=4)
            output_dims_info = {
                "output_dims": [output_dims[0], output_dims[1]]
            }
            with open(os.path.join(output_masks, "output_dims.json"), "w") as f:
                json.dump(output_dims_info, f, indent=4)
            estado.emitir_etapa(f"Extrayendo frames de {video}...")
            imagenes = extraer_frames(video_path, frames_dir, resize_enabled, resize_dims, bbox_recorte, estado)
            if not imagenes:
                if estado:
                    estado.emitir_error("No se encontraron imágenes para procesar.")
                return

            segment_size = len(imagenes) // nucleos
            q = multiprocessing.Queue()
            procesos = []
            if estado:
                estado.emitir_etapa("Aplicando morfología...")
                estado.emitir_progreso(0)
            for j in range(nucleos):
                ini = j * segment_size
                fin = (j + 1) * segment_size if j < nucleos - 1 else len(imagenes)
                segmento = imagenes[ini:fin]
                p = multiprocessing.Process(
                target=procesar_segmento,
                args=(segmento, frames_dir, output_masks, fondo, pipeline, q, output_dims, zona_no_valida)
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

