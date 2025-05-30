import os
import json
import multiprocessing
from pathlib import Path
import traceback
import numpy as np
import cv2

# ---------------------- EstadoProceso ----------------------
class EstadoProceso:
    def __init__(self):
        self.on_etapa = None
        self.on_progreso = None
        self.on_error = None
        self.on_total_videos = None
        self.on_video_progreso = None


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


# ---------------------- ESTADÍSTICOS DERIVADOS DE BBOX ----------------------




# ---------------------- Función auxiliar: extraer centroides ----------------------
def extraer_centroides(ruta_labels, ruta_salida_centroides):
    try:
        centroides_dict = {}
        archivos = sorted([f for f in os.listdir(ruta_labels) if f.endswith(".txt")])

        for archivo in archivos:
            ruta_txt = os.path.join(ruta_labels, archivo)
            frame_key = Path(archivo).stem  # frame_00000

            with open(ruta_txt, "r") as f:
                lineas = f.readlines()

            centroides_frame = []
            for linea in lineas:
                partes = linea.strip().split()
                if len(partes) != 5:
                    continue
                _, x1, y1, x2, y2 = map(float, partes)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centroides_frame.append([cx, cy])

            centroides_dict[frame_key] = centroides_frame

        with open(ruta_salida_centroides, "w") as f_out:
            json.dump(centroides_dict, f_out, indent=2)
        return True

    except Exception as e:
        print(f"[Error] al extraer centroides: {str(e)}")
        return False


# ---------------------- Estadístico Bbox: Distribución espacial ----------------------
def calcular_distribucion_espacial(ruta_centroides_json, salida_dir, dimensiones_entrada, tamanos_grid, num_procesos, estado):
    try:
        with open(ruta_centroides_json, "r") as f:
            centroides_dict = json.load(f)

        ancho, alto = dimensiones_entrada
        tareas = [(grid_size, centroides_dict, ancho, alto) for grid_size in tamanos_grid]

        with multiprocessing.Pool(processes=num_procesos) as pool:
            resultados = pool.starmap(procesar_grid, tareas)

        for grid_size, histograma in resultados:
            nombre_archivo = f"distribucion_espacial_{grid_size}.json"
            ruta_out = os.path.join(salida_dir, nombre_archivo)
            with open(ruta_out, "w") as f_out:
                json.dump({"grid_size": [grid_size, grid_size], "histograma": histograma}, f_out, indent=2)

            estado.emitir_etapa(f"Grid {grid_size}x{grid_size} completado")

    except Exception as e:
        estado.emitir_error(f"Error en calcular_distribucion_espacial: {str(e)}")

def procesar_grid(grid_size, centroides_dict, ancho, alto):
    filas, cols = grid_size, grid_size
    resultado_por_frame = {}

    for frame_key, centroides in centroides_dict.items():
        histograma = [[0 for _ in range(cols)] for _ in range(filas)]
        for cx, cy in centroides:
            i = min(int(cy / alto * filas), filas - 1)
            j = min(int(cx / ancho * cols), cols - 1)
            histograma[i][j] += 1
        resultado_por_frame[frame_key] = histograma

    return grid_size, resultado_por_frame

# ---------------------- Estadístico Bbox: Área media ----------------------
def calcular_areas_blobs(ruta_labels, salida_path, estado):
    try:
        resultado_por_frame = {}
        archivos = sorted([f for f in os.listdir(ruta_labels) if f.endswith(".txt")])

        total_frames = len(archivos)
        for i, archivo in enumerate(archivos):
            frame_key = Path(archivo).stem  # frame_00000
            ruta_txt = os.path.join(ruta_labels, archivo)

            with open(ruta_txt, "r") as f:
                lineas = f.readlines()

            areas = []
            for linea in lineas:
                partes = linea.strip().split()
                if len(partes) != 5:
                    continue
                _, x1, y1, x2, y2 = map(float, partes)
                area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                areas.append(area)

            if areas:
                media = float(np.mean(areas))
                std = float(np.std(areas))
            else:
                media = 0.0
                std = 0.0

            resultado_por_frame[frame_key] = {
                "areas": areas,
                "media": media,
                "std": std
            }

            if i % 25 == 0:
                porcentaje = int((i / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_areas_blobs: {str(e)}")

# ---------------------- Estadísticos Bbox: Distancia centroides ----------------------
def calcular_distancia_centroides(ruta_centroides_json, salida_path, estado):
    try:
        with open(ruta_centroides_json, "r") as f:
            centroides_dict = json.load(f)

        resultado_por_frame = {}
        total_frames = len(centroides_dict)
        for i, (frame_key, centroides) in enumerate(centroides_dict.items()):
            distancias = []
            n = len(centroides)
            for j in range(n):
                for k in range(j + 1, n):
                    cx1, cy1 = centroides[j]
                    cx2, cy2 = centroides[k]
                    dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                    distancias.append(dist)

            if distancias:
                media = float(np.mean(distancias))
                std = float(np.std(distancias))
            else:
                media = 0.0
                std = 0.0

            resultado_por_frame[frame_key] = {
                "distancias": distancias,
                "media": media,
                "std": std
            }

            if i % 25 == 0:
                porcentaje = int((i / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_distancia_centroides: {str(e)}")

# ---------------------- Estadístico Bbox: Coeficiente de agrupación ----------------------
def calcular_coef_agrupacion(ruta_centroides_json, ruta_labels, salida_path,
                              umbral_distancia, umbral_iou, estado):
    try:
        with open(ruta_centroides_json, "r") as f:
            centroides_dict = json.load(f)

        resultado_por_frame = {}
        total_frames = len(centroides_dict)
        archivos_labels = sorted([f for f in os.listdir(ruta_labels) if f.endswith(".txt")])

        for i, archivo in enumerate(archivos_labels):
            frame_key = Path(archivo).stem  # ej. "frame_00001"
            if frame_key not in centroides_dict:
                continue

            centroides = centroides_dict[frame_key]
            ruta_txt = os.path.join(ruta_labels, archivo)

            with open(ruta_txt, "r") as f:
                lineas = f.readlines()

            bboxes = []
            for linea in lineas:
                partes = linea.strip().split()
                if len(partes) != 5:
                    continue
                _, x1, y1, x2, y2 = map(float, partes)
                bboxes.append((x1, y1, x2, y2))

            n = len(centroides)
            num_agrupados = 0
            agrupados_idx = []

            for j in range(n):
                cx1, cy1 = centroides[j]
                x1a, y1a, x2a, y2a = bboxes[j]
                bbox_a = (x1a, y1a, x2a, y2a)

                agrupado = False
                for k in range(n):
                    if j == k:
                        continue
                    cx2, cy2 = centroides[k]
                    x1b, y1b, x2b, y2b = bboxes[k]
                    bbox_b = (x1b, y1b, x2b, y2b)

                    # Criterio 1: distancia
                    dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                    if dist < umbral_distancia:
                        agrupado = True
                        break

                    # Criterio 2: IoU
                    iou = calcular_iou(bbox_a, bbox_b)
                    if iou >= umbral_iou:
                        agrupado = True
                        break

                if agrupado:
                    num_agrupados += 1
                    agrupados_idx.append(j)

            agrupacion = num_agrupados / n if n > 0 else 0.0

            resultado_por_frame[frame_key] = {
                "centroides_total": n,
                "centroides_agrupados": num_agrupados,
                "agrupacion": agrupacion,
                "agrupados": agrupados_idx
            }

            if i % 25 == 0:
                porcentaje = int((i / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_coef_agrupacion con bbox: {str(e)}")


def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


# ---------------------- Estadístico Bbox: Entropía espacial ----------------------
def calcular_entropia_espacial(ruta_centroides_json, salida_path, dimensiones_entrada,
                                grid_size, estado):
    import numpy as np
    try:
        with open(ruta_centroides_json, "r") as f:
            centroides_dict = json.load(f)

        resultado_por_frame = {}
        total_frames = len(centroides_dict)
        ancho, alto = dimensiones_entrada
        filas = columnas = grid_size
        total_celdas = filas * columnas

        for i, (frame_key, centroides) in enumerate(centroides_dict.items()):
            histograma = np.zeros((filas, columnas), dtype=int)

            for cx, cy in centroides:
                col = min(int(cx / ancho * columnas), columnas - 1)
                fil = min(int(cy / alto * filas), filas - 1)
                histograma[fil, col] += 1

            total_centroides = np.sum(histograma)
            if total_centroides == 0:
                entropia = 0.0
            else:
                p = histograma.flatten() / total_centroides
                p_no_cero = p[p > 0]
                entropia = -np.sum(p_no_cero * np.log2(p_no_cero))
                max_entropia = np.log2(np.count_nonzero(p))
                entropia_normalizada = float(entropia / max_entropia) if max_entropia > 0 else 0.0

            resultado_por_frame[frame_key] = {
                "entropia": entropia_normalizada,
                "total_centroides": int(total_centroides)
            }

            if i % 25 == 0:
                porcentaje = int((i / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_entropia_espacial: {str(e)}")

# ---------------------- Estadístico Bbox: Índice de exploración ----------------------
def calcular_indice_exploracion(ruta_centroides_json, salida_path, dimensiones_entrada,
                                 grid_size, ventana_frames, estado):
    try:
        import numpy as np

        with open(ruta_centroides_json, "r") as f:
            centroides_dict = json.load(f)

        ancho, alto = dimensiones_entrada
        filas = columnas = grid_size
        total_celdas = filas * columnas

        # Ordenar los frames
        frame_keys = sorted(centroides_dict.keys())
        total_frames = len(frame_keys)

        resultado_por_ventana = {}
        celdas_globales = set()

        for i in range(0, total_frames, ventana_frames):
            window_keys = frame_keys[i:i + ventana_frames]
            celdas_en_ventana = set()

            for fk in window_keys:
                for cx, cy in centroides_dict.get(fk, []):
                    col = min(int(cx / ancho * columnas), columnas - 1)
                    fil = min(int(cy / alto * filas), filas - 1)
                    idx = fil * columnas + col
                    celdas_en_ventana.add(idx)
                    celdas_globales.add(idx)

            idx_str = f"ventana_{i:05d}"
            resultado_por_ventana[idx_str] = round(len(celdas_en_ventana) / total_celdas, 3)

            estado.emitir_etapa(f"Ventana {idx_str} procesada")
            estado.emitir_progreso(int((i + ventana_frames) / total_frames * 100))

        resultado = {
            "grid": [filas, columnas],
            "ventana_frames": ventana_frames,
            "por_ventana": resultado_por_ventana,
            "global": round(len(celdas_globales) / total_celdas, 3)
        }

        with open(salida_path, "w") as f_out:
            json.dump(resultado, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_indice_exploracion: {str(e)}")

# ---------------------- Estadístico Bbox: Distancia al centroide global ----------------------
def calcular_distancia_centroide_grupal(ruta_centroides_json, salida_path, estado):
    import numpy as np
    try:
        with open(ruta_centroides_json, "r") as f:
            centroides_dict = json.load(f)

        resultado_por_frame = {}
        total_frames = len(centroides_dict)

        for i, (frame_key, centroides) in enumerate(centroides_dict.items()):
            if not centroides:
                resultado_por_frame[frame_key] = {
                    "media": 0.0,
                    "std": 0.0,
                    "centroide_grupal": [0.0, 0.0]
                }
                continue

            coords = np.array(centroides)
            centro_grupal = np.mean(coords, axis=0)
            distancias = np.linalg.norm(coords - centro_grupal, axis=1)
            media = float(np.mean(distancias))
            std = float(np.std(distancias))

            resultado_por_frame[frame_key] = {
                "media": media,
                "std": std,
                "centroide_grupal": centro_grupal.tolist()
            }

            if i % 25 == 0:
                porcentaje = int((i / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        # Extraer solo los centroides grupales
        centroides_grupales_dict = {
            frame: valores["centroide_grupal"]
            for frame, valores in resultado_por_frame.items()
        }

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

        # Guardar solo los centroides grupales
        path_centroide_only = salida_path.replace("distancia_centroide_grupal.json", "centroide_grupal.json")
        with open(path_centroide_only, "w") as f_c:
            json.dump(centroides_grupales_dict, f_c, indent=2)


    except Exception as e:
        estado.emitir_error(f"Error en calcular_distancia_centroide_grupal: {str(e)}")

# ---------------------- Estadístico Bbox: Densidad Local ----------------------
def calcular_densidad_local(ruta_centroides_json, ruta_labels, salida_path, estado,
                            umbral_distancia=75, umbral_iou=0.3):
    import numpy as np
    import os

    try:
        with open(ruta_centroides_json, "r") as f:
            centroides_dict = json.load(f)

        resultado_por_frame = {}
        total_frames = len(centroides_dict)

        for idx, frame_key in enumerate(sorted(centroides_dict.keys())):
            centroides = centroides_dict[frame_key]
            path_txt = os.path.join(ruta_labels, f"{frame_key}.txt")

            if not os.path.exists(path_txt) or not centroides:
                resultado_por_frame[frame_key] = {
                    "densidad_media": 0.0,
                    "std": 0.0
                }
                continue

            with open(path_txt, "r") as f_txt:
                bboxes = [list(map(float, line.strip().split()[1:])) for line in f_txt if line.strip()]
                bboxes_abs = [  # [x1, y1, x2, y2]
                    [x - w/2, y - h/2, x + w/2, y + h/2]
                    for x, y, w, h in bboxes
                ]

            vecinos_por_blob = []
            for i, (c_i, bb_i) in enumerate(zip(centroides, bboxes_abs)):
                vecinos = 0
                for j, (c_j, bb_j) in enumerate(zip(centroides, bboxes_abs)):
                    if i == j:
                        continue
                    dist = np.linalg.norm(np.array(c_i) - np.array(c_j))

                    # IoU
                    xi1 = max(bb_i[0], bb_j[0])
                    yi1 = max(bb_i[1], bb_j[1])
                    xi2 = min(bb_i[2], bb_j[2])
                    yi2 = min(bb_i[3], bb_j[3])
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area_i = (bb_i[2] - bb_i[0]) * (bb_i[3] - bb_i[1])
                    area_j = (bb_j[2] - bb_j[0]) * (bb_j[3] - bb_j[1])
                    union_area = area_i + area_j - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0

                    if dist < umbral_distancia or iou >= umbral_iou:
                        vecinos += 1

                vecinos_por_blob.append(vecinos)

            densidad_media = float(np.mean(vecinos_por_blob)) if vecinos_por_blob else 0.0
            std = float(np.std(vecinos_por_blob)) if vecinos_por_blob else 0.0

            resultado_por_frame[frame_key] = {
                "densidad_media": densidad_media,
                "std": std
            }

            if idx % 25 == 0:
                porcentaje = int((idx / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_densidad_local: {str(e)}")





# ---------------------- ESTADÍSTICOS DERIVADOS DE MÁSCARAS ----------------------




# ---------------------- Estadístico Máscaras: Histograma de Densidad ----------------------
def calcular_histograma_densidad(ruta_masks, salida_dir, dimensiones_entrada, estado, tamanos_grid=[5, 10, 15, 20], num_procesos=4):
    import cv2
    try:
        archivos = sorted([f for f in os.listdir(ruta_masks) if f.endswith(('.png', '.jpg'))])
        total_frames = len(archivos)
        if total_frames == 0:
            estado.emitir_etapa("[Aviso] No se encontraron máscaras.")
            return

        ancho, alto = dimensiones_entrada

        with multiprocessing.Pool(processes=num_procesos) as pool:
            resultados = pool.map(procesar_grid, tamanos_grid, archivos, ruta_masks, ancho, alto, estado, total_frames)

        for grid_size, resultado in resultados:
            nombre_archivo = f"densidad_{grid_size}.json"
            ruta_out = os.path.join(salida_dir, nombre_archivo)
            with open(ruta_out, "w") as f_out:
                json.dump({
                    "grid_size": [grid_size, grid_size],
                    "densidad": resultado
                }, f_out, indent=2)

            estado.emitir_etapa(f"Grid {grid_size}x{grid_size} completado")

    except Exception as e:
        estado.emitir_error(f"Error en calcular_histograma_densidad: {str(e)}")

def procesar_grid(grid_size, archivos, ruta_masks, ancho, alto, estado, total_frames=None):
            filas, columnas = grid_size, grid_size
            resultado_por_frame = {}

            for idx, nombre_archivo in enumerate(archivos):
                ruta = os.path.join(ruta_masks, nombre_archivo)
                mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                h, w = mask.shape[:2]
                if (w, h) != (ancho, alto):
                    mask = cv2.resize(mask, (ancho, alto), interpolation=cv2.INTER_NEAREST)

                frame_key = Path(nombre_archivo).stem
                celda_h = alto // filas
                celda_w = ancho // columnas
                grid = [[0 for _ in range(columnas)] for _ in range(filas)]

                for i in range(filas):
                    for j in range(columnas):
                        y1 = i * celda_h
                        x1 = j * celda_w
                        y2 = alto if i == filas - 1 else (i + 1) * celda_h
                        x2 = ancho if j == columnas - 1 else (j + 1) * celda_w
                        sub_mask = mask[y1:y2, x1:x2]
                        grid[i][j] = int(np.count_nonzero(sub_mask))

                resultado_por_frame[frame_key] = grid

                if idx % 20 == 0:
                    estado.emitir_progreso(int((idx / total_frames) * 100))

            return grid_size, resultado_por_frame

# ---------------------- Procesamiento principal ----------------------
def procesar_bbox_stats(carpeta_trabajo, estadisticos_seleccionados, num_procesos, dimensiones_entrada, estado):
    try:

        carpetas_videos = [os.path.join(carpeta_trabajo, d) for d in os.listdir(carpeta_trabajo)
                           if os.path.isdir(os.path.join(carpeta_trabajo, d))]

        total_videos = len(carpetas_videos)
        if total_videos == 0:
            estado.emitir_error("No se encontraron subcarpetas de vídeos.")
            return

        estado.emitir_etapa("Iniciando estadísticos de BBox...")
        estado.emitir_total_videos(total_videos)

        for idx, carpeta_video in enumerate(carpetas_videos):
            try:
                estado.emitir_progreso(0)
                nombre_video = os.path.basename(carpeta_video)
                estado.emitir_etapa(f"Procesando {nombre_video} ({idx+1}/{total_videos})...")
                ruta_labels = os.path.join(carpeta_video, "bbox", "labels")

                if not os.path.exists(ruta_labels):
                    estado.emitir_etapa(f"[Aviso] {nombre_video} no tiene etiquetas YOLO.")
                    continue

                salida_stats = os.path.join(carpeta_video, "bbox_stats")
                os.makedirs(salida_stats, exist_ok=True)

                # Paso previo: extraer centroides y guardarlos
                ruta_centroides_json = os.path.join(salida_stats, "centroides.json")
                estado.emitir_etapa("Calculando centroides...")
                ok = extraer_centroides(ruta_labels, ruta_centroides_json)
                if not ok:
                    estado.emitir_etapa(f"[Aviso] No se pudieron calcular centroides para {nombre_video}.")
                    continue

                total_descriptores = len(estadisticos_seleccionados)
                contador = 0



                # Aquí se llamarían las funciones individuales de estadísticos
                if "distribucion_espacial" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Distribución espacial")
                    print(f"[INFO] Procesando distribución espacial para {nombre_video}...")
                    calcular_distribucion_espacial(
                        ruta_centroides_json,
                        salida_stats,
                        dimensiones_entrada,
                        tamanos_grid=[5, 10, 15, 20],
                        num_procesos=num_procesos,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))
                    estado.emitir_video_progreso(idx)

                if "area_media" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Área media de blobs")
                    print(f"[INFO] Procesando área media de blobs para {nombre_video}...")
                    ruta_salida_area = os.path.join(salida_stats, "areas_blobs.json")
                    calcular_areas_blobs(
                        ruta_labels=ruta_labels,
                        salida_path=ruta_salida_area,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "distancia_centroides" in estadisticos_seleccionados:
                    print(f"[INFO] Procesando distancia entre centroides para {nombre_video}...")
                    estado.emitir_etapa("Calculando: Distancia entre centroides")
                    ruta_salida_distancias = os.path.join(salida_stats, "distancia_centroides.json")
                    calcular_distancia_centroides(
                        ruta_centroides_json=ruta_centroides_json,
                        salida_path=ruta_salida_distancias,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "coeficiente_agrupacion" in estadisticos_seleccionados:
                    print(f"[INFO] Procesando coeficiente de agrupación para {nombre_video}...")
                    estado.emitir_etapa("Calculando: Coeficiente de agrupación (centroide + IoU)")
                    ruta_salida_agrupacion = os.path.join(salida_stats, "coef_agrupacion.json")
                    calcular_coef_agrupacion(
                        ruta_centroides_json=ruta_centroides_json,
                        ruta_labels=ruta_labels,
                        salida_path=ruta_salida_agrupacion,
                        umbral_distancia=75,
                        umbral_iou=0.3,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "entropia_espacial" in estadisticos_seleccionados:
                    print(f"[INFO] Procesando entropía espacial para {nombre_video}...")
                    estado.emitir_etapa("Calculando: Entropía espacial")
                    ruta_salida_entropia = os.path.join(salida_stats, "entropia.json")
                    calcular_entropia_espacial(
                        ruta_centroides_json=ruta_centroides_json,
                        salida_path=ruta_salida_entropia,
                        dimensiones_entrada=dimensiones_entrada,
                        grid_size=10,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "indice_exploracion" in estadisticos_seleccionados:
                    print(f"[INFO] Procesando índice de exploración para {nombre_video}...")
                    estado.emitir_etapa("Calculando: Índice de exploración")
                    ruta_salida_exploracion = os.path.join(salida_stats, "exploracion.json")
                    calcular_indice_exploracion(
                        ruta_centroides_json=ruta_centroides_json,
                        salida_path=ruta_salida_exploracion,
                        dimensiones_entrada=dimensiones_entrada,
                        grid_size=25,
                        ventana_frames=128,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "distancia_centroide_global" in estadisticos_seleccionados:
                    print(f"[INFO] Procesando distancia al centroide grupal para {nombre_video}...")
                    estado.emitir_etapa("Calculando: Distancia al centroide grupal")
                    ruta_salida_distancia_grupal = os.path.join(salida_stats, "distancia_centroide_grupal.json")
                    calcular_distancia_centroide_grupal(
                        ruta_centroides_json=ruta_centroides_json,
                        salida_path=ruta_salida_distancia_grupal,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "densidad_local" in estadisticos_seleccionados:
                    print(f"[INFO] Procesando densidad local para {nombre_video}...")
                    estado.emitir_etapa("Calculando: Densidad local")
                    ruta_salida_densidad = os.path.join(salida_stats, "densidad_local.json")
                    calcular_densidad_local(
                        ruta_centroides_json=ruta_centroides_json,
                        ruta_labels=ruta_labels,
                        salida_path=ruta_salida_densidad,
                        estado=estado,
                        umbral_distancia=75,
                        umbral_iou=0.3
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))






                # ...otros estadísticos aquí

                estado.emitir_progreso(int(((idx + 1) / total_videos) * 100))
                print(f"[INFO] Procesados {idx + 1} de {total_videos} vídeos.")


            except Exception as e:
                print(f"[Error] al procesar {carpeta_video}: {str(e)}")
                estado.emitir_etapa(f"[Error] en {carpeta_video}: {str(e)}")
                continue 

        estado.emitir_etapa("Cálculo completado.")

    except Exception as e:
        error_msg = f"Error en procesar_bbox_stats: {str(e)}\n{traceback.format_exc()}"
        estado.emitir_error(error_msg)


def procesar_mask_stats(carpeta_trabajo, estadisticos_seleccionados, num_procesos, dimensiones_entrada, estado):
    try:
        carpetas_videos = [os.path.join(carpeta_trabajo, d) for d in os.listdir(carpeta_trabajo)
                           if os.path.isdir(os.path.join(carpeta_trabajo, d))]

        total_videos = len(carpetas_videos)
        if total_videos == 0:
            estado.emitir_error("No se encontraron subcarpetas de vídeos.")
            return

        estado.emitir_etapa("Iniciando estadísticos de máscaras...")
        estado.emitir_total_videos(total_videos)

        for idx, carpeta_video in enumerate(carpetas_videos):
            try:
                estado.emitir_progreso(0)
                nombre_video = os.path.basename(carpeta_video)
                estado.emitir_etapa(f"Procesando {nombre_video} ({idx+1}/{total_videos})...")

                ruta_masks = os.path.join(carpeta_video, "máscaras")
                if not os.path.exists(ruta_masks):
                    estado.emitir_etapa(f"[Aviso] {nombre_video} no tiene máscaras.")
                    continue

                salida_stats = os.path.join(carpeta_video, "mask_stats")
                os.makedirs(salida_stats, exist_ok=True)

                total_descriptores = len(estadisticos_seleccionados)
                contador = 0

                # Aquí se insertará cada estadístico
                if "histograma_densidad" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Histograma de densidad")
                    calcular_histograma_densidad(
                        ruta_masks=ruta_masks,
                        salida_dir=salida_stats,
                        dimensiones_entrada=dimensiones_entrada,
                        estado=estado,
                        tamanos_grid=[5, 10, 15, 20],
                        num_procesos=num_procesos
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))


                # ...otros estadísticos aquí...

                estado.emitir_video_progreso(idx)

            except Exception as e:
                estado.emitir_etapa(f"[Error] en {carpeta_video}: {str(e)}")
                continue

        estado.emitir_etapa("Cálculo completado para estadísticas de máscaras.")

    except Exception as e:
        estado.emitir_error(f"Error en procesar_mask_stats: {str(e)}")



    