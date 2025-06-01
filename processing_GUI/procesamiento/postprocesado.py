import os
import json
import multiprocessing
from pathlib import Path
import traceback
import numpy as np
import cv2
from scipy.signal import correlate2d

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

# ---------------------- Estadístico Bbox: Velocidad centroide global y ángulo ----------------------
def calcular_velocidad_centroide(ruta_centroide_json, salida_path, estado, n_bins=12):
    try:
        if not os.path.exists(ruta_centroide_json):
            estado.emitir_etapa("No se encontró 'centroide_grupal.json'.")
            return

        with open(ruta_centroide_json, "r") as f:
            centros = json.load(f)

        frame_keys = sorted(centros.keys())
        if len(frame_keys) < 2:
            estado.emitir_etapa("No hay suficientes frames para calcular velocidad.")
            return

        resultado_por_frame = {}

        for i in range(1, len(frame_keys)):
            prev = centros[frame_keys[i - 1]]
            curr = centros[frame_keys[i]]

            if prev is None or curr is None or None in prev or None in curr:
                resultado_por_frame[frame_keys[i - 1]] = {
                    "velocidad": 0.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "angulo": None,
                    "bin": None
                }
                continue

            x1, y1 = prev
            x2, y2 = curr
            dx = x2 - x1
            dy = y2 - y1
            velocidad = float(np.sqrt(dx**2 + dy**2))

            angle_rad = np.arctan2(dy, dx)
            angle_deg = (np.degrees(angle_rad) + 360) % 360
            bin_id = int(angle_deg // (360 / n_bins))

            resultado_por_frame[frame_keys[i - 1]] = {
                "velocidad": velocidad,
                "dx": dx,
                "dy": dy,
                "angulo": angle_deg,
                "bin": bin_id
            }

            if i % 25 == 0:
                porcentaje = int((i / len(frame_keys)) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_velocidad_centroide: {str(e)}")




# ---------------------- ESTADÍSTICOS DERIVADOS DE MÁSCARAS ----------------------




# ---------------------- Estadístico Máscaras: Histograma de Densidad ----------------------
def procesar_grid_worker(grid_size, archivos, ruta_masks, ancho, alto):
    filas, columnas = grid_size, grid_size
    resultado_por_frame = {}

    for nombre_archivo in archivos:
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
        grid = [[0.0 for _ in range(columnas)] for _ in range(filas)]

        for i in range(filas):
            for j in range(columnas):
                y1 = i * celda_h
                x1 = j * celda_w
                y2 = alto if i == filas - 1 else (i + 1) * celda_h
                x2 = ancho if j == columnas - 1 else (j + 1) * celda_w
                sub_mask = mask[y1:y2, x1:x2]

                total_pixeles = sub_mask.shape[0] * sub_mask.shape[1]
                ocupados = np.count_nonzero(sub_mask)
                porcentaje = ocupados / total_pixeles if total_pixeles > 0 else 0.0

                grid[i][j] = round(porcentaje, 4)

        resultado_por_frame[frame_key] = grid

    return (grid_size, resultado_por_frame)

def calcular_histograma_densidad(ruta_masks, salida_dir, dimensiones_entrada, estado, tamanos_grid=[5, 10, 15, 20], num_procesos=4):
    try:
        archivos = sorted([f for f in os.listdir(ruta_masks) if f.endswith(('.png', '.jpg'))])
        total_frames = len(archivos)
        if total_frames == 0:
            estado.emitir_etapa("[Aviso] No se encontraron máscaras.")
            return

        ancho, alto = dimensiones_entrada

        tareas = [(g, archivos, ruta_masks, ancho, alto) for g in tamanos_grid]

        from multiprocessing import Pool
        with Pool(processes=num_procesos) as pool:
            resultados = pool.starmap(procesar_grid_worker, tareas)

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

# ---------------------- Estadístico Máscaras: Centro de masa global ----------------------
def calcular_centro_masa_mascaras(ruta_masks, salida_path, dimensiones_entrada, estado):
    try:
        archivos = sorted([f for f in os.listdir(ruta_masks) if f.endswith(('.png', '.jpg'))])
        total_frames = len(archivos)
        if total_frames == 0:
            estado.emitir_etapa("[Aviso] No se encontraron máscaras.")
            return

        ancho, alto = dimensiones_entrada
        resultado_por_frame = {}

        for idx, nombre_archivo in enumerate(archivos):
            ruta = os.path.join(ruta_masks, nombre_archivo)
            mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h, w = mask.shape[:2]
            if (w, h) != (ancho, alto):
                mask = cv2.resize(mask, (ancho, alto), interpolation=cv2.INTER_NEAREST)

            ys, xs = np.nonzero(mask)
            frame_key = Path(nombre_archivo).stem

            if len(xs) == 0:
                centro = [None, None]
            else:
                x_cm = float(np.mean(xs))
                y_cm = float(np.mean(ys))
                centro = [x_cm, y_cm]

            resultado_por_frame[frame_key] = centro

            if idx % 25 == 0:
                porcentaje = int((idx / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_centro_masa_mascaras: {str(e)}")

# ---------------------- Estadístico Máscaras: Varianza Espacial ----------------------
def calcular_varianza_espacial(ruta_masks, salida_path, dimensiones_entrada, estado):
    try:
        archivos = sorted([f for f in os.listdir(ruta_masks) if f.endswith(('.png', '.jpg'))])
        total_frames = len(archivos)
        if total_frames == 0:
            estado.emitir_etapa("[Aviso] No se encontraron máscaras.")
            return

        ancho, alto = dimensiones_entrada
        resultado_por_frame = {}

        for idx, nombre_archivo in enumerate(archivos):
            ruta = os.path.join(ruta_masks, nombre_archivo)
            mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h, w = mask.shape[:2]
            if (w, h) != (ancho, alto):
                mask = cv2.resize(mask, (ancho, alto), interpolation=cv2.INTER_NEAREST)

            ys, xs = np.nonzero(mask)
            frame_key = Path(nombre_archivo).stem

            if len(xs) == 0:
                resultado_por_frame[frame_key] = {
                    "varianza": 0.0,
                    "std": 0.0,
                    "n_pixeles": 0
                }
            else:
                coords = np.stack((xs, ys), axis=1).astype(np.float32)  # (N, 2)
                centro = np.mean(coords, axis=0)
                dists_sq = np.sum((coords - centro)**2, axis=1)
                varianza = float(np.mean(dists_sq))
                std = float(np.sqrt(varianza))

                resultado_por_frame[frame_key] = {
                    "varianza": varianza,
                    "std": std,
                    "n_pixeles": int(len(xs))
                }

            if idx % 25 == 0:
                porcentaje = int((idx / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_varianza_espacial: {str(e)}")

# ---------------------- Estadístico Máscaras: Velocidad Grupo ----------------------
def calcular_velocidad_grupo(ruta_masks, salida_path, dimensiones_entrada, estado):
    try:
        archivos = sorted([f for f in os.listdir(ruta_masks) if f.endswith(('.png', '.jpg'))])
        total_frames = len(archivos)
        if total_frames < 2:
            estado.emitir_etapa("[Aviso] Se necesitan al menos 2 frames para calcular velocidad.")
            return

        ancho, alto = dimensiones_entrada
        resultado_por_frame = {}

        n_bins = 12  # sectores angulares

        for idx in range(1, total_frames):
            nombre_anterior = archivos[idx - 1]
            nombre_actual = archivos[idx]

            ruta_anterior = os.path.join(ruta_masks, nombre_anterior)
            ruta_actual = os.path.join(ruta_masks, nombre_actual)

            mask_prev = cv2.imread(ruta_anterior, cv2.IMREAD_GRAYSCALE)
            mask_now = cv2.imread(ruta_actual, cv2.IMREAD_GRAYSCALE)

            if mask_prev is None or mask_now is None:
                continue

            h, w = mask_prev.shape[:2]
            if (w, h) != (ancho, alto):
                mask_prev = cv2.resize(mask_prev, (ancho, alto), interpolation=cv2.INTER_NEAREST)
                mask_now = cv2.resize(mask_now, (ancho, alto), interpolation=cv2.INTER_NEAREST)

            ys_prev, xs_prev = np.nonzero(mask_prev)
            ys_now, xs_now = np.nonzero(mask_now)

            frame_key = Path(nombre_anterior).stem

            if len(xs_prev) == 0 or len(xs_now) == 0:
                resultado_por_frame[frame_key] = {
                    "velocidad": 0.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "angulo": None,
                    "bin": None
                }
                continue

            x_prev = np.mean(xs_prev)
            y_prev = np.mean(ys_prev)
            x_now = np.mean(xs_now)
            y_now = np.mean(ys_now)

            dx = x_now - x_prev
            dy = y_now - y_prev
            velocidad = float(np.sqrt(dx**2 + dy**2))

            angle_rad = np.arctan2(dy, dx)
            angle_deg = (np.degrees(angle_rad) + 360) % 360
            bin_id = int(angle_deg // (360 / n_bins))

            resultado_por_frame[frame_key] = {
                "velocidad": velocidad,
                "dx": dx,
                "dy": dy,
                "angulo": angle_deg,
                "bin": bin_id
            }

            if idx % 25 == 0:
                porcentaje = int((idx / total_frames) * 100)
                estado.emitir_progreso(porcentaje)

        with open(salida_path, "w") as f_out:
            json.dump(resultado_por_frame, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_velocidad_grupo: {str(e)}")

# ---------------------- Estadístico Máscaras: Persistencia Espacial ----------------------
def worker_persistencia(args):
    ventana_id, archivos, ruta_masks, ancho, alto, grid_size = args
    filas = columnas = grid_size
    celda_h = alto // filas
    celda_w = ancho // columnas

    # Matriz para llevar la cuenta de persistencia por celda
    rachas = [[[] for _ in range(columnas)] for _ in range(filas)]
    activos = [[0 for _ in range(columnas)] for _ in range(filas)]

    for nombre_archivo in archivos:
        ruta = os.path.join(ruta_masks, nombre_archivo)
        mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = cv2.resize(mask, (ancho, alto), interpolation=cv2.INTER_NEAREST)
        binaria = (mask > 0).astype(np.uint8)

        for i in range(filas):
            for j in range(columnas):
                y1, y2 = i * celda_h, alto if i == filas - 1 else (i + 1) * celda_h
                x1, x2 = j * celda_w, ancho if j == columnas - 1 else (j + 1) * celda_w
                sub_mask = binaria[y1:y2, x1:x2]
                activa = int(np.any(sub_mask))
                if activa:
                    activos[i][j] += 1
                else:
                    if activos[i][j] > 0:
                        rachas[i][j].append(activos[i][j])
                        activos[i][j] = 0

    # Cierre de rachas al final
    for i in range(filas):
        for j in range(columnas):
            if activos[i][j] > 0:
                rachas[i][j].append(activos[i][j])

    persistencia_por_celda = {}
    medias = []

    for i in range(filas):
        for j in range(columnas):
            valores = rachas[i][j]
            if valores:
                media = float(np.mean(valores))
                maximo = int(np.max(valores))
                persistencia_por_celda[f"{i}_{j}"] = {"media": round(media, 2), "max": maximo}
                medias.append(media)

    if medias:
        media_global = round(np.mean(medias), 3)
        std_global = round(np.std(medias), 3)
    else:
        media_global = 0.0
        std_global = 0.0

    return ventana_id, {
        "por_celda": persistencia_por_celda,
        "media_global": media_global,
        "std_global": std_global
    }


def calcular_persistencia_espacial_por_ventana(ruta_masks, salida_dir, dimensiones_entrada, estado,
                                               tamanos_ventana=[64, 128, 256, 512], grid_size=20, num_procesos=4):
    try:
        archivos = sorted([f for f in os.listdir(ruta_masks) if f.endswith(('.png', '.jpg'))])
        total_frames = len(archivos)
        if total_frames == 0:
            estado.emitir_etapa("[Aviso] No se encontraron máscaras.")
            return

        ancho, alto = dimensiones_entrada

        for ventana in tamanos_ventana:
            estado.emitir_etapa(f"Procesando persistencia con ventana de {ventana} frames")
            tareas = []
            for i in range(0, total_frames, ventana):
                window_archivos = archivos[i:i+ventana]
                nombre = f"ventana_{i:05d}"
                tareas.append((nombre, window_archivos, ruta_masks, ancho, alto, grid_size))

            resultados = {}
            with multiprocessing.Pool(processes=num_procesos) as pool:
                for idx, resultado in enumerate(pool.imap(worker_persistencia, tareas)):
                    ventana_id, datos = resultado
                    resultados[ventana_id] = datos
                    porcentaje = int((idx / len(tareas)) * 100)
                    estado.emitir_progreso(porcentaje)

            nombre_archivo = f"persistencia_{ventana}.json"
            ruta_out = os.path.join(salida_dir, nombre_archivo)
            with open(ruta_out, "w") as f_out:
                json.dump(resultados, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_persistencia_espacial_por_ventana: {str(e)}")


# ---------------------- Estadístico Máscara: Dispersión temporal ----------------------
def worker_dispersion(args):
    ventana_id, archivos, ruta_masks, ancho, alto, grid_size = args
    filas = columnas = grid_size
    celda_h = alto // filas
    celda_w = ancho // columnas

    celdas_ocupadas = np.zeros((filas, columnas), dtype=np.uint8)

    for nombre_archivo in archivos:
        ruta = os.path.join(ruta_masks, nombre_archivo)
        mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = cv2.resize(mask, (ancho, alto), interpolation=cv2.INTER_NEAREST)
        binaria = (mask > 0).astype(np.uint8)

        for i in range(filas):
            for j in range(columnas):
                y1, y2 = i * celda_h, alto if i == filas - 1 else (i + 1) * celda_h
                x1, x2 = j * celda_w, ancho if j == columnas - 1 else (j + 1) * celda_w
                sub = binaria[y1:y2, x1:x2]
                if np.any(sub):
                    celdas_ocupadas[i, j] = 1

    total_activas = int(celdas_ocupadas.sum())
    total_posibles = filas * columnas
    porcentaje = round(total_activas / total_posibles, 3)

    return ventana_id, {
        "celdas_activas": total_activas,
        "porcentaje": porcentaje
    }


def calcular_dispersion_temporal_por_ventana(ruta_masks, salida_dir, dimensiones_entrada, estado,
                                             tamanos_ventana=[64, 128, 256, 512], grid_size=10, num_procesos=4):
    try:
        archivos = sorted([f for f in os.listdir(ruta_masks) if f.endswith(('.png', '.jpg'))])
        total_frames = len(archivos)
        if total_frames == 0:
            estado.emitir_etapa("[Aviso] No se encontraron máscaras.")
            return

        ancho, alto = dimensiones_entrada

        for ventana in tamanos_ventana:
            estado.emitir_etapa(f"Procesando dispersión con ventana de {ventana} frames")
            tareas = []
            for i in range(0, total_frames, ventana):
                nombre = f"ventana_{i:05d}"
                subset = archivos[i:i+ventana]
                tareas.append((nombre, subset, ruta_masks, ancho, alto, grid_size))

            resultados = {}
            with multiprocessing.Pool(processes=num_procesos) as pool:
                for idx, resultado in enumerate(pool.imap(worker_dispersion, tareas)):
                    clave, datos = resultado
                    resultados[clave] = datos
                    porcentaje = int((idx / len(tareas)) * 100)
                    estado.emitir_progreso(porcentaje)

            nombre_archivo = f"dispersion_{ventana}.json"
            ruta_out = os.path.join(salida_dir, nombre_archivo)
            with open(ruta_out, "w") as f_out:
                json.dump(resultados, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_dispersion_temporal_por_ventana: {str(e)}")

# ---------------------- Estadístico Máscaras: Entropía Espacial ----------------------
def worker_entropia_binaria(args):
    ventana_id, archivos, ruta_masks, ancho, alto, grid_size = args
    filas = columnas = grid_size
    celda_h = alto // filas
    celda_w = ancho // columnas

    conteo = np.zeros((filas, columnas), dtype=np.int32)
    total_frames = len(archivos)

    for nombre_archivo in archivos:
        ruta = os.path.join(ruta_masks, nombre_archivo)
        mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        mask = cv2.resize(mask, (ancho, alto), interpolation=cv2.INTER_NEAREST)
        binaria = (mask > 0).astype(np.uint8)

        for i in range(filas):
            for j in range(columnas):
                y1, y2 = i * celda_h, alto if i == filas - 1 else (i + 1) * celda_h
                x1, x2 = j * celda_w, ancho if j == columnas - 1 else (j + 1) * celda_w
                sub = binaria[y1:y2, x1:x2]
                if np.any(sub):
                    conteo[i, j] += 1

    # Normalización
    p = conteo.astype(np.float32) / total_frames
    p = p[p > 0]  # eliminar ceros para el log

    entropia = -np.sum(p * np.log2(p)) if p.size > 0 else 0.0
    entropia_max = np.log2(grid_size * grid_size)
    entropia_norm = entropia / entropia_max if entropia_max > 0 else 0.0
    return ventana_id, {"entropia": float(round(entropia_norm, 4))}


def calcular_entropia_binaria_por_ventana(ruta_masks, salida_dir, dimensiones_entrada, estado,
                                          tamanos_ventana=[64, 128, 256, 512], grid_size=30, num_procesos=4):
    try:
        archivos = sorted([f for f in os.listdir(ruta_masks) if f.endswith(('.png', '.jpg'))])
        total_frames = len(archivos)
        if total_frames == 0:
            estado.emitir_etapa("[Aviso] No se encontraron máscaras.")
            return

        ancho, alto = dimensiones_entrada

        for ventana in tamanos_ventana:
            estado.emitir_etapa(f"Procesando entropía binaria con ventana de {ventana} frames")
            tareas = []
            for i in range(0, total_frames, ventana):
                nombre = f"ventana_{i:05d}"
                subset = archivos[i:i+ventana]
                tareas.append((nombre, subset, ruta_masks, ancho, alto, grid_size))

            resultados = {}
            with multiprocessing.Pool(processes=num_procesos) as pool:
                for idx, resultado in enumerate(pool.imap(worker_entropia_binaria, tareas)):
                    clave, datos = resultado
                    resultados[clave] = datos
                    porcentaje = int((idx / len(tareas)) * 100)
                    estado.emitir_progreso(porcentaje)

            nombre_archivo = f"entropia_binaria_{ventana}.json"
            ruta_out = os.path.join(salida_dir, nombre_archivo)
            with open(ruta_out, "w") as f_out:
                json.dump(resultados, f_out, indent=2)

    except Exception as e:
        estado.emitir_error(f"Error en calcular_entropia_binaria_por_ventana: {str(e)}")



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

                if "velocidad_centroide" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Velocidad del centroide grupal")
                    ruta_centroide = os.path.join(salida_stats, "centroide_grupal.json")
                    ruta_salida = os.path.join(salida_stats, "velocidad_centroide.json")
                    calcular_velocidad_centroide(
                        ruta_centroide_json=ruta_centroide,
                        salida_path=ruta_salida,
                        estado=estado
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

                ruta_masks = os.path.join(carpeta_video, "masks")
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
                    print(f"[INFO] Procesando histograma de densidad para {nombre_video}...")
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

                if "centro_masa_grupo" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Centro de masa global (máscaras)")
                    ruta_salida = os.path.join(salida_stats, "centro_masa.json")
                    print(f"[INFO] Procesando centro de masa global para {nombre_video}...")
                    calcular_centro_masa_mascaras(
                        ruta_masks=ruta_masks,
                        salida_path=ruta_salida,
                        dimensiones_entrada=dimensiones_entrada,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "varianza_espacial" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Varianza espacial")
                    ruta_salida_varianza = os.path.join(salida_stats, "varianza_espacial.json")
                    print(f"[INFO] Procesando varianza espacial para {nombre_video}...")
                    calcular_varianza_espacial(
                        ruta_masks=ruta_masks,
                        salida_path=ruta_salida_varianza,
                        dimensiones_entrada=dimensiones_entrada,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "velocidad_grupo" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Velocidad del grupo")
                    print(f"[INFO] Procesando velocidad del grupo para {nombre_video}...")
                    ruta_salida_velocidad = os.path.join(salida_stats, "velocidad_grupo.json")
                    calcular_velocidad_grupo(
                        ruta_masks=ruta_masks,
                        salida_path=ruta_salida_velocidad,
                        dimensiones_entrada=dimensiones_entrada,
                        estado=estado
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "persistencia_zona" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Persistencia espacial por ventanas")
                    print(f"[INFO] Procesando persistencia espacial por ventanas para {nombre_video}...")
                    calcular_persistencia_espacial_por_ventana(
                        ruta_masks=ruta_masks,
                        salida_dir=salida_stats,
                        dimensiones_entrada=dimensiones_entrada,
                        estado=estado,
                        tamanos_ventana=[64, 128, 256, 512],
                        grid_size=20
                    )
                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "dispersion_temporal" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Dispersión temporal por ventanas")

                    calcular_dispersion_temporal_por_ventana(
                        ruta_masks=ruta_masks,
                        salida_dir=salida_stats,
                        dimensiones_entrada=dimensiones_entrada,
                        estado=estado,
                        tamanos_ventana=[64, 128, 256, 512],
                        grid_size=30
                    )

                    contador += 1
                    estado.emitir_progreso(int((contador / total_descriptores) * 100))

                if "entropia_binaria" in estadisticos_seleccionados:
                    estado.emitir_etapa("Calculando: Entropía binaria espacial por ventanas")

                    calcular_entropia_binaria_por_ventana(
                        ruta_masks=ruta_masks,
                        salida_dir=salida_stats,
                        dimensiones_entrada=dimensiones_entrada,
                        estado=estado,
                        tamanos_ventana=[64, 128, 256, 512],
                        grid_size=30
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



    