import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import cv2
from collections import defaultdict, deque



# Funciones para cargar y procesar las etiquetas de detección
def parse_yolo_label(path, image_size=1024):
    detections = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            _, x, y, w, h, _ = map(float, parts[:6])  # ignoramos clase e id
            x_center, y_center = x * image_size, y * image_size
            width, height = w * image_size, h * image_size
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            area = width * height
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'centroid': [(x1 + x2) / 2, (y1 + y2) / 2],
                'area': area,
                'track_id': None,
                'start_frame': None
            })
    return detections


# Funcion para cargar todas las detecciones por cada frame
def cargar_detecciones_por_frame(labels_dir, image_size=1024):
    frame_data = {}
    txt_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    for idx, path in enumerate(txt_files):
        frame_name = os.path.splitext(os.path.basename(path))[0]
        detections = parse_yolo_label(path, image_size=image_size)
        for d in detections:
            d['start_frame'] = idx
        detections_filtradas = eliminar_duplicados(detections)
        frame_data[frame_name] = detections_filtradas
    return frame_data
    
# Función para construir el grafo temporal a partir de la informacion por cada frame de los labels generados por el modelo YOLO
# El grafo es un diccionario compuesto por:
# - frame_id: el id del frame
# - id_node: el id del nodo dentro del frame
# - centroid: el centroide del blob
# - bbox: la caja delimitadora del blob
# - area: el área del blob
# - edges: lista de conexiones a nodos en el siguiente frame
# - track_id: el id de la trayectoria asignada al nodo
def construir_grafo_temporal(detecciones_por_frame):
    grafo = {}
    for frame_id, blobs in detecciones_por_frame.items():
        grafo[frame_id] = []
        for i, blob in enumerate(blobs):
            nodo = {
                "id_node": f"{frame_id}_{i}",  # ID único local
                "frame": frame_id,
                "centroid": blob["centroid"],
                "bbox": blob["bbox"],
                "area": blob["area"],
                "edges": [],  
                "track_id": None  
            }
            grafo[frame_id].append(nodo)
    return grafo

# Función para calcular la Intersección sobre la Unión (IoU) entre dos bounding boxes
def calcular_iou(bbox1, bbox2):
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

# Función para calcular el coste de conexión entre dos blobs
# El coste se calcula como una combinación de la distancia euclidiana entre centroides, la iou, la diferencia de área y la diferencia de ratio de aspecto
# Se puede ajustar el peso de cada componente (alpha, beta, gamma, delta) para modificar la importancia relativa
def calcular_coste_conexion(blob1, blob2, pesos, image_size=1024):
    alpha, beta, gamma, delta = pesos

    iou = calcular_iou(blob1["bbox"], blob2["bbox"])
    
    d_centroid = np.linalg.norm(np.array(blob1["centroid"]) - np.array(blob2["centroid"])) / image_size
    area_diff = abs(blob1["area"] - blob2["area"]) / max(blob1["area"], blob2["area"])

    def aspect_ratio(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w / h if h != 0 else 0

    ratio1 = aspect_ratio(blob1["bbox"])
    ratio2 = aspect_ratio(blob2["bbox"])
    aspect_diff = abs(ratio1 - ratio2) / max(ratio1, ratio2) if max(ratio1, ratio2) != 0 else 0

    coste = (
        alpha * (1 - iou) +
        beta * d_centroid +
        gamma * area_diff +
        delta * aspect_diff
    )
    return coste


# Función para calcular el coste de conexión entre dos blobs, considerando la trayectoria pasada
# Esta función es similar a calcular_coste_conexion, pero incluye una predicción de la posición del blob2 basada en la trayectoria pasada de blob1
def calcular_coste_conexion_predicho(blob1, blob2, pesos, trayectoria_pasada, image_size=1024):
    alpha, beta, gamma, delta, epsilon = pesos

    def aspect_ratio(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w / h if h != 0 else 0

    iou = calcular_iou(blob1["bbox"], blob2["bbox"])
    d_centroid = np.linalg.norm(np.array(blob1["centroid"]) - np.array(blob2["centroid"]))/ image_size
    area_diff = abs(blob1["area"] - blob2["area"]) / max(blob1["area"], blob2["area"])
    ratio1 = aspect_ratio(blob1["bbox"])
    ratio2 = aspect_ratio(blob2["bbox"])
    aspect_diff = abs(ratio1 - ratio2) / max(ratio1, ratio2) if max(ratio1, ratio2) != 0 else 0

    if len(trayectoria_pasada) >= 2:
        velocidades = [np.array(trayectoria_pasada[i+1]) - np.array(trayectoria_pasada[i]) for i in range(len(trayectoria_pasada)-1)]
        v_media = np.mean(velocidades, axis=0)
        centro_pred = np.array(blob1["centroid"]) + v_media
        d_pred = np.linalg.norm(centro_pred - np.array(blob2["centroid"]))/image_size
    else:
        d_pred = 0

    coste = (alpha * (1 - iou)) + (beta * d_centroid) + (gamma * area_diff) + (delta * aspect_diff) + (epsilon * d_pred)
    return coste

# Función para reasignar nodos desde un buffer de frames inactivos a nuevos nodos
# Esta función utiliza el algoritmo húngaro para asignar los nodos del buffer a los nuevos nodos
def reasignar_desde_buffer(nuevos_nodos, buffer, historico_centroides, pesos_reasignacion, max_coste):
    if not nuevos_nodos or not buffer:
        return [], nuevos_nodos

    buffer_ids = list(buffer.keys())
    cost_matrix = np.zeros((len(nuevos_nodos), len(buffer_ids)))
    for i, nodo in enumerate(nuevos_nodos):
        for j, track_id in enumerate(buffer_ids):
            nodo_buffer = buffer[track_id]['ultimo_nodo']
            trayectoria = list(historico_centroides[track_id])
            cost_matrix[i, j] = calcular_coste_conexion_predicho(nodo_buffer, nodo, pesos_reasignacion, trayectoria)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    asignaciones = []
    no_asignados = set(range(len(nuevos_nodos)))
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_coste:
            nodo = nuevos_nodos[r]
            track_id = buffer_ids[c]
            nodo['track_id'] = track_id
            asignaciones.append((nodo, track_id))
            historico_centroides[track_id].append(nodo["centroid"])
            buffer.pop(track_id)
            no_asignados.discard(r)

    nodos_restantes = [nuevos_nodos[i] for i in no_asignados]
    return asignaciones, nodos_restantes

# Función para conectar nodos entre frames adyacentes utilizando el algoritmo húngaro y asignar los IDs de las trayectorias
def conectar_nodos_hungaro(grafo, pesos, umbral_coste=np.inf, max_ids_activos=50, max_coste_reasignacion=60, pesos_reasignacion=None):
    frames = sorted(grafo.keys())
    historico_centroides = defaultdict(lambda: deque(maxlen=5))
    buffer_ids = {}
    ids_activos = set()
    pool_ids_disponibles = set(range(max_ids_activos))

    nodos_f0 = grafo[frames[0]]
    for nodo in nodos_f0:
        if pool_ids_disponibles:
            new_id = pool_ids_disponibles.pop()
            nodo["track_id"] = new_id
            historico_centroides[new_id].append(nodo["centroid"])
            ids_activos.add(new_id)

    if pesos_reasignacion is None:
        pesos_reasignacion = pesos

    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i+1]
        nodos_f1 = grafo[f1]
        nodos_f2 = grafo[f2]

        if not nodos_f1 or not nodos_f2:
            continue

        cost_matrix = np.zeros((len(nodos_f1), len(nodos_f2)))
        for r, n1 in enumerate(nodos_f1):
            for c, n2 in enumerate(nodos_f2):
                historial = historico_centroides.get(n1["track_id"], [])
                cost_matrix[r, c] = calcular_coste_conexion_predicho(n1, n2, pesos, list(historial))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        asignados_f2 = set()
        usados_ids = set()
        for r, c in zip(row_ind, col_ind):
            coste = cost_matrix[r, c]
            if coste <= umbral_coste:
                n1 = nodos_f1[r]
                n2 = nodos_f2[c]
                n1["edges"].append({"to": n2["id_node"], "cost": coste})
                n2["track_id"] = n1["track_id"]
                historico_centroides[n2["track_id"]].append(n2["centroid"])
                usados_ids.add(n2["track_id"])
                asignados_f2.add(c)

        for n in nodos_f1:
            if n["track_id"] not in usados_ids:
                buffer_ids[n["track_id"]] = {"ultimo_nodo": n, "frame": f1}
                ids_activos.discard(n["track_id"])
                pool_ids_disponibles.add(n["track_id"])

        nuevos = [nodos_f2[j] for j in range(len(nodos_f2)) if j not in asignados_f2]
        _, nuevos = reasignar_desde_buffer(nuevos, buffer_ids, historico_centroides, pesos_reasignacion, max_coste_reasignacion)

        for nodo in nuevos:
            if pool_ids_disponibles:
                new_id = pool_ids_disponibles.pop()
                nodo["track_id"] = new_id
                historico_centroides[new_id].append(nodo["centroid"])
                ids_activos.add(new_id)



# # Función para asignar IDs a los nodos utilizando un buffer de frames inactivos
# # Se asignan IDs a los nodos en función de su conexión con nodos en frames anteriores
# # Se utiliza un diccionario para llevar un seguimiento de las trayectorias activas
# # y se asignan nuevos IDs a los nodos que no están conectados a ninguna trayectoria activa
# def asignar_ids_con_buffer(grafo, pesos,  umbral_reconexion=np.inf, max_frames_inactivos=15):
#     trayectorias_activas = {}  # track_id -> {ultimo_frame, ultimo_nodo}
#     siguiente_id = 0
#     frames = sorted(grafo.keys())

#     for frame in frames:
#         for nodo in grafo[frame]:
#             if nodo["track_id"] is not None:
#                 continue

#             mejor_coste = np.inf
#             mejor_id = None

#             for track_id, estado in trayectorias_activas.items():
#                 f_ult = int(estado["ultimo_frame"].split('_')[-1])
#                 f_act = int(frame.split('_')[-1])
#                 if f_act - f_ult <= max_frames_inactivos:
#                     coste = calcular_coste_conexion(estado["ultimo_nodo"], nodo, pesos)
#                     if coste < mejor_coste and coste <= umbral_reconexion:
#                         mejor_coste = coste
#                         mejor_id = track_id

#             if mejor_id is not None:
#                 nodo["track_id"] = mejor_id
#                 trayectorias_activas[mejor_id] = {"ultimo_frame": frame, "ultimo_nodo": nodo}
#             else:
#                 nodo["track_id"] = siguiente_id
#                 trayectorias_activas[siguiente_id] = {"ultimo_frame": frame, "ultimo_nodo": nodo}
#                 siguiente_id += 1


# Función para eliminar duplicados basándose en IoU y distancia entre centroides

def eliminar_duplicados(detecciones, iou_thresh=0.4, dist_thresh=20):
    keep = []
    eliminados = set()
    for i in range(len(detecciones)):
        if i in eliminados:
            continue
        d1 = detecciones[i]
        for j in range(i + 1, len(detecciones)):
            if j in eliminados:
                continue
            d2 = detecciones[j]
            iou = calcular_iou(d1['bbox'], d2['bbox'])
            dist = np.linalg.norm(np.array(d1['centroid']) - np.array(d2['centroid']))
            if iou > iou_thresh and dist < dist_thresh:
                if d1['area'] < d2['area']:
                    mas_pequeno, mas_grande = d1, d2
                else:
                    mas_pequeno, mas_grande = d2, d1
                if mas_grande['track_id'] is not None and mas_pequeno['track_id'] is not None:
                    if mas_grande.get('start_frame', 1e9) < mas_pequeno.get('start_frame', 1e9):
                        mas_pequeno['track_id'] = mas_grande['track_id']
                        mas_pequeno['start_frame'] = mas_grande['start_frame']
                elif mas_grande['track_id'] is not None:
                    mas_pequeno['track_id'] = mas_grande['track_id']
                    mas_pequeno['start_frame'] = mas_grande.get('start_frame')
                eliminados.add(detecciones.index(mas_grande))
    for i, d in enumerate(detecciones):
        if i not in eliminados:
            keep.append(d)
    return keep



def fusionar_trayectorias(grafo, umbral_frames=5, max_distancia=50, max_area_dif=0.3, max_aspect_dif=0.3):
    trayectorias = {}
    for frame_id in grafo:
        for nodo in grafo[frame_id]:
            tid = nodo["track_id"]
            if tid not in trayectorias:
                trayectorias[tid] = []
            trayectorias[tid].append(nodo)

    ids = sorted(trayectorias.keys())
    for i, id_a in enumerate(ids):
        if not trayectorias[id_a]:
            continue
        fin_a = max(trayectorias[id_a], key=lambda n: n["frame"])
        frame_fin_a = int(fin_a["frame"].split('_')[-1])

        for id_b in ids[i+1:]:
            if not trayectorias[id_b]:
                continue
            inicio_b = min(trayectorias[id_b], key=lambda n: n["frame"])
            frame_ini_b = int(inicio_b["frame"].split('_')[-1])

            if 0 < frame_ini_b - frame_fin_a <= umbral_frames:
                d_centroid = np.linalg.norm(np.array(fin_a["centroid"]) - np.array(inicio_b["centroid"]))
                area_diff = abs(fin_a["area"] - inicio_b["area"]) / max(fin_a["area"], inicio_b["area"])

                def aspect_ratio(b):
                    w = b[2] - b[0]
                    h = b[3] - b[1]
                    return w / h if h != 0 else 0

                ar_a = aspect_ratio(fin_a["bbox"])
                ar_b = aspect_ratio(inicio_b["bbox"])
                aspect_diff = abs(ar_a - ar_b) / max(ar_a, ar_b) if max(ar_a, ar_b) != 0 else 0

                if d_centroid <= max_distancia and area_diff <= max_area_dif and aspect_diff <= max_aspect_dif:
                    for nodo in trayectorias[id_b]:
                        nodo["track_id"] = id_a
                    trayectorias[id_a].extend(trayectorias[id_b])
                    trayectorias[id_b] = []



def visualizar_conexiones(grafo, frame_id):
    if frame_id not in grafo:
        print("Frame no disponible.")
        return

    nodos = grafo[frame_id]
    fig, ax = plt.subplots(figsize=(8, 8))

    id_colores = {}
    cmap = plt.get_cmap("tab20")

    for nodo in nodos:
        x1, y1, x2, y2 = nodo["bbox"]
        tid = nodo["track_id"]
        if tid not in id_colores:
            id_colores[tid] = cmap(tid % 20)
        color = id_colores[tid]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        cx, cy = nodo['centroid']
        ax.text(cx, cy, f"{tid}", color=color, fontsize=8)

    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.invert_yaxis()
    ax.set_title(f"Bounding boxes en {frame_id}")
    plt.grid(True)
    plt.show()


def visualizar_trayectoria_individual(grafo, track_id_objetivo, image_size=1024):
    puntos = []

    for frame_id in sorted(grafo.keys()):
        for nodo in grafo[frame_id]:
            if nodo["track_id"] == track_id_objetivo:
                x, y = nodo["centroid"]
                puntos.append((int(frame_id.split('_')[-1]), x, y))

    if not puntos:
        print(f"No se encontró la trayectoria para el track_id {track_id_objetivo}")
        return

    puntos = sorted(puntos)
    xs = [p[1] for p in puntos]
    ys = [p[2] for p in puntos]
    frames = [p[0] for p in puntos]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, marker='o', linestyle='-', color='blue')
    for i, (x, y, f) in enumerate(zip(xs, ys, frames)):
        ax.text(x, y, str(f), fontsize=7, color='red')

    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.invert_yaxis()
    ax.set_title(f"Trayectoria de ID {track_id_objetivo}")
    plt.grid(True)
    plt.show()



def guardar_frames_con_ids(grafo, path_imagenes_originales, path_salida, image_size=1024):
    os.makedirs(path_salida, exist_ok=True)

    id_colores = {}
    cmap = plt.get_cmap("tab20")

    for frame_id in sorted(grafo.keys()):
        ruta_img = os.path.join(path_imagenes_originales, f"{frame_id}.jpg")
        if not os.path.exists(ruta_img):
            print(f"[Aviso] Imagen no encontrada: {ruta_img}")
            continue

        img = cv2.imread(ruta_img)
        img = cv2.resize(img, (image_size, image_size))

        for nodo in grafo[frame_id]:
            tid = nodo["track_id"]
            if tid is None:
                continue
            if tid not in id_colores:
                id_colores[tid] = tuple(int(c * 255) for c in cmap(tid % 20)[:3])

            color = id_colores[tid]
            x1, y1, x2, y2 = map(int, nodo["bbox"])
            cx, cy = map(int, nodo["centroid"])

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        output_path = os.path.join(path_salida, f"{frame_id}.jpg")
        cv2.imwrite(output_path, img)

    print(f"✅ Imágenes guardadas en: {path_salida}")


# Código de prueba o ejecución principal
if __name__ == "__main__":

    #labels_path = "/home/gmanty/code/output_20s/tracking/tracking_pez/labels"
    labels_path = "/home/gmanty/code/output_20s/tracking/tracking_pez/labels"
    detecciones_por_frame = cargar_detecciones_por_frame(labels_path)
    grafo_temporal = construir_grafo_temporal(detecciones_por_frame)

    pesos = (0.55, 0.2, 0.05, 0.05, 0.15)  # α, β, γ, δ
    pesos_reasignacion = (0.15, 0.32, 0.11, 0.11, 0.31)  # α, β, γ, δ, ε
    conectar_nodos_hungaro(
        grafo_temporal,
        pesos=pesos,
        umbral_coste=50,
        max_ids_activos=50,
        max_coste_reasignacion=60,
        pesos_reasignacion=pesos_reasignacion
    )

    # fusionar_trayectorias(grafo_temporal, umbral_frames=5, max_distancia=50, max_area_dif=0.3, max_aspect_dif=0.3)


    print("Ejemplo de nodos conectados:")
    for f in list(grafo_temporal.keys())[:1]:
        for nodo in grafo_temporal[f]:
            print(f"{nodo['id_node']} -> {nodo['edges']}")

    # Visualiza conexiones de un frame 
    guardar_frames_con_ids(
        grafo=grafo_temporal,
        path_imagenes_originales="/home/gmanty/code/output_20s/images_yolov8_tapado",
        path_salida="/home/gmanty/code/output_20s/prueba_salida_grafo_ACOTADO_remember",
        image_size=1024
    )


