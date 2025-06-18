import os
import json
import numpy as np
from tqdm import tqdm

from cutie.inference import data
 
 # ---------------------- EstadoProceso ----------------------
class EstadoProceso:
    def __init__(self, cola=None):
        self.cola = cola
        self.on_etapa = None
        self.on_progreso = None
        self.on_error = None
        self.on_total_videos = None
        self.on_video_progreso = None

    def emitir_etapa(self, mensaje):
        if self.on_etapa:
            self.on_etapa(mensaje)
        if self.cola:
            self.cola.put(("etapa", mensaje))

    def emitir_progreso(self, porcentaje):
        if self.on_progreso:
            self.on_progreso(porcentaje)
        if self.cola:
            self.cola.put(("progreso", porcentaje))

    def emitir_error(self, mensaje):
        if self.on_error:
            self.on_error(mensaje)
        if self.cola:
            self.cola.put(("etapa", f"[ERROR] {mensaje}"))

    def emitir_total_videos(self, total):
        if self.on_total_videos:
            self.on_total_videos(total)

    def emitir_video_progreso(self, index):
        if self.on_video_progreso:
            self.on_video_progreso(index)
            
            
            
            
            
# ============================
# GENERAR INPUT PARA MOMENT
# ============================
import re

def ordenar_claves(d):
    """
    Devuelve los valores del dict ordenados por el entero contenido en las claves (que pueden tener prefijos como 'frame_0001').
    """
    def extraer_num(clave):
        # Extrae el primer número que encuentre en la clave
        match = re.search(r'\d+', clave)
        if match:
            return int(match.group())
        else:
            raise ValueError(f"No se encontró número en la clave: {clave}")

    return [d[k] for k in sorted(d.keys(), key=extraer_num)]


# -------------- FUNCIONES DE CARGA BBOX STATS----------------
def cargar_areas_blobs(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["media_areas_blobs", "std_areas_blobs"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos la métrica 'densidad_media' y la de varianza 'std' de cada frame
    valores_ordenados = ordenar_claves(data)
    media = [v["media"] for v in valores_ordenados]
    std = [v["std"] for v in valores_ordenados]
    # cada uno tiene shape (T,)

    # Convertimos a array numpy y unificamos como canales
    serie = np.stack([media, std], axis=0).astype(np.float32)
    # serie tiene shape final (2, T)
    

    return serie , info_canal

def cargar_distribucion_espacial(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["distribucion_espacial_1_1", "distribucion_espacial_1_2", "distribucion_espacial_1_3", "distribucion_espacial_1_4",
                   "distribucion_espacial_2_1", "distribucion_espacial_2_2", "distribucion_espacial_2_3", "distribucion_espacial_2_4"]
    
    # Abrimos el JSON
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos los valores: lista de longitud T, cada uno es una matriz (5x5)
    # Dimensión intermedia: (T, 2, 4)
    valores_ordenados = ordenar_claves(data["histograma"])
    serie = [np.array(matriz).flatten() for matriz in valores_ordenados]
    
    # Convertimos a array NumPy: (T, 25)
    serie = np.array(serie, dtype=np.float32)


    # Transponemos para tener canales en la primera dimensión: (25, T)
    return serie.T , info_canal


def cargar_coef_agrupacion(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["coef_agrupacion"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
        # data es un dict con T claves (una por frame)
        # Cada valor es un dict con múltiples métricas

    # Extraemos únicamente la métrica 'agrupacion' de cada frame
    valores_ordenados = ordenar_claves(data)
    serie = [v["agrupacion"] for v in valores_ordenados]
    # serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]
    # serie tiene shape final (1, T)
    # el [None, :] añade una dimensión de canal, para que sea compatible con el resto de descriptores

    return serie, info_canal

def cargar_densidad_local(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["densidad_local_media", "densidad_local_std"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos la métrica 'densidad_media' y la de varianza 'std' de cada frame
    valores_ordenados = ordenar_claves(data)
    media = [v["densidad_media"] for v in valores_ordenados]
    std = [v["std"] for v in valores_ordenados]

    # cada uno tiene shape (T,)

    # Convertimos a array numpy y unificamos como canales
    serie = np.stack([media, std], axis=0).astype(np.float32)
    # serie tiene shape final (2, T)

    return serie, info_canal    

def cargar_entropia(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["entropia"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Extraemos únicamente la métrica 'entropia' de cada frame
    valores_ordenados = ordenar_claves(data)
    serie = [v["entropia"] for v in valores_ordenados]   # serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]
    # serie tiene shape final (1, T)

    return serie, info_canal

def cargar_distancia_centroide_grupal(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["distancia_centroide_grupal_media", "distancia_centroide_grupal_std"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos la métrica 'densidad_media' y la de varianza 'std' de cada frame
    valores_ordenados = ordenar_claves(data)
    media = [v["media"] for v in valores_ordenados]
    std = [v["std"] for v in valores_ordenados]
    # cada uno tiene shape (T,)

    # Convertimos a array numpy y unificamos como canales
    serie = np.stack([media, std], axis=0).astype(np.float32)
    # serie tiene shape final (2, T)

    return serie , info_canal

def cargar_centroide_grupal(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["centroide_grupal_x", "centroide_grupal_y"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    # Extraemos la los valores de cada eje
    valores_ordenados = ordenar_claves(data)
    x = [v[0] for v in valores_ordenados]
    y = [v[1] for v in valores_ordenados]
    # cada uno tiene shape (T,)

    # Convertimos a array numpy y unificamos como canales
    serie = np.stack([x, y], axis=0).astype(np.float32)
    # serie tiene shape final (2, T)

    return serie, info_canal

def cargar_exploracion(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["exploracion"]
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    # Extraemos la métrica 'exploracion' de cada frame
    valores_ordenados = ordenar_claves(data["por_ventana"])
    serie = [v for v in valores_ordenados]
    # serie tiene shape (T/size_ventana,)
    size_ventana = data["ventana_frames"]
    # Repetimos cada valor de exploración menos el ultimo para expandirlo a 64 frames, hasta que el valor llege a serie.size pero sin excederlo
    serie_expandida = []
    for valor in serie[:-1]:
        serie_expandida.extend([valor] * size_ventana)
    # Repetimos el último valor de la propia serie expandida una vez para completar la longitud
    # Ahora serie_expandida tiene shape (T,)
    # Convertimos a array numpy y añadimos eje de canal
    serie_expandida = np.array(serie_expandida, dtype=np.float32)[None, :]
    # serie_expandida tiene shape final (1, T)
    return serie_expandida , info_canal

def cargar_velocidad_centroide(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["velocidad_centroide"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Extraemos únicamente la métrica 'entropia' de cada frame
    valores_ordenados = ordenar_claves(data)
    serie = [v["velocidad"] for v in valores_ordenados]
    # serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]
    # serie tiene shape final (1, T)

    return serie , info_canal
# -------------- FUNCIONES DE CARGA TRAY STATS ----------------

def cargar_velocidades(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["velocidad_media"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Dentro del apartado "media_por_frame", cada clave es un frame y el valor es la velocidad media
    valores_ordenados = ordenar_claves(data["media_por_frame"])
    serie = [v for v in valores_ordenados]
    # serie tiene shape (T-1,)
    # Nota: T-1 porque la velocidad se calcula entre frames consecutivos

    # Añadimos una copia de la primera velocidad al inicio para mantener la misma longitud
    serie.insert(0, serie[0]) 
    # Ahora serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]

    return serie , info_canal

def cargar_dispersion_velocidades(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["dispersion_velocidades"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Dentro del apartado "media_por_frame", cada clave es un frame y el valor es la velocidad media
    valores_ordenados = ordenar_claves(data["dispersion_por_frame"])
    serie = [v for v in valores_ordenados]
    # serie tiene shape (T-1,)
    # Nota: T-1 porque la velocidad se calcula entre frames consecutivos

    # Añadimos una copia de la primera dispersion al inicio para mantener la misma longitud
    serie.insert(0, serie[0])  # Añadimos la primera dispersion al inicio
    # Ahora serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]

    return serie , info_canal

def cargar_porcentaje_giros(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["porcentaje_giros_bruscos"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Dentro del apartado "media_por_frame", cada clave es un frame y el valor es la velocidad media
    valores_ordenados = ordenar_claves(data["porcentaje_giros_bruscos"])
    serie = [v for v in valores_ordenados]
    # serie tiene shape (T-2)
    # Nota: T-2 porque el porcentaje de giros bruscos se calcula usando frames consecutivos y el frame actual

    # Añadimos una dos copias del primer porcentaje de giros al inicio para mantener la misma longitud
    serie.insert(0, serie[0])  # Añadimos la primera dispersion al inicio
    serie.insert(0, serie[0])  # Añadimos la primera dispersion al inicio
    # Ahora serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]

    return serie, info_canal

def cargar_media_y_std_giros(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["media_giros", "std_giros"]
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Dentro del apartado "media_por_frame", cada clave es un frame y el valor es la velocidad media
    valores_ordenados = ordenar_claves(data["media_por_frame"])
    media = [v for v in valores_ordenados]
    std = [v for v in ordenar_claves(data["std_por_frame"])]
    # serie tiene shape (T-2)
    # Nota: T-2 porque el porcentaje de giros bruscos se calcula usando frames consecutivos y el frame actual

    # Añadimos una dos copias del primer porcentaje de giros al inicio para mantener la misma longitud
    media.insert(0, media[0])  # Añadimos la primera media al inicio
    media.insert(0, media[0])  # Añadimos la primera media al inicio
    std.insert(0, std[0])  # Añadimos la primera std al inicio
    std.insert(0, std[0])  # Añadimos la primera std al inicio
    # Ahora media tiene shape (T,)

    # Convertimos a array numpy y unificamos como canales
    serie = np.stack([media, std], axis=0).astype(np.float32)
    # serie tiene shape final (2, T)

    return serie , info_canal

def cargar_entropia_direcciones(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["entropia_direcciones"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Dentro del apartado "entropia_por_frame", cada clave es un frame y el valor es la velocidad media
    valores_ordenados = ordenar_claves(data["entropia_por_frame"])
    serie = [v for v in valores_ordenados]
    # serie tiene shape (T-1,)
    # Nota: T-1 porque la velocidad se calcula entre frames consecutivos

    # Añadimos una copia de la primera dispersion al inicio para mantener la misma longitud
    serie.insert(0, serie[0])
    # Ahora serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]

    return serie , info_canal

def cargar_polarizacion(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["polarizacion"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Dentro del apartado "polarizacion_por_frame", cada clave es un frame y el valor es la velocidad media
    valores_ordenados = ordenar_claves(data["polarizacion_por_frame"])
    serie = [v for v in valores_ordenados]
    # serie tiene shape (T-1,)
    # Nota: T-1 porque la velocidad se calcula entre frames consecutivos

    # Añadimos una copia de la primera dispersion al inicio para mantener la misma longitud
    serie.insert(0, serie[0])
    # Ahora serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]

    return serie, info_canal

def cargar_persistencia(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["persistencia"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos las métricas
    media = data["por_ventana"]["64"]["media"]
    std = data["por_ventana"]["64"]["std"]


    size_ventana = 64
    serie_expandida_media = []
    serie_expandida_std = []

    # Expandimos cada valor menos el último
    # for valor_media, valor_std in zip(media[:-1], std[:-1]):
    #     serie_expandida_media.extend([valor_media] * size_ventana)
    #     serie_expandida_std.extend([valor_std] * size_ventana)
    for valor_media in media[:-1]:
        serie_expandida_media.extend([valor_media] * size_ventana)
    for valor_std in std[:-1]:
        serie_expandida_std.extend([valor_std] * size_ventana)


    # Unificamos en un array con canal
    serie = np.stack([serie_expandida_media, serie_expandida_std], axis=0).astype(np.float32)
    return serie , info_canal


# -------------- FUNCIONES DE CARGA MASK STATS- ---------------
def cargar_varianza_espacial(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["varianza_espacial", "std_varianza_espacial"]
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos la métrica 'varianza_espacial' y la de varianza 'std' de cada frame
    valores_ordenados = ordenar_claves(data)
    varianza = [v["varianza"] for v in valores_ordenados]
    std = [v["std"] for v in valores_ordenados]
    # cada uno tiene shape (T,)

    # Convertimos a array numpy y unificamos como canales
    serie = np.stack([varianza, std], axis=0).astype(np.float32)
    # serie tiene shape final (2, T)

    return serie , info_canal



def cargar_entropia_binaria(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["entropia_binaria"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos la métrica 'entropia_binaria' de cada frame
    valores_ordenados = ordenar_claves(data)
    serie = [v["entropia"] for v in valores_ordenados]
    # serie tiene shape (T/size_ventana,)

    size_ventana = 64
    # Repetimos cada valor de entropía menos el ultimo para expandirlo a 64 frames
    serie_expandida = []
    for valor in serie[:-1]:
        serie_expandida.extend([valor] * size_ventana)
    # Repetimos el último valor de la propia serie expandida una vez para completar la longitud
    serie_expandida.append(serie_expandida[-1])
    # Ahora serie_expandida tiene shape (T,)
    # Convertimos a array numpy y añadimos eje de canal
    serie_expandida = np.array(serie_expandida, dtype=np.float32)[None, :]
    # serie_expandida tiene shape final (1, T)
    return serie_expandida , info_canal

def cargar_centro_masa(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["centro_masa_x", "centro_masa_y"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    # Extraemos la los valores de cada eje
    valores_ordenados = ordenar_claves(data)
    x = [v[0] for v in valores_ordenados]
    y = [v[1] for v in valores_ordenados]
    # cada uno tiene shape (T,)

    # Convertimos a array numpy y unificamos como canales
    serie = np.stack([x, y], axis=0).astype(np.float32)
    # serie tiene shape final (2, T)

    return serie, info_canal

def cargar_densidad(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["densidad"]
    
    # Abrimos el JSON
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos los valores: lista de longitud T, cada uno es una matriz (5x5)
    # Dimensión intermedia: (T, 5, 5)
    valores_ordenados = ordenar_claves(data["densidad"])
    serie = [np.array(matriz).flatten() for matriz in valores_ordenados]

    # Convertimos a array NumPy: (T, 25)
    serie = np.array(serie, dtype=np.float32)

    # Transponemos para tener canales en la primera dimensión: (25, T)
    return serie.T , info_canal

def cargar_dispersion_px(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["dispersion_px"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos la métrica 'entropia_binaria' de cada frame
    valores_ordenados = ordenar_claves(data)
    serie = [v["porcentaje"] for v in valores_ordenados]
    # serie tiene shape (T/size_ventana,)

    size_ventana = 64
    # Repetimos cada valor de entropía menos el ultimo para expandirlo a 64 frames
    serie_expandida = []
    for valor in serie[:-1]:
        serie_expandida.extend([valor] * size_ventana)
    # Repetimos el último valor de la propia serie expandida una vez para completar la longitud
    serie_expandida.append(serie_expandida[-1])
    # Ahora serie_expandida tiene shape (T,)
    # Convertimos a array numpy y añadimos eje de canal
    serie_expandida = np.array(serie_expandida, dtype=np.float32)[None, :]
    # serie_expandida tiene shape final (1, T)
    return serie_expandida, info_canal

def cargar_velocidad_grupo(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["velocidad_grupo"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    # Extraemos la métrica 'velocidad_grupo' de cada frame
    valores_ordenados = ordenar_claves(data)
    serie = [v["velocidad"] for v in valores_ordenados]
    # serie tiene shape (T,)
    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]
    # serie tiene shape final (1, T)
    return serie,  info_canal


# ------------------ LISTA DE DESCRIPTORES ----------------

DESCRIPTORES = [
    ("bbox_stats/distribucion_espacial_2_4.json", cargar_distribucion_espacial),
    ("bbox_stats/areas_blobs.json", cargar_areas_blobs),
    ("bbox_stats/coef_agrupacion.json", cargar_coef_agrupacion),
    ("bbox_stats/densidad_local.json", cargar_densidad_local),
    ("bbox_stats/entropia.json", cargar_entropia),
    ("bbox_stats/distancia_centroide_grupal.json", cargar_distancia_centroide_grupal),
    ("bbox_stats/centroide_grupal.json", cargar_centroide_grupal),
    ("bbox_stats/exploracion.json", cargar_exploracion),
    ("bbox_stats/velocidad_centroide.json", cargar_velocidad_centroide),
    ("trayectorias_stats/velocidades.json", cargar_velocidades),
    ("trayectorias_stats/dispersion_velocidades.json", cargar_dispersion_velocidades),
    ("trayectorias_stats/angulo_cambio_direccion.json", cargar_porcentaje_giros),
    ("trayectorias_stats/angulo_cambio_direccion.json", cargar_media_y_std_giros),
    ("trayectorias_stats/direcciones.json", cargar_entropia_direcciones),
    ("trayectorias_stats/direcciones.json", cargar_polarizacion),
    ("trayectorias_stats/persistencia_espacial.json", cargar_persistencia),
    ("mask_stats/varianza_espacial.json", cargar_varianza_espacial),
    ("mask_stats/entropia_binaria_64.json", cargar_entropia_binaria),
    ("mask_stats/centro_masa.json", cargar_centro_masa),
    ("mask_stats/densidad_2_4.json", cargar_densidad),
    ("mask_stats/dispersion_64.json", cargar_dispersion_px),
    ("mask_stats/velocidad_grupo.json", cargar_velocidad_grupo),
    
]

# ------------------ PROCESAMIENTO PRINCIPAL ----------------
def procesar_video(carpeta,input_dir, output_dir):
    try:
        path_base = os.path.join(input_dir, carpeta)
        arrays = []
        longitudes = []
        info_canales = []

        for ruta_rel, loader in DESCRIPTORES:
            full_path = os.path.join(path_base, ruta_rel)
            array, info_canal = loader(full_path)
            arrays.append(array)
            longitudes.extend([array.shape[1]] * array.shape[0])
            info_canales.extend(info_canal)

        # Almacenamos en un json cada canal con su información
        # for i, (array, info_canal) in enumerate(zip(arrays, info_canales)):
        #     canal_info = {
        #         f"canal_{i}": {
        #             "shape": array.shape,
        #             "descripcion": info_canal,
        #             "longitud": array.shape[1],
        #             "media": array.mean(axis=1).tolist(),
        #             "std": array.std(axis=1).tolist(),
        #             "valores": array.tolist()
        #         }
        #     }
        #     canal_json_path = os.path.join(output_dir, f"{carpeta}_canal_{i}_info.json")
        #     with open(canal_json_path, "w") as f:
        #         json.dump(canal_info, f, indent=4)
        # canal_8 = arrays[7]  # Asumiendo que el canal 8 es el octavo en la lista
        # canal_8_info = {
        #     "canal_8": {
        #         "shape": canal_8.shape,
        #         "descripcion": "Distribución espacial 2x4",
        #         "longitud": canal_8.shape[1],
        #         "media": canal_8.mean(axis=1).tolist(),
        #         "std": canal_8.std(axis=1).tolist(),
        #         "valores": canal_8.tolist()
        #     }
        # }
        # canal_8_json_path = os.path.join(output_dir, f"{carpeta}_canal_8_info.json")
        # with open(canal_8_json_path, "w") as f:
        #     json.dump(canal_8_info, f, indent=4)    
        # canal_idx = 0
        # for array in arrays:
        #     for ch_idx in range(array.shape[0]):
        #         canal = array[ch_idx]
        #         nombre = info_canales[canal_idx]
        #         # Guardar JSON con info de este canal
        #         canal_info = {
        #             f"canal_{canal_idx}": {
        #                 "shape": canal.shape,
        #                 "descripcion": nombre,
        #                 "longitud": canal.shape[0],
        #                 "media": float(canal.mean()),
        #                 "std": float(canal.std()),
        #                 "valores": canal.tolist()
        #             }
        #         }
        #         canal_json_path = os.path.join(output_dir, f"{carpeta}_canal_{canal_idx}_info.json")
        #         with open(canal_json_path, "w") as f:
        #             json.dump(canal_info, f, indent=4)
                
        #         canal_idx += 1

        

        
        print("Longitudes de los arrays:", longitudes)
        

        longitud_min = min(longitudes)
        arrays = [a[:, :longitud_min] for a in arrays]
        data = np.concatenate(arrays, axis=0)

        # Calcular siguiente múltiplo de 512
        siguiente_multiplo = ((longitud_min // 512) + 1) * 512
        diferencia = siguiente_multiplo - longitud_min

        # Si la diferencia es menor o igual a 5, repetimos últimos valores hasta completar
        if diferencia <= 5:
            print(f"⚠️ Aplicando padding: añadiendo {diferencia} frames repetidos para completar {siguiente_multiplo}")
            # Tomamos los últimos valores y los repetimos
            ultimos_valores = data[:, -1:]  # shape: (C, 1)
            padding = np.repeat(ultimos_valores, diferencia, axis=1)  # shape: (C, diferencia)
            data = np.concatenate([data, padding], axis=1)
            longitud_min = data.shape[1]
            usable = longitud_min  # ahora el usable es el siguiente múltiplo directamente
        else:
            usable = longitud_min - (longitud_min % 512)

        bloques = np.stack(np.split(data[:, :usable], usable // 512, axis=1))
        
        # Creamos el json de informacion de canales. Recorrera info_canales y guardar,a el indice y la descripcion del canal correspondiente 
        info_json = {
            "info_canales": {
                f"canal_{i}": desc for i, desc in enumerate(info_canales)
            },
            "shape_final": bloques.shape,
            "usable_frames": usable,
            "bloques": bloques.shape[0],
            "longitud_total": longitud_min
        }
  

        os.makedirs(output_dir, exist_ok=True)
        
        fps = 25
        for i, bloque in enumerate(bloques):
            # Calcula inicio y fin en frames
            frame_inicio = i * 512
            frame_fin = frame_inicio + 512 - 1

            # Calcula tiempo
            tiempo_inicio = frame_inicio / fps
            tiempo_fin = frame_fin / fps

            # Formatea timestamp: mmss
            def format_t(t):
                m = int(t // 60)
                s = int(t % 60)
                return f"{m:02d}m{s:02d}s"

            ts_inicio = format_t(tiempo_inicio)
            ts_fin = format_t(tiempo_fin)

            # Encuentra la fecha del video
            fecha = os.path.basename(os.path.dirname(input_dir))
            fecha = fecha.replace("-", "")

            nombre = f"{fecha}_{carpeta}_{ts_inicio}_{ts_fin}.npz"
            np.savez_compressed(os.path.join(output_dir, nombre), data=bloque)

            # Guardamos el JSON de información
            info_json_path = os.path.join(output_dir, f"{fecha}_{carpeta}_info.json")
            with open(info_json_path, "w") as f:
                json.dump(info_json, f, indent=4) 
            
        
                # ==== LOG DETALLADO ====
        print(f"✅ {carpeta}")
        print(f"   - Canales totales: {data.shape[0]}")
        print(f"   - Longitud total: {data.shape[1]}")
        print(f"   - Usable (múltiplo de 512): {usable}")
        print(f"   - Bloques: {bloques.shape[0]}")
        print(f"   - Shape final: {bloques.shape}  (bloques, canales, 512)")
        print()

    except Exception as e:
        print(f"❌ Error en {carpeta}: {e}")
        print("Longitudes de los arrays:", longitudes)



def generar_inputs_moment(input_dir, estado = None):
    output_dir = os.path.join(input_dir, "moment_inputs")
    carpetas = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if estado:
        estado.emitir_total_videos(len(carpetas))
    
    for idx, carpeta in enumerate(sorted(carpetas)):
        if carpeta.startswith("moment"):
            if estado:
                estado.emitir_video_progreso(idx+1)
            continue
        if estado:
            estado.emitir_etapa(f"Procesando: {carpeta}")
        procesar_video(carpeta, input_dir, output_dir)
        if estado:
            estado.emitir_video_progreso(idx+1)


