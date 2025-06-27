import os
import json
import numpy as np
from tqdm import tqdm

INPUT_DIR = "/home/gms/AnemoNAS/prueba_GUI/output_5s"  
OUTPUT_DIR = os.path.join(INPUT_DIR, "moment_inputs")

# ============================
# FUNCIONES DE CARGA POR JSON
# ============================

# --------------BBOX STATS----------------
def cargar_areas_blobs(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["media_areas_blobs", "std_areas_blobs"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos la métrica 'densidad_media' y la de varianza 'std' de cada frame
    media = [v["media"] for v in data.values()]
    std = [v["std"] for v in data.values()]
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
    serie = [np.array(matriz).flatten() for matriz in data["histograma"].values()]
    
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
    serie = [v["agrupacion"] for v in data.values()]
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
    media = [v["densidad_media"] for v in data.values()]
    std = [v["std"] for v in data.values()]
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
    serie = [v["entropia"] for v in data.values()]
    # serie tiene shape (T,)

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
    media = [v["media"] for v in data.values()]
    std = [v["std"] for v in data.values()]
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
    x = [v[0] for v in data.values()]
    y = [v[1] for v in data.values()]
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
    serie = [v for v in data["por_ventana"].values()]
    # serie tiene shape (T/size_ventana,)
    size_ventana = 128
    # Repetimos cada valor de exploración menos el ultimo para expandirlo a 128 frames, hasta que el valor llege a serie.size pero sin excederlo
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
    serie = [v["velocidad"] for v in data.values()]
    # serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]
    # serie tiene shape final (1, T)

    return serie , info_canal
# --------------TRAY STATS----------------

def cargar_velocidades(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["velocidad_media"]
    
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Dentro del apartado "media_por_frame", cada clave es un frame y el valor es la velocidad media
    serie = [v for v in data["media_por_frame"].values()]
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
    serie = [v for v in data["dispersion_por_frame"].values()]
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
    serie = [v for v in data["porcentaje_giros_bruscos"].values()]
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
    media = [v for v in data["media_por_frame"].values()]
    std = [v for v in data["std_por_frame"].values()]
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
    serie = [v for v in data["entropia_por_frame"].values()]
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
    serie = [v for v in data["polarizacion_por_frame"].values()]
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
    for valor_media, valor_std in zip(media[:-1], std[:-1]):
        serie_expandida_media.extend([valor_media] * size_ventana)
        serie_expandida_std.extend([valor_std] * size_ventana)

    # Agregamos una repetición del último valor expandido
    # Nota: Esto toma el último valor agregado, no el original
    if serie_expandida_media:
        serie_expandida_media.append(serie_expandida_media[-1])
        serie_expandida_std.append(serie_expandida_std[-1])
    else:
        # Caso en el que solo había un bloque (media y std con un valor)
        serie_expandida_media.append(media[-1])
        serie_expandida_std.append(std[-1])

    # Unificamos en un array con canal
    serie = np.stack([serie_expandida_media, serie_expandida_std], axis=0).astype(np.float32)
    return serie , info_canal


# --------------MASK STATS----------------
def cargar_varianza_espacial(path):
    # Extraemos la información del canal, solamente los nombres de las variables que van en cada canal
    info_canal = ["varianza_espacial", "std_varianza_espacial"]
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos la métrica 'varianza_espacial' y la de varianza 'std' de cada frame
    varianza = [v["varianza"] for v in data.values()]
    std = [v["std"] for v in data.values()]
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
    serie = [v["entropia"] for v in data.values()]
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
    x = [v[0] for v in data.values()]
    y = [v[1] for v in data.values()]
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
    serie = [np.array(matriz).flatten() for matriz in data["densidad"].values()]
    
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
    serie = [v["porcentaje"] for v in data.values()]
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
    serie = [v["velocidad"] for v in data.values()]
    # serie tiene shape (T,)
    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]
    # serie tiene shape final (1, T)
    return serie,  info_canal


# ============================
# MAPA DESCRIPTORES
# ============================

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

# ============================
# PROCESAMIENTO PRINCIPAL
# ============================

def procesar_video(carpeta):
    try:
        path_base = os.path.join(INPUT_DIR, carpeta)
        arrays = []
        longitudes = []
        info_canales = []

        for ruta_rel, loader in DESCRIPTORES:
            full_path = os.path.join(path_base, ruta_rel)
            array, info_canal = loader(full_path)
            arrays.append(array)
            longitudes.append(array.shape[1])
            info_canales.extend(info_canal)

        longitud_min = min(longitudes)
        arrays = [a[:, :longitud_min] for a in arrays]
        data = np.concatenate(arrays, axis=0)
        
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

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
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

            nombre = f"{carpeta}_{ts_inicio}_{ts_fin}.npz"
            np.savez_compressed(os.path.join(OUTPUT_DIR, nombre), data=bloque)
            
        
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

# ============================
# MAIN LOOP
# ============================

def main():
    carpetas = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    for carpeta in tqdm(sorted(carpetas)):
        procesar_video(carpeta)

if __name__ == "__main__":
    main()
