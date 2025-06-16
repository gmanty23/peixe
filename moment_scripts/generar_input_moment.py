import os
import json
import numpy as np
from tqdm import tqdm

INPUT_DIR = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852"  
OUTPUT_DIR = os.path.join(INPUT_DIR, "moment_inputs")

# ============================
# FUNCIONES DE CARGA POR JSON
# ============================

def cargar_distribucion_espacial_5(path):
    # Abrimos el JSON
    with open(path, "r") as f:
        data = json.load(f)

    # Extraemos los valores: lista de longitud T, cada uno es una matriz (5x5)
    # Dimensión intermedia: (T, 5, 5)
    serie = [np.array(matriz).flatten() for matriz in data["histograma"].values()]
    
    # Convertimos a array NumPy: (T, 25)
    serie = np.array(serie, dtype=np.float32)

    # Transponemos para tener canales en la primera dimensión: (25, T)
    return serie.T


def cargar_coef_agrupacion(path):
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

    return serie

def cargar_densidad_local(path):
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

    return serie

def cargar_entropia(path):
    # Abrimos el archivo JSON y cargamos los datos
    with open(path, "r") as f:
        data = json.load(f)
    
    # Extraemos únicamente la métrica 'entropia' de cada frame
    serie = [v["entropia"] for v in data.values()]
    # serie tiene shape (T,)

    # Convertimos a array numpy y añadimos eje de canal
    serie = np.array(serie, dtype=np.float32)[None, :]
    # serie tiene shape final (1, T)

    return serie

def cargar_distancia_centroide_grupal(path):
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

    return serie

def cargar_centroide_grupal(path):
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

    return serie

def cargar_velocidades(path):
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

    return serie

def cargar_dispersion_velocidades(path):
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

    return serie

def cargar_porcentaje_giros(path):
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

    return serie

def cargar_media_y_std_giros(path):
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

    return serie

def cargar_entropia_direcciones(path):
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

    return serie

def cargar_polarizacion(path):
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

    return serie

def cargar_varianza_espacial(path):
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

    return serie



def cargar_entropia_binaria(path):
    # Abrimos el archivo JSON y cargamos los datos
    with open(path) as f:
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
    return serie_expandida

def cargar_exploracion(path):
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
    return serie_expandida


# ============================
# MAPA DESCRIPTORES
# ============================

DESCRIPTORES = [
    ("bbox_stats/distribucion_espacial_5.json", cargar_distribucion_espacial_5),
    ("bbox_stats/coef_agrupacion.json", cargar_coef_agrupacion),
    ("bbox_stats/densidad_local.json", cargar_densidad_local),
    ("bbox_stats/entropia.json", cargar_entropia),
    ("bbox_stats/distancia_centroide_grupal.json", cargar_distancia_centroide_grupal),
    ("bbox_stats/centroide_grupal.json", cargar_centroide_grupal),
    ("trayectorias_stats/velocidades.json", cargar_velocidades),
    ("trayectorias_stats/dispersion_velocidades.json", cargar_dispersion_velocidades),
    ("trayectorias_stats/angulo_cambio_direccion.json", cargar_porcentaje_giros),
    ("trayectorias_stats/angulo_cambio_direccion.json", cargar_media_y_std_giros),
    ("trayectorias_stats/direcciones.json", cargar_entropia_direcciones),
    ("trayectorias_stats/direcciones.json", cargar_polarizacion),
    ("mask_stats/varianza_espacial.json", cargar_varianza_espacial),
    ("mask_stats/entropia_binaria_64.json", cargar_entropia_binaria),
    ("bbox_stats/exploracion.json", cargar_exploracion),
]

# ============================
# PROCESAMIENTO PRINCIPAL
# ============================

def procesar_video(carpeta):
    try:
        path_base = os.path.join(INPUT_DIR, carpeta)
        arrays = []
        longitudes = []

        for ruta_rel, loader in DESCRIPTORES:
            full_path = os.path.join(path_base, ruta_rel)
            array = loader(full_path)
            arrays.append(array)
            longitudes.append(array.shape[1])

        longitud_min = min(longitudes)
        arrays = [a[:, :longitud_min] for a in arrays]
        data = np.concatenate(arrays, axis=0)
        usable = longitud_min - (longitud_min % 512)
        bloques = np.stack(np.split(data[:, :usable], usable // 512, axis=1))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f"{carpeta}.npz"), data=bloques)
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
