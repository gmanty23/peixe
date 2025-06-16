import os
import numpy as np
from tqdm import tqdm

INPUT_DIR = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/moment_inputs"  # <- CAMBIA esto por la ruta real
EXPECTED_CHANNELS = 41
EXPECTED_BLOCK_SIZE = 512

DESCRIPTORES = [
    # BBox stats (31 canales)
    ("distribucion_espacial_5.json", 25),
    ("coef_agrupacion.json", 1),
    ("densidad_local.json", 1),
    ("entropia.json", 1),
    ("distancia_centroide_grupal.json", 1),
    ("centroide_grupal.json", 2),

    # Trayectorias stats (6 canales)
    ("velocidades.json", 1),
    ("dispersion_velocidades.json", 1),
    ("angulo_cambio_direccion.json (porcentaje_giros_bruscos)", 1),
    ("angulo_cambio_direccion.json (media_por_frame)", 1),
    ("direcciones.json (entropia_por_frame)", 1),
    ("direcciones.json (polarizacion_por_frame)", 1),

    # Window descriptors (4 canales)
    ("varianza_espacial.json", 1),
    ("entropia_binaria_64.json", 1),
    ("persistencia_64.json", 1),
    ("exploracion.json", 1),
]

def verificar_npz(path):
    with np.load(path) as data:
        arr = data["data"]
        errores = []
        if arr.ndim != 3:
            errores.append(f"  - Error: dimensión incorrecta {arr.shape}")
        else:
            b, c, t = arr.shape
            if c != EXPECTED_CHANNELS:
                errores.append(f"  - Error: se esperaban {EXPECTED_CHANNELS} canales, pero hay {c}")
            if t != EXPECTED_BLOCK_SIZE:
                errores.append(f"  - Error: se esperaban bloques de {EXPECTED_BLOCK_SIZE} frames, pero hay {t}")
        return errores

def main():
    archivos = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]
    print("Verificando estructura de archivos .npz...\n")
    print("Esperados:")
    for nombre, cantidad in DESCRIPTORES:
        print(f"  {nombre} -> {cantidad} canales")
    print(f"\nTOTAL esperado: {EXPECTED_CHANNELS} canales por bloque\n")

    for archivo in tqdm(archivos):
        ruta = os.path.join(INPUT_DIR, archivo)
        errores = verificar_npz(ruta)
        if errores:
            print(f"\nErrores en {archivo}:")
            for err in errores:
                print(err)

if __name__ == "__main__":
    main()
