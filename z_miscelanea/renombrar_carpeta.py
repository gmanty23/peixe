import os

# === Configuración ===
ruta_carpeta = '/home/gms/AnemoNAS/POR_DIA/11-12-2023/'  
prefijo = '11_'
# =====================

def renombrar_contenido_carpeta(ruta_carpeta, prefijo='11_'):
    if not os.path.isdir(ruta_carpeta):
        print(f"La ruta '{ruta_carpeta}' no es una carpeta válida.")
        return

    for nombre in os.listdir(ruta_carpeta):
        ruta_original = os.path.join(ruta_carpeta, nombre)
        nuevo_nombre = prefijo + nombre
        ruta_nueva = os.path.join(ruta_carpeta, nuevo_nombre)

        # Evita renombrar si el nombre ya empieza con el prefijo
        if nombre.startswith(prefijo):
            continue

        try:
            os.rename(ruta_original, ruta_nueva)
            print(f"Renombrado: {nombre} -> {nuevo_nombre}")
        except Exception as e:
            print(f"Error al renombrar '{nombre}': {e}")

if __name__ == '__main__':
    renombrar_contenido_carpeta(ruta_carpeta, prefijo)
