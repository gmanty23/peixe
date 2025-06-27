import os
import shutil
from tqdm import tqdm

# Configura tus carpetas de entrada y salida
dir_input = "/home/gmanty/code/USCL2-194221-194721"  # <-- CAMBIA esto a tu ruta real
dir_output = "/home/gmanty/code/anotacion_YOLO"  # <-- CAMBIA esto a tu ruta real
os.makedirs(dir_output, exist_ok=True)

# Crear carpeta de salida si no existe
os.makedirs(dir_output, exist_ok=True)

# Recorrer las carpetas dentro de la carpeta input
for subfolder in tqdm(os.listdir(dir_input), desc="Procesando carpetas", unit="carpeta"):
    subfolder_path = os.path.join(dir_input, subfolder)
    
    # Comprobar si es una carpeta
    if os.path.isdir(subfolder_path):
        imagenes_path = os.path.join(subfolder_path, 'imagenes_og')

        # Comprobar si existe la carpeta imagenes_og
        if os.path.exists(imagenes_path):
            images = sorted(os.listdir(imagenes_path))

            # Tomar 1 de cada 10 imágenes
            selected_images = images[::10]


            for img_name in selected_images:
                src_path = os.path.join(imagenes_path, img_name)
                # Añadir el nombre de la subcarpeta al nombre del archivo para evitar conflictos
                dst_name = f"{img_name}"
                dst_path = os.path.join(dir_output, dst_name)
                shutil.copy(src_path, dst_path)

print("Proceso completado.")
