import os
import shutil
import time
from tqdm import tqdm

def copiar_directorios_con_barra(input_dir, output_dir):
    # Asegurarse de que el directorio de salida existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Recorrer todas las subcarpetas del directorio de entrada
    subcarpetas = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    # Barra de progreso para recorrer las subcarpetas
    pbar = tqdm(total=len(subcarpetas), desc="Copiando directorios", unit="dir")

    start_time = time.time()

    for subcarpeta in subcarpetas:
        subcarpeta_input = os.path.join(input_dir, subcarpeta)
        subcarpeta_output = os.path.join(output_dir, subcarpeta)

        # Verificar si existen las carpetas bbox/ y masks_rle en la subcarpeta
        if os.path.isdir(os.path.join(subcarpeta_input, 'bbox')) and os.path.isdir(os.path.join(subcarpeta_input, 'masks_rle')):
            pbar.set_description(f"Procesando {subcarpeta}")

            # Crear la subcarpeta en el directorio de salida si no existe
            if not os.path.exists(subcarpeta_output):
                os.makedirs(subcarpeta_output)

            # Copiar las carpetas 'bbox' y 'masks_rle' junto con sus archivos
            shutil.copytree(os.path.join(subcarpeta_input, 'bbox'), os.path.join(subcarpeta_output, 'bbox'), dirs_exist_ok=True)
            shutil.copytree(os.path.join(subcarpeta_input, 'masks_rle'), os.path.join(subcarpeta_output, 'masks_rle'), dirs_exist_ok=True)

            # Actualizar barra de progreso
            pbar.update(1)
        else:
            pbar.set_description(f"Saltando {subcarpeta} (falta 'bbox' o 'masks_rle')")

    # Mostrar velocidad de transferencia
    elapsed_time = time.time() - start_time
    total_size = sum(os.path.getsize(os.path.join(input_dir, f, 'bbox')) for f in subcarpetas if os.path.isdir(os.path.join(input_dir, f, 'bbox')))
    speed = total_size / elapsed_time / 1024 / 1024  # Convertir a MB/s
    pbar.set_postfix(speed=f"{speed:.2f} MB/s")

if __name__ == "__main__":
    input_dir = "/media/gms/EXTERNAL_USB/06-12-23/0926-1852"
    output_dir = "/home/gms/AnemoNAS/POR_DIA/06-12-2023"
    copiar_directorios_con_barra(input_dir, output_dir)
