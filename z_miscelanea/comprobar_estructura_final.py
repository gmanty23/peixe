import os
import glob
from tqdm import tqdm
from colorama import init, Fore, Style

# Inicializa colorama
init(autoreset=True)

def comprobar_estructura(path_trabajo):
    # Obtiene la lista de carpetas dentro del path de trabajo
    carpetas = [f for f in os.listdir(path_trabajo) if os.path.isdir(os.path.join(path_trabajo, f))]

    # Recorre las carpetas con barra de progreso
    for carpeta in tqdm(carpetas, desc="Comprobando carpetas"):
        carpeta_path = os.path.join(path_trabajo, carpeta)
        
        # Define lo que debe contener
        estructura = {
            "bbox": False,
            "bbox_stats": False,
            "mask_stats": False,
            "masks_rle": False,
            "trayectorias_stats": False,
            "video": False
        }

        # Comprueba existencia y contenido de cada carpeta
        for subcarpeta in estructura.keys():
            if subcarpeta == "video":
                # Buscar único archivo .mp4
                mp4_files = glob.glob(os.path.join(carpeta_path, "*.mp4"))
                if len(mp4_files) == 1:
                    estructura["video"] = True
            else:
                subcarpeta_path = os.path.join(carpeta_path, subcarpeta)
                if os.path.isdir(subcarpeta_path) and os.listdir(subcarpeta_path):
                    estructura[subcarpeta] = True

        # Si alguna parte de la estructura falla, imprime
        if not all(estructura.values()):
            print(f"{Fore.RED}✗ {carpeta}{Style.RESET_ALL}")

if __name__ == "__main__":
    # Cambia este path por el path de trabajo que desees
    path_trabajo = "/mnt/d/14-12-23/"
    comprobar_estructura(path_trabajo)
