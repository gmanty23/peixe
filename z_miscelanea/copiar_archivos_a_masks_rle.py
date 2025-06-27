import os
import shutil
from tqdm import tqdm

# VARIABLES DE ENTRADA
archivo1 = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852-listo/USCL2-101113-101613/masks_rle/output_dims.json"
archivo2 = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852-listo/USCL2-101113-101613/masks_rle/recorte_morphology.json"
directorio_raiz = "/home/gmanty/code/AnemoNAS/07-12-23/1812-2002/"

def buscar_masks_rle(directorio_raiz):
    """
    Busca todos los directorios masks_rle dentro de directorio_raiz
    """
    rutas_encontradas = []
    for root, dirs, files in os.walk(directorio_raiz):
        for dir_name in dirs:
            if dir_name == "masks_rle":
                path_masks_rle = os.path.join(root, dir_name)
                rutas_encontradas.append(path_masks_rle)
    return rutas_encontradas

def copiar_archivos(rutas_masks_rle, archivo1, archivo2):
    """
    Copia los archivos en cada directorio masks_rle encontrado, mostrando una barra de progreso.
    """
    for path_masks_rle in tqdm(rutas_masks_rle, desc="Copiando archivos", unit="directorio"):
        try:
            shutil.copy(archivo1, path_masks_rle)
            shutil.copy(archivo2, path_masks_rle)
        except Exception as e:
            print(f"\nError al copiar archivos a {path_masks_rle}: {e}")

if __name__ == "__main__":
    rutas_masks_rle = buscar_masks_rle(directorio_raiz)
    if rutas_masks_rle:
        copiar_archivos(rutas_masks_rle, archivo1, archivo2)
        print("Proceso completado.")
    else:
        print("No se encontraron carpetas 'masks_rle'.")
