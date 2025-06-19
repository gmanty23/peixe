import os
from tqdm import tqdm

def reemplazar_nombre_archivos(directorio, string_a_sustituir, string_de_sustitucion):
    # Obtener todos los archivos en el directorio
    archivos = []
    for root, dirs, files in os.walk(directorio):
        for archivo in files:
            archivos.append(os.path.join(root, archivo))
    
    # Barra de progreso
    pbar = tqdm(archivos, desc="Procesando archivos", unit="archivo")

    for archivo_path in pbar:
        # Obtener el nombre del archivo y la ruta
        directorio_archivo = os.path.dirname(archivo_path)
        nombre_archivo = os.path.basename(archivo_path)

        # Verificar si el nombre del archivo contiene el texto a sustituir
        if string_a_sustituir in nombre_archivo:
            # Crear el nuevo nombre de archivo
            nuevo_nombre = nombre_archivo.replace(string_a_sustituir, string_de_sustitucion)

            # Crear la nueva ruta con el nombre modificado
            nuevo_archivo_path = os.path.join(directorio_archivo, nuevo_nombre)

            # Renombrar el archivo
            os.rename(archivo_path, nuevo_archivo_path)

            # Actualizar la barra de progreso
            pbar.set_postfix(archivo=archivo_path, accion="Renombrado")

        else:
            pbar.set_postfix(archivo=archivo_path, accion="No encontrado")

if __name__ == "__main__":
    # Definir las variables aquí
    directorio = "/home/gms/AnemoNAS/POR_DIA/13-12-2023/moment_inputs"  # Path al directorio
    string_a_sustituir = "POR_DIA"  # El texto que se quiere sustituir
    string_de_sustitucion = "131223"  # El texto que reemplazará al anterior

    # Llamar a la función con las variables definidas
    reemplazar_nombre_archivos(directorio, string_a_sustituir, string_de_sustitucion)
