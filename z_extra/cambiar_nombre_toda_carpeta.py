import os

# Configura tu carpeta y el texto a añadir
folder_path = '/home/gmanty/code/anotacion_YOLO'   
texto_extra = '_USCL2-194221-194721'             

# Recorrer todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Comprobar si es un archivo
    if os.path.isfile(file_path):
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}{texto_extra}{ext}"
        new_file_path = os.path.join(folder_path, new_filename)

        os.rename(file_path, new_file_path)

print("Renombrado completado.")
