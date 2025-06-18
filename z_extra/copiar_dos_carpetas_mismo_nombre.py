import os
import shutil

# Define los paths de los directorios
dir1 = "/home/gms/AnemoNAS/POR_DIA/_temp_2/"
dir2 = "/home/gms/AnemoNAS/POR_DIA/13-12-2023/"

# Recorre las carpetas dentro del primer directorio
for folder_name in os.listdir(dir1):
    path1 = os.path.join(dir1, folder_name)
    path2 = os.path.join(dir2, folder_name)

    # Verifica si es una carpeta en dir1 y si existe una carpeta con el mismo nombre en dir2
    if os.path.isdir(path1) and os.path.isdir(path2):
        print(f"Moviendo contenido de: {path1} a {path2}")

        # Mueve todo el contenido de la carpeta de dir1 a la carpeta correspondiente en dir2
        for item in os.listdir(path1):
            item_path = os.path.join(path1, item)
            dest_path = os.path.join(path2, item)

            if os.path.exists(dest_path):
                # Si ya existe el destino, eliminarlo antes de mover el nuevo
                if os.path.isdir(dest_path):
                    shutil.rmtree(dest_path)
                    print(f"Eliminada carpeta existente: {dest_path}")
                else:
                    os.remove(dest_path)
                    print(f"Eliminado archivo existente: {dest_path}")

            shutil.move(item_path, dest_path)
            print(f"Movido: {item_path} -> {dest_path}")

        # Elimina la carpeta vacía de dir1
        os.rmdir(path1)
        print(f"Eliminada carpeta vacía: {path1}")
