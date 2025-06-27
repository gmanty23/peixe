#Script con el objetivo de dividir el video completo separado en frames en subdirectorios de 512 imágenes cada uno.
#Esto se hace por que este es el numero de muestras que debe tomar cada secuencia como input del modelo 'moment'.


import os #Para listar archivos y crear carpetas
import shutil  #Permite mover archivos
import math #Para hacer los calculos matematicos
import time #Para medir el tiempo de ejecución

def dividir_imagenes_en_subdirectorios(ruta_directorio, img_x_dir = 512):
    #Verificar si la carpeta existe
    if not os.path.exists(ruta_directorio):
        print(f"La carpeta no existe")
        return
    
    #Listar las imagenes de la carpeta
    imagenes = sorted([f for f in os.listdir(ruta_directorio) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])  #Ojo para moverlas en orden 
    
    #Contar cuantas imágenes hay
    n_imagenes = len(imagenes)
    if n_imagenes == 0:
        print(f"No hay imágenes en la carpeta")
        return
    
    #Contar cuantos subdirectorios se necesitan
    n_subdirectorios = math.ceil(n_imagenes / img_x_dir)
    
    print(f"Total de imágenes: {n_imagenes}")
    print(f"Se crearán {n_subdirectorios} subdirectorios.")
    
    t_inicial = time.time()
    
    #Crear los subdirectorios y disribuir las imágenes
    for i in range(n_subdirectorios):
        ruta_subdirectorio = os.path.join(ruta_directorio, f"{i}")
        os.makedirs(ruta_subdirectorio, exist_ok=True)
        
        inicio = i * img_x_dir
        fin = inicio + img_x_dir
        if fin > n_imagenes:
            fin = n_imagenes
        imgs_lote = imagenes[inicio:fin]
        
        for img in imgs_lote:
            ruta_original = os.path.join(ruta_directorio, img)
            ruta_nueva = os.path.join(ruta_subdirectorio, img)
            shutil.move(ruta_original, ruta_nueva)
            
    t_final = time.time()
    t_total = t_final - t_inicial
    print(f"\nProceso completado en {t_total:.2f} segundos.")
            
            
#Ejecutamos el script
ruta = "/home/gms/AnemoNAS/Workspace/06-12-23-Lateral-50Peces/temp/images/"
dividir_imagenes_en_subdirectorios(ruta)

    
    