#Procesamiento de un video con el objetivo de obtener un frame de el fondo sin peces a traves de el uso de la mediana, dado que tiene menos ruido por los peces que la media.


# import cv2
# import numpy as np
# from tqdm import tqdm

# #Cargar el video
# vid = cv2.VideoCapture("/home/gms/AnemoNAS/temp/USCL2-195223-195723.mp4")

# #Verificar si el video se abrió correctamente
# if not vid.isOpened():
#     print("Error al abrir el video")
#     exit()
    
# # Obtener la cantidad total de frames del video
# total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# print(f"El video tiene {total_frames} frames en total.")

    
# #Leer los n_frames que queremos tomar como referencia y almacenarlos en una lista
# n_frames = 300
# frames = []
# for i in tqdm(range(n_frames), desc="Leyendo frames", unit="frames"):
#     ret, frame = vid.read()
#     if not ret:
#         break
#     frames.append(frame)

# #Convertir la lista de frames a un array de numpy, una matriz de 4 dimensiones (num_frames, altura, ancho, canales).
# frames = np.array(frames)

# #Calcular la mediana de los frames para encontrar la imagen de fondo
# print(f"Calculando la mediana de {n_frames} frames...")
# fondo = np.median(frames, axis=0).astype(np.uint8)  

# #Guardar la imagen de fondo generada
# cv2.imwrite("/home/gms/AnemoNAS/temp/background.png", fondo)
# print(f"Fondo calculado con éxito")


# # # Volver a procesar el video desde el inicio
# # vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

# # while True:
# #     ret, frame = vid.read()
# #     if not ret:
# #         break

# #     # Restar el fondo del frame actual
# #     diff = cv2.absdiff(frame, fondo)

# #     # Convertir a escala de grises
# #     gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# #     # Aplicar un umbral para destacar los peces
# #     _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

# #     # Mostrar los resultados
# #     cv2.imshow("Original", frame)
# #     cv2.imshow("Diferencia con fondo", diff)
# #     cv2.imshow("Peces detectados", thresh)

# #     if cv2.waitKey(30) & 0xFF == 27:  # Presionar ESC para salir
# #         break

# vid.release()
# cv2.destroyAllWindows()



#Codigo que lo subdivide en bloques

import cv2
import numpy as np
import os
import multiprocessing
import time


#Con multiprocessing

# Función que calcula la mediana de un grupo de frames
def calcular_mediana_grupo(group_num, group_size, video_path):
    frames_group = []  # Buffer temporal para el grupo
    
    # Abrir el video en cada proceso porque si no da error
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print(f"Error: No se pudo abrir el video en el proceso {group_num}.")
        return None
    
    # Establecer el punto de inicio de lectura del grupo
    vid.set(cv2.CAP_PROP_POS_FRAMES, group_num * group_size)
    
    # Leer y almacenar los frames del grupo
    for _ in range(group_size):
        ret, frame = vid.read()
        if not ret:
            break
        frames_group.append(frame)

    if len(frames_group) > 0:
        # Calcular la mediana del grupo
        frames_array = np.array(frames_group)
        group_median = np.median(frames_array, axis=0).astype(np.uint8)
        
        # Guardar la mediana del grupo en el subdirectorio con el nombre del grupo
        group_filename = os.path.join("/home/gms/AnemoNAS/temp/intermedia", f"mediana_grupo_{group_num+1}.png")
        cv2.imwrite(group_filename, group_median)
        
        print(f"Proceso {group_num}: Mediana del grupo calculada con éxito.")
        
        return group_median
    else:
        return None


# Función que maneja el proceso de la mediana final
def procesar_video():
    # Cargar el video
    video_path = "/home/gms/AnemoNAS/temp/USCL2-195223-195723.mp4"

    # Cargar el video
    vid = cv2.VideoCapture(video_path)

    # Verificar si el video se abrió correctamente
    if not vid.isOpened():
        print("Error: No se pudo abrir el video.")
        exit()

    # Parámetros
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # Número total de frames en el video
    group_size = 30  # Tamaño de cada grupo de frames
    num_groups = total_frames // group_size  # Número de grupos
    print(f"El video tiene {total_frames} frames en total, lo que serán {num_groups} medianas intermedias")

    # Crear un subdirectorio para guardar las medianas
    output_dir = "medianas_intermedias"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Comenzamos a medir el tiempo de ejecución
    start_time = time.time()

    # Usamos un Pool para paralelizar los cálculos de las medianas de los grupos
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        medianas_intermedias = list(
            pool.starmap(calcular_mediana_grupo, [(group_num, group_size, video_path) for group_num in range(num_groups)])
        )

    # Filtrar las medianas válidas (algunas pueden ser None si no se leyeron correctamente los frames)
    medianas_intermedias = [mediana for mediana in medianas_intermedias if mediana is not None]

    # Calcular la mediana final a partir de las medianas intermedias
    print("Calculando la mediana final...")
    median_frames_array = np.array(medianas_intermedias)
    final_background = np.median(median_frames_array, axis=0).astype(np.uint8)

    # Guardamos el fondo calculado (opcional)
    cv2.imwrite("/home/gms/AnemoNAS/temp/fondo_median_final.png", final_background)

   
    vid.release()
    
    # Calcular el tiempo total de ejecución
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos.")

# Llamamos a la función que procesará el video
if __name__ == "__main__":
    procesar_video()







# #Sin multiprocessing
# #Cargar el video
# vid = cv2.VideoCapture("/home/gms/AnemoNAS/temp/USCL2-195223-195723.mp4")

# #Verificar si el video se abrió correctamente
# if not vid.isOpened():
#     print("Error al abrir el video")
#     exit()
    
# # Parámetros
# total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
# group_size = 100
# num_groups = total_frames // group_size
# print(f"El video tiene {total_frames} frames en total, lo que serán {num_groups} medianas intermedias")

# medianas_intermedias = []
    
# #Para cada grupo, se leen y almacenan los frames para despues calcular la mediana
# for n_grupo in tqdm (range(num_groups), desc="Calculando medianas intermedias", unit="grupo"):
#     frames = []
#     for _ in range(group_size):
#         ret, frame = vid.read()
#         if not ret:
#             break
#         frames.append(frame)
        
#     if len(frames) > 0:
#         frames_array = np.array(frames)
#         mediana_grupo = np.median(frames_array, axis=0).astype(np.uint8)
#         medianas_intermedias.append(mediana_grupo)
        
#         filename = os.path.join("/home/gms/AnemoNAS/temp/intermedia", f"mediana_grupo_{n_grupo+1}.png")
#         cv2.imwrite(filename, mediana_grupo)

# # Calcular la mediana final a partir de las medianas intermedias
# print("Calculando la mediana final...")
# median_frames_array = np.array(medianas_intermedias)
# final_background = np.median(median_frames_array, axis=0).astype(np.uint8)

# # Guardamos el fondo calculado (opcional)
# cv2.imwrite("/home/gms/AnemoNAS/temp/background.png", final_background)

# vid.release()




# #Mediana carpeta
# import os
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# def calcular_mediana(carpeta, grupo_size=100):
#     # Listar todas las imágenes en la carpeta
#     archivos = [f for f in os.listdir(carpeta) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
#     # Lista para almacenar las medianas de cada grupo
#     medianas_grupos = []
    
#     # Procesar las imágenes en grupos de 100
#     for i in tqdm(range(0, len(archivos), grupo_size), desc="Procesando grupos"):
#         # Obtener las imágenes del grupo
#         grupo = archivos[i:i+grupo_size]
#         imagenes = []
        
#         for archivo in grupo:
#             ruta_imagen = os.path.join(carpeta, archivo)
#             imagen = Image.open(ruta_imagen)
#             imagen = imagen.convert('RGB')  # Convertir a RGB en caso de que la imagen esté en otro formato
#             imagenes.append(np.array(imagen))  # Convertir la imagen a un array de numpy
        
#         # Calcular la mediana del grupo de imágenes
#         imagenes_np = np.array(imagenes)
#         mediana_grupo = np.median(imagenes_np, axis=0).astype(np.uint8)  # Calcular la mediana a lo largo del eje 0 (imágenes)
        
#         # Agregar la mediana del grupo a la lista de medianas
#         medianas_grupos.append(mediana_grupo)

#     # Calcular la mediana final a partir de las medianas de los grupos
#     medianas_grupos_np = np.array(medianas_grupos)
#     mediana_final = np.median(medianas_grupos_np, axis=0).astype(np.uint8)  # Calcular la mediana de las medianas

#     # Convertir la mediana final de vuelta a una imagen
#     mediana_final_imagen = Image.fromarray(mediana_final)
    
#     return mediana_final_imagen

# # Definir la carpeta que contiene las imágenes
# carpeta_imagenes = "/home/gms/AnemoNAS/temp/intermedia_final/"

# # Calcular la mediana
# mediana = calcular_mediana(carpeta_imagenes)

# # Mostrar o guardar la mediana
# mediana.show()
# mediana.save("/home/gms/AnemoNAS/temp/mediana_imagen_triple.png")