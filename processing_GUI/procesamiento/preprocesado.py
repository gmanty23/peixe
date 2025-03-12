import cv2
import numpy as np
import os
import multiprocessing
import time



# --------------------- FUNCIONES PARA EXTRAER IMÁGENES DEL VIDEO ---------------------

# Se va calculando tambien el progreso a mostrar en la barra de progreso 
    # Comunicación entre Procesos:

    #     Usaremos una multiprocessing.Queue para enviar el progreso de cada proceso secundario al proceso principal.

    # Actualización de la Barra de Progreso:

    #     El proceso principal leerá los mensajes de la cola y actualizará la barra de progreso en la interfaz gráfica.

    # Cálculo del Progreso Total:

    #     El progreso total se calculará sumando el progreso de todos los procesos secundarios.
    
# Función que extrae los frames de un segmento de un video y los guarda como imágenes numeradas en un directorio
def extraer_imagenes_segmento(video_path, output_path, frame_inicial, frame_final,segment_id, progress_queue):
    
 
    try:
        # Abrir el video por el frame inicial del segmento
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            print("Error: No se pudo abrir el video en el proceso {segment_id}.")
            return
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)
        frame_count = frame_inicial
        total_frames_segmento = frame_final - frame_inicial + 1
        
        # Extraer los frames del segmento y guardarlos como imágenes
        while frame_count <= frame_final:
            ret, frame = vid.read()
            if not ret:
                break
            frame_filename = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            
            # Enviar el progreso al hilo principal
            progress = int(((frame_count - frame_inicial) / total_frames_segmento) * 100)
            progress_queue.put((segment_id,progress))
            
        # Liberar recursos
        vid.release()
        
        print(f"Proceso {segment_id} terminado correctamente.")
    except Exception as e:
            print(f"Error en el proceso {segment_id}: {e}")


# Función que extrae los frames de un video y los guarda como imágenes numeradas en un directorio 
def extraer_imagenes(video_path, output_path, progress_callback=None, num_procesos=None):
    
    try:
        start_time = time.time()
        
        # Detectar el número de procesadores disponibles
        if num_procesos is None:
            num_procesos = multiprocessing.cpu_count()
            
        # Abrir el video y contar los frames
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            print("Error: No se pudo abrir el video.")
            return None 
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"El video tiene {total_frames} frames en total.")
        vid.release()
        
        # Crear el directorio de salida
        images_og_path = os.path.join(output_path, "images_og")
        if not os.path.exists(images_og_path):
            os.makedirs(images_og_path)
            
        #Crear una cola para comnucair el progreso de los procesos
        progress_queue = multiprocessing.Queue()
            
        # Segmentar los videos en función del numero de procesadores
        segment_size = total_frames // num_procesos
        procesos = []
        
        for i in range(num_procesos):
            frame_inicial = i * segment_size
            frame_final = (i+1) * segment_size - 1 if i < num_procesos - 1 else total_frames - 1
            p = multiprocessing.Process(target=extraer_imagenes_segmento, args=(video_path, images_og_path, frame_inicial, frame_final,i,progress_queue))
            procesos.append(p)
            p.start()
            
        # Leer el progreso de los procesos y mostrarlo en la barra de progreso
        progreso_total = [0] * num_procesos   # Lista para almacenar el progreso de cada proceso
        while any(p.is_alive() for p in procesos):
            while not progress_queue.empty(): 
                segment_id, progress = progress_queue.get()
                progreso_total[segment_id] = progress
                
                progreso_promedio = int(sum(progreso_total) // num_procesos)
                if progress_callback:
                    progress_callback(progreso_promedio)
            
        # Vaciar la cola por completo
            while not progress_queue.empty():
                progress_queue.get()
                
        # Esperar a que todos los procesos terminen
        for p in procesos:
            p.join()

        print(f"Imágenes extraídas y guardadas en: {images_og_path}")
        
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de la extracción de frames: {execution_time:.2f} segundos")
        
        return images_og_path
    except Exception as e:
        print(f"Error en el proceso principal: {e}")



# # Función que extrae los frames de un video y los guarda como imágenes numeradas en un directorio
# # Va calculando tambien el progreso a mostrar en la barra de progreso
# def extraer_imagenes(video_path, output_path, cancelar_flag, progress_callback=None):

#     start_time = time.time()
    
#     # Abrir el video
#     vid = cv2.VideoCapture(video_path)
#     if not vid.isOpened():
#         print("Error: No se pudo abrir el video.")
#         return None
    
#     # Crear el directorio de salida
#     images_og_path = os.path.join(output_path, "images_og")
#     if not os.path.exists(images_og_path):
#         os.makedirs(images_og_path)
    
#     # Extraer los frames del video y guardarlos como imágenes
#     total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_count = 0
#     while True:
#         if cancelar_flag:
#             break
#         ret, frame = vid.read()
#         if not ret:
#             break
#         # Guardar el frame como imagen
#         frame_filename = os.path.join(images_og_path, f"frame_{frame_count:04d}.jpg")
#         cv2.imwrite(frame_filename, frame)
#         #print(f"Frame {frame_count} guardado")
#         frame_count += 1
        
#         if progress_callback:
#             progress = int((frame_count / total_frames) * 100)
#             progress_callback(progress)
    
#     # Liberar recursos
#     vid.release()
#     print(f"Imágenes extraídas y guardadas en: {images_og_path}")
    
    

#     end_time = time.time()
    
#     execution_time = end_time - start_time
#     print(f"Tiempo de ejecución de la extracción de frames: {execution_time:.2f} segundos")
    
    
#     return images_og_path







# --------------------- FUNCIÓNES PARA REDIMENSIONAR IMÁGENES ---------------------

def redimensionar_imagenes_segmento(input_path, output_path, imagenes, ancho, alto, id_segmento, progress_queue):
    try:
        
        
        total_imagenes_segmento = len(imagenes)
        for i, img_name in enumerate(imagenes):
            img_path = os.path.join(input_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, (ancho, alto))
                img_resized_path_segment = os.path.join(output_path, img_name)
                cv2.imwrite(img_resized_path_segment, img_resized)
                
                progress = int(((i+1) / total_imagenes_segmento) * 100)
                progress_queue.put((id_segmento, progress))
        print(f"Proceso {id_segmento} terminado correctamente.")
    except Exception as e:
        print(f"Error en el proceso {id_segmento}: {e}")
        
        
def redimensionar_imagenes(input_path, output_path, ancho, alto, progress_callback=None, num_procesos=None):
    try:
        
        start_time = time.time()
        
        # Detectar el número de procesadores disponibles
        if num_procesos is None:
            num_procesos = multiprocessing.cpu_count()
            
        # Obtener la lista de imágenes a redimensionar
        imagenes = [f for f in os.listdir(input_path) if f.endswith(".jpg")]
        total_imagenes = len(imagenes)
        if total_imagenes == 0:
            print("Error: No se encontraron imágenes en el directorio de entrada.")
            return None
        
        # Crear el directorio de salida
        images_resized_path = os.path.join(output_path, "images_resized")
        if not os.path.exists(images_resized_path):
            os.makedirs(images_resized_path)    
        
        # Crear una cola para comnucair el progreso de los procesos
        progress_queue = multiprocessing.Queue()
        
        # Dividir las imagenes en segmentos para cada proceso
        segment_size = total_imagenes // num_procesos
        procesos = []
        
        # Procesar cada segmento 
        for i in range(num_procesos):
            frame_inicial = i * segment_size 
            frame_final = (i+1) * segment_size if i < num_procesos - 1 else total_imagenes
            imagenes_segmento = imagenes[frame_inicial:frame_final]  
            
            p = multiprocessing.Process(target=redimensionar_imagenes_segmento, args=(input_path, images_resized_path, imagenes_segmento, ancho, alto, i, progress_queue))
            procesos.append(p)
            p.start()
            
        # Leer el progreso de la cola y mostrarlo en la barra de progreso
        progreso_total = [0] * num_procesos   
        while any(p.is_alive() for p in procesos):
            while not progress_queue.empty(): 
                segment_id, progress = progress_queue.get()
                progreso_total[segment_id] = progress
                
                progreso_promedio = int(sum(progreso_total) // num_procesos)
                if progress_callback:
                    progress_callback(progreso_promedio)
                    
        # Vaciar la cola por completo
        while not progress_queue.empty():
            progress_queue.get()
            
        # Esperar a que todos los procesos terminen
        for p in procesos:
            p.join()
            
        print(f"Imágenes redimensionadas guardadas en: {images_resized_path}")
    
        end_time = time.time()
    
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de la reducción de resolucion de frames: {execution_time:.2f} segundos")
        
    except Exception as e:
        print(f"Error en el proceso principal: {e}")
    
        

# #Funcion para reducir la resolución del video a fullHD (1920x1080)
# def reducir_resolucion_imagenes(input_path, output_path, width, height, cancelar_flag, progress_callback=None):

#     start_time = time.time()
    
#     # Crear el directorio de salida
#     images_resized_path = os.path.join(output_path, "images_resized")
#     if not os.path.exists(images_resized_path):
#         os.makedirs(images_resized_path)
    
#     # Obtener la lista de imágenes a redimensionar
#     imagenes = [f for f in os.listdir(input_path) if f.endswith(".jpg")]
#     total_images = len(imagenes)
     
#     # Procesar cada imagen
#     for i, archivo in enumerate(imagenes):
#         if cancelar_flag:
#             break
        
#         img_path = os.path.join(input_path, archivo)
#         img = cv2.imread(img_path)
#         if img is not None:
#             img_resized = cv2.resize(img, (width, height))
#             img_resized_path = os.path.join(images_resized_path, archivo)
#             cv2.imwrite(img_resized_path, img_resized)
            
#             if progress_callback:
#                 progress = int(((i+1) / total_images) * 100)
#                 progress_callback(progress)
                
#     print(f"Imágenes redimensionadas guardadas en: {images_resized_path}")
    
#     end_time = time.time()
    
#     execution_time = end_time - start_time
#     print(f"Tiempo de ejecución de la reducción de resolucion de frames: {execution_time:.2f} segundos")
    
    
    
#     return images_resized_path
    
    




# #Función que calcula la mediana de un grupo de frames
# def calcular_mediana_grupo(group_num, group_size, video_path):
#     frames_group = []  # Buffer temporal para el grupo
    
#     # Abrir el video en cada proceso porque si no da error
#     vid = cv2.VideoCapture(video_path)
#     if not vid.isOpened():
#         print(f"Error: No se pudo abrir el video en el proceso {group_num}.")
#         return None
    
#     # Establecer el punto de inicio de lectura del grupo
#     vid.set(cv2.CAP_PROP_POS_FRAMES, group_num * group_size)
    
#     # Leer y almacenar los frames del grupo
#     for _ in range(group_size):
#         ret, frame = vid.read()
#         if not ret:
#             break
#         frames_group.append(frame)

#     if len(frames_group) > 0:
#         # Calcular la mediana del grupo
#         frames_array = np.array(frames_group)
#         group_median = np.median(frames_array, axis=0).astype(np.uint8)
        
#         # Guardar la mediana del grupo en el subdirectorio con el nombre del grupo
#         group_filename = os.path.join("processing_GUI/cache_intermedio/medianas_intermedias", f"mediana_grupo_{group_num+1}.png")
#         cv2.imwrite(group_filename, group_median)
        
#         print(f"Proceso {group_num}: Mediana del grupo calculada con éxito.")
        
#         return group_median
#     else:
#         return None
    
    
    
# # Función que maneja el proceso de la mediana final
# def calcular_mediana_video(video_path):
#     # Abrir el video
#     vid = cv2.VideoCapture(video_path)

#     # Verificar si el video se abrió correctamente
#     if not vid.isOpened():
#         print("Error: No se pudo abrir el video.")
#         return

#     # Parámetros
#     total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # Número total de frames en el video
#     group_size = 40  # Tamaño de cada grupo de frames
#     num_groups = total_frames // group_size  # Número de grupos
#     print(f"El video tiene {total_frames} frames en total, lo que serán {num_groups} medianas intermedias")
    
#     # Usamos un Pool para paralelizar los cálculos de las medianas de los grupos
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         medianas_intermedias = list(
#             pool.starmap(calcular_mediana_grupo, [(group_num, group_size, video_path) for group_num in range(num_groups)])
#         )
    
#     # Filtrar las medianas válidas
#     medianas_intermedias = [mediana for mediana in medianas_intermedias if mediana is not None]

#     # Calcular la mediana final a partir de las medianas intermedias
#     print("Calculando la mediana final...")
#     median_frames_array = np.array(medianas_intermedias)
#     final_background = np.median(median_frames_array, axis=0).astype(np.uint8)
    
#     output_path = "processing_GUI/cache_intermedio/fondo_median_final.png"
#     cv2.imwrite(output_path, final_background)
#     print("Fondo calculado con éxito")

    
#     vid.release()
    
#     return None


# #Funcion para reducir la resolución del video a fullHD
# def reducir_resolucion(video_path, output_path):
#     print("El output path es: ", output_path)
#     #Abrir el video
#     print(f"Video Path: {video_path}") 
#     vid = cv2.VideoCapture(video_path)
#     if not vid.isOpened():
#         print("Error: No se pudo abrir el video.")
#         return
    
#     # Establecer el códec para guardar el nuevo video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Usamos mp4 como formato de salida
#     fps = vid.get(cv2.CAP_PROP_FPS)  # Obtenemos el fps original del video
#     width, height = 1920, 1080  # Resolución FullHD
     
#     # Crear el escritor de video para guardar el video con la resolución reducida
#     images_resized_path = os.path.join(output_path, "images_resized")
#     print(images_resized_path)
#     if not os.path.exists(images_resized_path):
#         os.makedirs(images_resized_path)
#     out = cv2.VideoWriter(images_resized_path, fourcc, fps, (width, height))
    
#     while True:
#         # Leer el siguiente fotograma del video
#         ret, frame = vid.read()
#         if not ret:
#             break
        
#         # Redimensionar el fotograma al tamaño deseado
#         frame_resized = cv2.resize(frame, (width, height))
        
#         # Guardar el fotograma redimensionado
#         out.write(frame_resized)
        
#     # Liberar recursos
#     vid.release()
#     out.release()
#     print(f"Video reducido a FullHD y guardado en: {images_resized_path}")
#     return images_resized_path