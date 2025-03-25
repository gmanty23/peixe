import cv2
import numpy as np
import os
import multiprocessing
import time
import debugpy
import random



# --------------------- FUNCIONES PARA EXTRAER IMÁGENES DEL VIDEO ---------------------

# Se va calculando tambien el progreso a mostrar en la barra de progreso 
#     Comunicación entre Procesos:

#         Usaremos una multiprocessing.Queue para enviar el progreso de cada proceso secundario al proceso principal.

#     Actualización de la Barra de Progreso:

#         El proceso principal leerá los mensajes de la cola y actualizará la barra de progreso en la interfaz gráfica.

#     Cálculo del Progreso Total:

#         El progreso total se calculará sumando el progreso de todos los procesos secundarios.
    
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
            print(f"Error en la extracción de imágenes en el proceso {segment_id}: {e}")


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
        images_og_path = os.path.join(output_path, "imagenes_og")
        if not os.path.exists(images_og_path):
            os.makedirs(images_og_path)
            
        #Crear una cola para comunicar el progreso de los procesos
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
        print(f"Error en la extracción de imágenes el proceso principal: {e}")





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
        print(f"Error en el redimensionamiento de imagenes en el proceso {id_segmento}: {e}")
        
        
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
        images_resized_path = os.path.join(output_path, "imagenes_resized")
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

        return images_resized_path
        
    except Exception as e:
        print(f"Error en el redimensionamiento de las imagenes en el proceso principal: {e}")
    
        



# --------------------- FUNCIONES PARA ATENUAR EL FONDO DE LAS IMÁGENES ---------------------
# Función que calcula la mediana de un grupo de frames
def calcular_mediana_segmento(input_path, imagenes_grupo, id_segmento, progress_queue, output_path):
    try:
        
        imagenes_segmento = [] # Buffer temporal para el grupo 

        for i, img_name in enumerate(imagenes_grupo):
            img_path = os.path.join(input_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                imagenes_segmento.append(img)

        if imagenes_segmento:
            # Calcular la mediana del grupo
            imagenes_array = np.array(imagenes_segmento)
            mediana_grupo = np.median(imagenes_array, axis=0).astype(np.uint8)
            
            # Guardar la mediana del grupo en el subdirectorio con el nombre del grupo
            mediana_grupo_path = os.path.join(output_path, f"mediana_grupo_{id_segmento+1}.jpg")
            cv2.imwrite(mediana_grupo_path, mediana_grupo)
            
            print(f"Proceso {id_segmento}: Mediana del grupo calculada con éxito.")

            # Enviar progreso - cada grupo completado es un paso
            progress_queue.put(1)
            
            return mediana_grupo
        else:
            print(f"Error: No se pudieron cargar las imágenes del segmento {id_segmento}.")
            return None
        
    except Exception as e:
        print(f"Error en el cálculo de la mediana en el proceso {id_segmento}: {e}")



# Función que calcula la mediana final a partir de las medianas intermedias con el objetivo de tener una aproximación del fondo estático de la imagen
def calcular_mediana(input_path,sizeGrupo,total_imagenes,imagenes, progress_callback=None, num_procesos=None):
    try:

        # Detectar el número de procesadores disponibles
        if num_procesos is None:
            num_procesos = multiprocessing.cpu_count()

        # Crear un directorio para almacenar las medianas intermedias
        cache_path = "processing_GUI/procesamiento/cache/medianas_intermedias"
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        # Crear una cola para manejar el progreso
        progress_queue = multiprocessing.Queue()

        # Dividir las imágenes en grupos de tamaño group_size de manera aleatoria, para no estar sesgadas por la continuaidad de las secuencias
        random.shuffle(imagenes)
        grupos_imagenes = [imagenes[i:i + sizeGrupo] for i in range(0, total_imagenes, sizeGrupo)]

        # Calcular el total de medianas para medir el progreso
        total_medianas = len(grupos_imagenes) + 1 #Medianas intermedias y mediana final
        progreso = 0

        print(f"El video tiene {total_imagenes} frames en total, lo que serán {total_medianas} medianas intermedias")

        # Crear los procesos para calcular las medianas intermedias
        procesos = []
        for i, imagenes_segmento in enumerate(grupos_imagenes):
            # Crear y lanzar un proceso para calcular la mediana del segmento
            p = multiprocessing.Process(target=calcular_mediana_segmento, args=(input_path, imagenes_segmento, i, progress_queue, cache_path))
            procesos.append(p)
            p.start()

        # Monitorear el progreso de los procesos
        while any(p.is_alive() for p in procesos):
            while not progress_queue.empty():
                progress_queue.get()
                progreso += 1
                progreso_porcentaje = int((progreso / total_medianas) * 100)
                if progress_callback:
                    progress_callback(progreso_porcentaje)

        # Esperar a que todos los procesos terminen
        for p in procesos:
            p.join()

        # Leer las medianas intermedias generadas por los procesos
        medianas_intermedias = [cv2.imread(os.path.join(cache_path, f)) for f in os.listdir(cache_path) if f.endswith('.jpg')]
        
        if medianas_intermedias:
            # Calcular la mediana final a partir de las medianas intermedias
            array_medianas_intermedias = np.array(medianas_intermedias)
            fondo_final = np.median(array_medianas_intermedias, axis=0).astype(np.uint8)
            cv2.imwrite( "processing_GUI/procesamiento/cache/fondo_final/fondo_final.jpg", fondo_final)
            # Guardar la imagen de la mediana final
            #debugpy.breakpoint()
            print("Mediana final calculada ")
            # Actualizar progreso al 100%
            if progress_callback:
                progress_callback(100)

            return fondo_final
        else:
            print("Error: No se pudieron calcular las medianas intermedias correctamente.")
        
    except Exception as e:
        print(f"Error en el cálculo de la mediana en el proceso principal: {e}")


# Función principal que atenúa el fondo de las imágenes
def atenuar_fondo_imagenes(input_path, output_path, sizeGrupo, factor_at, umbral_dif, apertura_flag, cierre_flag, apertura_kernel_size, cierre_kernel_size, progress_callback=None, num_procesos=None):
    try:
        
        start_time = time.time()
        
        print("El input path es: ", input_path)

        # Obtener la lista de imágenes a redimensionar
        imagenes = [f for f in os.listdir(input_path) if f.endswith(".jpg")]
        total_imagenes = len(imagenes)
        if total_imagenes == 0:
            print("Error: No se encontraron imágenes en el directorio de entrada.")
            return None
           
        # Calcular el fondo del video a través de un análisis de medianas
        fondo_final = calcular_mediana(input_path,sizeGrupo,total_imagenes,imagenes, progress_callback, num_procesos)

        imagenes.sort()  # Ordenar antes de procesar

        # Crear los directorios de salida
        output_path_imagenes_diferencias = os.path.join(output_path, "imagenes_diferencias")
        if not os.path.exists(output_path_imagenes_diferencias):
            os.makedirs(output_path_imagenes_diferencias)
        output_path_imagenes_fondo_at =  os.path.join(output_path, "imagenes_fondo_atenuado")
        if not os.path.exists(output_path_imagenes_fondo_at):
            os.makedirs(output_path_imagenes_fondo_at)

        if progress_callback:
                progress_callback(0)

        progreso_porcentaje = 0
        


        # Procesar imagenes restando el fondo
        for i, img_name in enumerate(imagenes):
            img_path = os.path.join(input_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Calcular la diferencia entre la imagen y el fondo
                diferencia = cv2.absdiff(img, fondo_final)
                # Convertir la diferencia a escala de grises
                diferencia_gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)
                # Aplicar umbralización para obtener una máscara binaria
                _, diferencia_umbral = cv2.threshold(diferencia_gris, umbral_dif, 255, cv2.THRESH_BINARY)
                # Guardar las imágenes de las diferencias 
                cv2.imwrite(os.path.join(output_path_imagenes_diferencias, f'diferencia_{i:04d}.jpg'), diferencia_umbral)
                #Si seleccionado, aplicar apertura morfológica para eliminar ruido impulsivo 
                if apertura_flag:
                    apertura_kernel = np.ones(apertura_kernel_size, np.uint8)
                    diferencia_umbral = cv2.morphologyEx(diferencia_umbral, cv2.MORPH_OPEN, apertura_kernel)
                #Si seleccionado, aplicar cierre morfológico para rellenar huecos en los peces
                if cierre_flag:
                    cierre_kernel = np.ones(cierre_kernel_size, np.uint8)
                    diferencia_umbral = cv2.morphologyEx(diferencia_umbral, cv2.MORPH_CLOSE, cierre_kernel)
                # Obtener imágenes con el fondo atenuado
                # Crear una máscara invertida para atenuar el fondo (donde no hay peces)
                fondo_mask = cv2.bitwise_not(diferencia_umbral)
                 #Atenuar el fondo multiplicando el fondo por el factor de atenuación
                fondo_atenuado = img * factor_at  # Apagar el fondo multiplicando por el factor de atenuación
                # Crear la imagen final: las áreas con los peces se mantienen intactas, y las del fondo se atenúan
                img_fondo_at = cv2.bitwise_and(img, img, mask=diferencia_umbral)  # Los peces se mantienen
                img_fondo_at += cv2.bitwise_and(fondo_atenuado.astype(np.uint8), fondo_atenuado.astype(np.uint8), mask=fondo_mask)  # El fondo se atenúa
                cv2.imwrite(os.path.join(output_path_imagenes_fondo_at, f'fondo_at_{i:04d}.jpg'), img_fondo_at)
                
          
                # Actualizar el progreso
                progreso_porcentaje = int(((i+1) / total_imagenes) * 100)
                if progress_callback:
                    progress_callback(progreso_porcentaje)
        
        end_time = time.time()
    
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de la atenuacion del fondo: {execution_time:.2f} segundos")

        return None

    except Exception as e:
        print(f"Error en el proceso principal: {e}")
    
        
