import cv2
import numpy as np
import os
import multiprocessing
import time
import debugpy
import random
import queue


# --------------------- FUNCIONES PARA EXTRAER IMÁGENES DEL VIDEO ---------------------

# Se va calculando tambien el progreso a mostrar en la barra de progreso 
#     Comunicación entre Procesos:

#         Usaremos una multiprocessing.Queue para enviar el progreso de cada proceso secundario al proceso principal.

#     Actualización de la Barra de Progreso:

#         El proceso principal leerá los mensajes de la cola y actualizará la barra de progreso en la interfaz gráfica.

#     Cálculo del Progreso Total:

#         El progreso total se calculará sumando el progreso de todos los procesos secundarios.
    
# Función que extrae los frames de un segmento de un video y los guarda como imágenes numeradas en un directorio
# Si está activada la opción de adaptar a moment, se guardan en subdirectorios
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
            num_procesos = multiprocessing.cpu_count() - 1
            
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
        os.makedirs(images_og_path, exist_ok=True)
            
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
            num_procesos = multiprocessing.cpu_count() - 1
            
        # Obtener la lista de imágenes a redimensionar
        imagenes = [f for f in os.listdir(input_path) if f.endswith(".jpg")]
        total_imagenes = len(imagenes)
        if total_imagenes == 0:
            print("Error: No se encontraron imágenes en el directorio de entrada.")
            return None
        
        # Crear el directorio de salida
        images_resized_path = os.path.join(output_path, "imagenes_resized")
        os.makedirs(images_resized_path, exist_ok=True)  
        
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
def calcular_mediana_segmento(input_path, lotes_imagenes, id_proceso, progress_queue, output_path):
    try:
        print(f"🟢 Proceso {id_proceso} arrancó con {len(lotes_imagenes)} grupos.")
        for i, (indice_lote, imagenes_grupo) in enumerate(lotes_imagenes):
            imagenes_segmento = []
            for img_name in imagenes_grupo:
                img_path = os.path.join(input_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    imagenes_segmento.append(img)

            if imagenes_segmento:
                imagenes_array = np.array(imagenes_segmento)
                mediana_grupo = np.median(imagenes_array, axis=0).astype(np.uint8)

                mediana_grupo_path = os.path.join(output_path, f"mediana_grupo_{indice_lote+1}.jpg")
                cv2.imwrite(mediana_grupo_path, mediana_grupo)

                print(f"Proceso {id_proceso}: Mediana del grupo {indice_lote+1} calculada con éxito.")
                progress_queue.put(1)
            else:
                print(f"Error: No se pudieron cargar imágenes del grupo {indice_lote}.")
    except Exception as e:
        print(f"Error en el proceso {id_proceso}: {e}")

# Funcion necesaria en videos muy largos, en los que el numero de medianas intermedias supera la capacidad de la memoria
def reducir_medianas_jerarquicamente(medianas, max_grupo=60):
    while len(medianas) > max_grupo:
        print(f"Reduciendo {len(medianas)} medianas a grupos de máximo {max_grupo}...")
        nuevas_medianas = []
        for i in range(0, len(medianas), max_grupo):
            grupo = medianas[i:i + max_grupo]
            grupo_array = np.array(grupo)
            mediana_grupo = np.median(grupo_array, axis=0).astype(np.uint8)
            nuevas_medianas.append(mediana_grupo)
        medianas = nuevas_medianas
    return medianas


# Función que calcula la mediana final a partir de las medianas intermedias con el objetivo de tener una aproximación del fondo estático de la imagen
def calcular_mediana(input_path, sizeGrupo, total_imagenes, imagenes, progress_callback_especifico=None, num_procesos=None):
    try:
        import queue  # Para manejar excepciones de timeout en la cola

        if num_procesos is None:
            num_procesos = multiprocessing.cpu_count() -1 

        cache_path = "processing_GUI/procesamiento/cache/medianas_intermedias"
        os.makedirs(cache_path, exist_ok=True)

        random.shuffle(imagenes)
        grupos_imagenes = [imagenes[i:i + sizeGrupo] for i in range(0, total_imagenes, sizeGrupo)]
        total_grupos = len(grupos_imagenes)
        total_medianas = total_grupos  # +1 por la final

        print(f"El video tiene {total_imagenes} frames en total, lo que serán {total_medianas} medianas intermedias")

        # Dividir los grupos entre los procesos
        lotes_por_proceso = [[] for _ in range(num_procesos)]
        for idx, grupo in enumerate(grupos_imagenes):
            lotes_por_proceso[idx % num_procesos].append((idx, grupo))

        for i, lote in enumerate(lotes_por_proceso):
            grupos_ids = [x[0] + 1 for x in lote]
            print(f"🧩 Proceso {i} tiene los grupos: {grupos_ids}")

        progress_queue = multiprocessing.Queue()
        procesos = []

        # Lanzar procesos
        for i in range(num_procesos):
            p = multiprocessing.Process(
                target=calcular_mediana_segmento,
                args=(input_path, lotes_por_proceso[i], i, progress_queue, cache_path)
            )
            procesos.append(p)
            p.start()

        # Seguimiento del progreso con protección por timeout
        progreso = 0
        while progreso < total_grupos:
            try:
                progress_queue.get(timeout=100)
                progreso += 1
                porcentaje = int((progreso / total_medianas) * 100)
                if progress_callback_especifico:
                    progress_callback_especifico(porcentaje)
            except queue.Empty:
                print("⚠️ Advertencia: timeout esperando progreso. Puede que un proceso haya fallado.")
                break

        # Esperar a que todos los procesos terminen
        for p in procesos:
            p.join()

        # Calcular mediana final
        medianas_intermedias = [
            cv2.imread(os.path.join(cache_path, f))
            for f in os.listdir(cache_path)
            if f.endswith('.jpg')
        ]

        if medianas_intermedias:
            medianas_reducidas = reducir_medianas_jerarquicamente(medianas_intermedias, sizeGrupo)
            fondo_final = np.median(np.array(medianas_reducidas), axis=0).astype(np.uint8)
            cv2.imwrite("processing_GUI/procesamiento/cache/fondo_final/fondo_final.jpg", fondo_final)

            print("Mediana final calculada")
            if progress_callback_especifico:
                progress_callback_especifico(100)

            for f in os.listdir(cache_path):
                os.remove(os.path.join(cache_path, f))

            return fondo_final
        else:
            print("Error: No se pudieron calcular las medianas intermedias correctamente.")
    except Exception as e:
        print(f"Error en el cálculo de la mediana en el proceso principal: {e}")

# Función que realiza el procesamiento de atenuacion de fondo en cada segmento
def atenuar_fondo_imagenes_segmento(input_path, output_path, imagenes, fondo_final, factor_at, umbral_dif, 
                                  apertura_flag, cierre_flag, dilatacion_flag, apertura_kernel_size, cierre_kernel_size, dilatacion_kernel_size, 
                                  id_segmento, progress_queue):
    try:
        output_path_imagenes_diferencias = os.path.join(output_path, "imagenes_diferencias")
        output_path_imagenes_fondo_at = os.path.join(output_path, "imagenes_fondo_atenuado")
        
        total_imagenes_segmento = len(imagenes)
        
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
                
                # Operaciones morfológicas, si se han seleccionado
                if apertura_flag:
                    apertura_kernel = np.ones(apertura_kernel_size, np.uint8)
                    diferencia_umbral = cv2.morphologyEx(diferencia_umbral, cv2.MORPH_OPEN, apertura_kernel)
                if dilatacion_flag:
                    dilatacion_kernel = np.ones(dilatacion_kernel_size, np.uint8)
                    diferencia_umbral = cv2.dilate(diferencia_umbral, dilatacion_kernel, iterations=1)
                if cierre_flag:
                    cierre_kernel = np.ones(cierre_kernel_size, np.uint8)
                    diferencia_umbral = cv2.morphologyEx(diferencia_umbral, cv2.MORPH_CLOSE, cierre_kernel)
                
                
                # Guardar diferencia
                cv2.imwrite(os.path.join(output_path_imagenes_diferencias, f'diferencia_{img_name.split("_")[1]}'), diferencia_umbral)
                
                # Atenuar fondo
                # Crear una máscara invertida para atenuar el fondo (donde no hay peces)
                fondo_mask = cv2.bitwise_not(diferencia_umbral)
                # Atenuar el fondo multiplicando el fondo por el factor de atenuación seleccionado
                fondo_atenuado = img * factor_at
                # Crear la imagen final: las áreas con los peces se mantienen intactas, y las del fondo se atenúan
                img_fondo_at = cv2.bitwise_and(img, img, mask=diferencia_umbral)
                img_fondo_at += cv2.bitwise_and(fondo_atenuado.astype(np.uint8), fondo_atenuado.astype(np.uint8), mask=fondo_mask)
                
                # Guardar resultado
                cv2.imwrite(os.path.join(output_path_imagenes_fondo_at, f'fondo_at_{img_name.split("_")[1]}'), img_fondo_at)
                
                # Enviar progreso
                progress = int(((i+1) / total_imagenes_segmento) * 100)
                progress_queue.put((id_segmento, progress))
                
        print(f"Proceso {id_segmento} terminado correctamente.")
        
    except Exception as e:
        print(f"Error en la atenuacion de fondo en el proceso {id_segmento}: {e}")



# Función principal que atenúa el fondo de las imágenes
def atenuar_fondo_imagenes(input_path, output_path, sizeGrupo, factor_at, umbral_dif, 
                          apertura_flag, cierre_flag, dilatacion_flag, apertura_kernel_size, cierre_kernel_size, dilatacion_kernel_size,
                          progress_callback_especifico=None, progress_callback_etapa=None, num_procesos=None):
    try:
        start_time = time.time()
        
        # Configuración de multiprocessing
        if num_procesos is None:
            num_procesos = 8 #multiprocessing.cpu_count() - 1
            
        # Obtener y ordenar imágenes
        imagenes = [f for f in os.listdir(input_path) if f.endswith(".jpg")]
        imagenes.sort()
        total_imagenes = len(imagenes)
        if total_imagenes == 0:
            print("Error: No se encontraron imágenes.")
            return None
            
        # Calcular fondo y avisar de la etapa
        if progress_callback_etapa:
            progress_callback_etapa("Calculando fondo...")
        fondo_final = calcular_mediana(input_path, sizeGrupo, total_imagenes, imagenes, progress_callback_especifico, num_procesos)
        #Guardar la imagen de la mediana final
        fondo_final_path = "processing_GUI/procesamiento/cache/fondo_final.jpg"
        cv2.imwrite(fondo_final_path, fondo_final)
        # Crear directorios de salida si no existen
        os.makedirs(os.path.join(output_path, "imagenes_diferencias"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "imagenes_fondo_atenuado"), exist_ok=True)
        
        # Crear cola de progreso
        progress_queue = multiprocessing.Queue()
        
        # Dividir trabajo
        segment_size = total_imagenes // num_procesos
        procesos = []
        
        # Procesar cada segmento y avisa de la etapa
        if progress_callback_etapa:
            progress_callback_etapa("Atenuando fondo...")

        for i in range(num_procesos):
            inicio = i * segment_size
            fin = (i+1) * segment_size if i < num_procesos - 1 else total_imagenes
            segmento = imagenes[inicio:fin]
            
            p = multiprocessing.Process(
                target=atenuar_fondo_imagenes_segmento,
                args=(input_path, output_path, segmento, fondo_final, factor_at, umbral_dif,
                      apertura_flag, cierre_flag,dilatacion_flag, apertura_kernel_size, cierre_kernel_size, dilatacion_kernel_size,
                      i, progress_queue)
            )
            procesos.append(p)
            p.start()
        
        # Manejar progreso
        progreso_total = [0] * num_procesos
        while any(p.is_alive() for p in procesos):
            while not progress_queue.empty():
                segment_id, progress = progress_queue.get()
                progreso_total[segment_id] = progress
                progreso_promedio = int(sum(progreso_total) / num_procesos)
                if progress_callback_especifico:
                    progress_callback_especifico(progreso_promedio)
        
        # Limpiar y esperar
        while not progress_queue.empty():
            progress_queue.get()
            
        for p in procesos:
            p.join()
            
        end_time = time.time()
        print(f"Tiempo ejecución atenuación fondo: {end_time - start_time:.2f} segundos")

        output_path_fondo_atenuado = os.path.join(output_path, "imagenes_fondo_atenuado")
        
        return output_path_fondo_atenuado
        
    except Exception as e:
        print(f"Error en la atenuación del fondo en el proceso principal: {e}")


# --------------------- FUNCIONES PARA DIVIDIR RESULTADOS POR CARPETAS ---------------------
# Función que divide las imágenes en bloques de 512 frames si esta activada 'adaptar_moment_flag' 
def dividir_bloques(input_path, output_path, img_type, progress_callback=None):

    try:
        # Obtener lista de imagenes
        imagenes = [f for f in os.listdir(input_path) if f.endswith(".jpg")]
        imagenes.sort()
        total_imagenes = len(imagenes)
        if total_imagenes == 0:
            print("Error: No se encontraron imágenes.")
            return None
        # Dividir en bloques de 512 frames
        bloque_size = 512
        num_bloques = (total_imagenes + bloque_size - 1) // bloque_size   # Añadir +bloque_size - 1 para redondear hacia arriba
        bloques_procesados = 0
        for i in range(num_bloques):
            #Crear workspace del bloque
            workspace_path = os.path.join(output_path, f"Workspace_bloque_{i+1}")
            os.makedirs(workspace_path, exist_ok=True)
            bloque_imagenes = imagenes[i*bloque_size:(i+1)*bloque_size]
            bloque_path = os.path.join(workspace_path, f"{img_type}")
            os.makedirs(bloque_path, exist_ok=True)
            for img_name in bloque_imagenes:
                img_path = os.path.join(input_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    cv2.imwrite(os.path.join(bloque_path, img_name), img)
                else:
                    print(f"Error al leer la imagen {img_name}.")
            bloques_procesados += 1
            # Enviar progreso
            if progress_callback:
                progreso = int((bloques_procesados / num_bloques) * 100)
                progress_callback(progreso)
        print(f"Imágenes divididas en bloques de {bloque_size} frames y guardadas en: {output_path}")

    except Exception as e:
        print(f"Error en la división de resultados: {e}")

    
        
