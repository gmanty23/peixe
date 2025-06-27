import cv2
import numpy as np
import os

# Cargar la imagen de fondo, la pecera sin los peces
fondo = cv2.imread("/home/gms/AnemoNAS/temp/fondo_median_final.png")

#Abrir el video
video = cv2.VideoCapture("/home/gms/AnemoNAS/temp/USCL2-195223-195723.mp4")
if not video.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

#Comparamos las dimensiones del video y del fondo para asegurarnos de que son las mismas
alto_fondo, ancho_fondo = fondo.shape[:2]

ret, frame = video.read()
if not ret:
    print("Error al leer el primer fotograma del video")
    video.release()
    exit()
alto_video, ancho_video = frame.shape[:2]

if (alto_fondo != alto_video) or (ancho_fondo != ancho_video):
    print("Las dimensiones del video y la imagen de fondo no coinciden. No se realizará el procesamiento. Las dimensiones del video son: {alto_video}x{ancho_video} y las dimensiones de la imagen de fondo son: {alto_fondo}x{ancho_fondo}")
    video.release()
    exit()
print(f"Las dimensiones del video y la imagen de fondo coinciden: {alto_video}x{ancho_video}")
    
    
 # Crear los subdirectorios si no existen
os.makedirs('diferencias', exist_ok=True)
os.makedirs('peces', exist_ok=True)
os.makedirs('peces_con_fondo_atenuado', exist_ok=True)   

# Volver al inicio del video
video.set(cv2.CAP_PROP_POS_FRAMES, 0)


# Establecer el tamaño de la ventana
ancho_ventana = 1920
alto_ventana = 1080
cv2.namedWindow("Diferencia", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Diferencia", ancho_ventana, alto_ventana)

cv2.namedWindow("Peces", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Peces", ancho_ventana, alto_ventana)

cv2.namedWindow("Peces con fondo atenuado", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Peces con fondo atenuado", ancho_ventana, alto_ventana)

factor_atenuacion = 0.4  # Reducir el fondo a un x% de su intensidad original
kernel = np.ones((10,10), np.uint8)  # Kernel para procesado con operaciones morfológicas
frame_count = 0  # Contador para numerar las imágenes

while True:
    # Leer el siguiente fotograma del video
    ret, frame = video.read()
    
    # Si no se pudo leer el fotograma, significa que el video ha terminado
    if not ret:
        break

    # Convertir el fotograma y el fondo a escala de grises para simplificar
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fondo_gray = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY)

    # Restar el fondo del fotograma
    diferencia = cv2.absdiff(frame_gray, fondo_gray)

    # Aplicar un umbral para resaltar las diferencias (los peces)
    _, diferencia_umbral = cv2.threshold(diferencia, 7, 255, cv2.THRESH_BINARY)


    # Mostrar la diferencia
    cv2.imshow("Diferencia", diferencia_umbral)  #Modo video
    # cv2.imshow("Diferencia", diferencia_umbral) # Modo primer frame
    # cv2.waitKey(0)  # Modo primer frame
    # cv2.imwrite(f'substraccion_fondo.jpg', diferencia_umbral)

    
    # Aplicar apertura para eliminar pequeños puntos de ruido
    diferencia_umbral = cv2.morphologyEx(diferencia_umbral, cv2.MORPH_OPEN, kernel)
    
    # cv2.imshow("Diferencia", diferencia_umbral)
    # cv2.waitKey(0)  
    # cv2.imwrite(f'substraccion_fondo+apertura.jpg', diferencia_umbral)

    #El cierre me rellena los agujeros entre peces
    # # Aplicar cierre para rellenar pequeños huecos en los peces detectados
    # diferencia_umbral = cv2.morphologyEx(diferencia_umbral, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow("Diferencia", diferencia_umbral)
    # cv2.waitKey(0) 
    # cv2.imwrite(f'substraccion_fondo+aperturacierre.jpg', diferencia_umbral)

    #Igual util como post procesado y no como preprocesado
    # # Crear una copia de la imagen para rellenar huecos
    # im_floodfill = diferencia_umbral.copy()
    # # Crear una máscara para floodFill (debe ser 2 píxeles más grande que la imagen)
    # h, w = diferencia_umbral.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)
    # # Rellenar los huecos dentro de los peces con floodFill
    # cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # # Invertir la imagen rellenada
    # im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # # Combinar con la imagen original para obtener solo los huecos rellenos
    # diferencia_umbral = diferencia_umbral | im_floodfill_inv
    # # Mostrar la imagen después del relleno de huecos
    # cv2.imshow("Diferencia", diferencia_umbral)
    # cv2.waitKey(0)
    # cv2.imwrite(f'substraccion_fondo+apertura+floodfill.jpg', diferencia_umbral)

        
    # Mostrar el fotograma original con las diferencias resaltadas
    resultado = cv2.bitwise_and(frame, frame, mask=diferencia_umbral)
    cv2.imshow("Peces", resultado) # Modo video

    # Crear una máscara invertida para atenuar el fondo (donde no hay peces)
    fondo_mask = cv2.bitwise_not(diferencia_umbral)
    
    #Atenuar el fondo multiplicando el fondo por el factor de atenuación
    fondo_atenuado = frame * factor_atenuacion  # Apagar el fondo multiplicando por el factor de atenuación

    # Crear la imagen final: las áreas con los peces se mantienen intactas, y las del fondo se atenúan
    resultado_con_fondo_atenuado = cv2.bitwise_and(frame, frame, mask=diferencia_umbral)  # Los peces se mantienen
    resultado_con_fondo_atenuado += cv2.bitwise_and(fondo_atenuado.astype(np.uint8), fondo_atenuado.astype(np.uint8), mask=fondo_mask)  # El fondo se atenúa

    # Mostrar la imagen con el fondo atenuado
    cv2.imshow("Peces con fondo atenuado", resultado_con_fondo_atenuado)
    
    # Guardar las imágenes en los subdirectorios correspondientes
    #cv2.imwrite(f'/home/gms/AnemoNAS/temp/med_diferencias/diferencias_{frame_count:04d}.jpg', diferencia_umbral)
    #cv2.imwrite(f'/home/gms/AnemoNAS/temp/med_peces/med_peces_{frame_count:04d}.jpg', resultado)
    #cv2.imwrite(f'/home/gms/AnemoNAS/temp/med_peces_atenuado/peces_atenuados_{frame_count:04d}.jpg', resultado_con_fondo_atenuado)

    frame_count += 1
    
    # Esperar a que el usuario presione una tecla para continuar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el video y cerrar las ventanas de OpenCV
video.release()
cv2.destroyAllWindows()