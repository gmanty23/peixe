# import cv2
# import numpy as np

# # Cargar el video
# #video_path = "/home/gmanty/code/output_20s.mp4"
# video_path = "/home/gms/AnemoNAS/Workspace/output_20s.mp4"
# cap = cv2.VideoCapture(video_path)

# # Verificar si el video se abrió correctamente
# if not cap.isOpened():
#     print("Error: No se pudo abrir el video.")
#     exit()

# # Crear sustractores de fondo
# mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
# knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

# # Leer varios frames para calcular el fondo (mediana)
# background_frames = []
# num_background_frames = 50  # Número de frames usados para el cálculo del fondo

# for i in range(num_background_frames):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     background_frames.append(gray_frame)
    
# # Calcular el fondo como la mediana de los frames capturados
# if background_frames:
#     background = np.median(np.array(background_frames), axis=0).astype(dtype=np.uint8)
# else:
#     print("Error: No se pudieron obtener suficientes frames para calcular el fondo.")
#     cap.release()
#     exit()

# # Reiniciar el video y leer el primer frame
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# ret, frame = cap.read()
# if not ret:
#     print("Error: No se pudo leer el primer frame.")
#     cap.release()
#     exit()

# gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # Aplicar MOG2 y KNN al primer frame
# fg_mog2 = mog2.apply(gray_frame)
# fg_knn = knn.apply(gray_frame)

# # Aplicar absdiff con el fondo calculado
# fg_absdiff = cv2.absdiff(gray_frame, background)
# _, fg_absdiff = cv2.threshold(fg_absdiff, 30, 255, cv2.THRESH_BINARY)

# # Reducir tamaño de las imágenes para mostrarlas mejor
# scale = 0.5  # Reducir al 50%
# frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
# fg_mog2_resized = cv2.resize(fg_mog2, None, fx=scale, fy=scale)
# fg_knn_resized = cv2.resize(fg_knn, None, fx=scale, fy=scale)
# fg_absdiff_resized = cv2.resize(fg_absdiff, None, fx=scale, fy=scale)

# # Mostrar resultados
# cv2.imshow("Original", frame_resized)
# cv2.imshow("MOG2", fg_mog2_resized)
# cv2.imshow("KNN", fg_knn_resized)
# cv2.imshow("AbsDiff", fg_absdiff_resized)

# # Esperar tecla y cerrar
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Liberar video
# cap.release()

# import cv2
# import os

# # Ruta de las imágenes
# input_path = "/home/gms/AnemoNAS/Workspace/imagenes_resized/"

# # Filtrar solo imágenes (JPG, PNG, etc.)
# imagenes = sorted([f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

# # Parámetros del sustractor
# sizeGrupo = min(499, len(imagenes))  # Evita error si hay menos de 499 imágenes
# substractor_knn = cv2.createBackgroundSubtractorKNN(history=499, dist2Threshold=3, detectShadows=False)

# # Alimentar el modelo con imágenes
# for imagen in imagenes[:sizeGrupo]:
#     img_path = os.path.join(input_path, imagen)
#     img = cv2.imread(img_path)
    
#     if img is None:
#         print(f"Error al leer {imagen}, se omite.")
#         continue
    
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     substractor_knn.apply(gray_img)  # Aprende el fondo

# # Obtener el fondo calculado
# fondo_final = substractor_knn.getBackgroundImage()

# # Verificar si el fondo fue generado
# if fondo_final is None:
#     print("Error: No se pudo calcular el fondo.")
# else:
#     # Aplicar filtro de mediana para reducir ruido impulsivo
#     fondo_final_filtrado = cv2.medianBlur(fondo_final, 5)  # Kernel 5x5
    
#     # Mostrar resultados
#     fondo_final = cv2.cvtColor(fondo_final, cv2.COLOR_GRAY2BGR)
#     fondo_final_filtrado = cv2.cvtColor(fondo_final_filtrado, cv2.COLOR_GRAY2BGR)
#     cv2.imshow("Fondo final original", fondo_final)
#     cv2.imshow("Fondo final filtrado", fondo_final_filtrado)
#     cv2.waitKey(0)  # Espera a que el usuario presione una tecla
#     cv2.destroyAllWindows()  # Cierra las ventanas abiertas

import cv2
import os

# Ruta de las imágenes
input_path = "/home/gms/AnemoNAS/Workspace/imagenes_resized/"

# Filtrar solo imágenes (JPG, PNG, etc.)
imagenes = sorted([f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

# Parámetros del sustractor MOG2
sizeGrupo = len(imagenes)  # Evita error si hay menos de 499 imágenes
substractor_mog2 = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=3, detectShadows=False)
i = 0
# Alimentar el modelo con imágenes a color
for imagen in imagenes[:sizeGrupo]:
    img_path = os.path.join(input_path, imagen)
    img = cv2.imread(img_path)
    print(f"Imagen {i}")
    i += 1
    
    if img is None:
        print(f"Error al leer {imagen}, se omite.")
        continue
    
    substractor_mog2.apply(img)  # Aprende el fondo con imágenes a color

# Obtener el fondo calculado
fondo_final = substractor_mog2.getBackgroundImage()

# Verificar si el fondo fue generado
if fondo_final is None:
    print("Error: No se pudo calcular el fondo.")
else:
    # Aplicar filtro de mediana para reducir ruido impulsivo

    # Mostrar resultados (no es necesario convertir a BGR)
    cv2.imshow("Fondo final original (MOG2, Color)", fondo_final)
    cv2.waitKey(0)  # Espera a que el usuario presione una tecla
    cv2.destroyAllWindows()  # Cierra las ventanas abiertas
