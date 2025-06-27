import cv2
import numpy as np

# Paths a la imagen y la máscara
imagen_path = '/home/gmanty/code/calculos_memoria/frame0000.jpg'
mascara_path = '/home/gmanty/code/calculos_memoria/frame_00000.png'

# Cargar la imagen y la máscara a color
imagen = cv2.imread(imagen_path)
mascara_color = cv2.imread(mascara_path)

if imagen is None:
    print("Error: No se pudo cargar la imagen.")
elif mascara_color is None:
    print("Error: No se pudo cargar la máscara.")
else:
    # Redimensionar máscara si no coincide
    if (mascara_color.shape[0] != imagen.shape[0]) or (mascara_color.shape[1] != imagen.shape[1]):
        mascara_color = cv2.resize(mascara_color, (imagen.shape[1], imagen.shape[0]))

    # Convertir la máscara a escala de grises (para poder binarizarla)
    mascara_gray = cv2.cvtColor(mascara_color, cv2.COLOR_BGR2GRAY)

    # Binarizar: cualquier valor > 0 pasa a 255 (máscara activa)
    _, mascara_bin = cv2.threshold(mascara_gray, 0, 255, cv2.THRESH_BINARY)

    # Crear imagen de color "Calippo" (verde-lima)
    calippo_color = (0, 255, 128)  # BGR verde-lima
    calippo_img = np.full(imagen.shape, calippo_color, dtype=np.uint8)

    # Aplicar la máscara al color
    calippo_masked = cv2.bitwise_and(calippo_img, calippo_img, mask=mascara_bin)

    # Mezcla translúcida: combinar calippo_masked con la imagen original
    mezcla = cv2.addWeighted(imagen, 1.0, calippo_masked, 0.5, 0)

    # Combinar de forma correcta: donde hay máscara, usa la mezcla; donde no, la imagen original
    final = np.where(mascara_bin[..., None] == 255, mezcla, imagen)

    # Mostrar el resultado
    cv2.imshow('Imagen con máscara Calippo translúcida', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar el resultado
    cv2.imwrite('imagen_calippo_translucida.jpg', final)
    print("Imagen guardada como 'imagen_calippo_translucida.jpg'.")
