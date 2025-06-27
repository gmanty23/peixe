import cv2

# Ruta de la imagen
imagen_path = '/home/gmanty/code/calculos_memoria/_temp_/USCL2-184715-185215_p1/frame740.jpg'

# Leer la imagen
imagen = cv2.imread(imagen_path)

# Verifica que la imagen se cargó
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
else:
    # Bounding box: x, y, w, h
    x, y, w, h = 550, 960, 2225, 1186

    # Recortar la imagen
    recorte = imagen[y:y+h, x:x+w]

    # Redimensionar a 1920x1080
    redimensionada = cv2.resize(recorte, (1920, 1080))

    # Guardar el resultado
    cv2.imwrite('imagen_recortada_redimensionada.jpg', redimensionada)

    # Mostrar el resultado (opcional)
    cv2.imshow('Imagen Recortada y Redimensionada', redimensionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
