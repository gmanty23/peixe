import cv2
import numpy as np

# Cargar el video (cambia 'video.mp4' por tu video o usa 0 para webcam)
video_path = "/home/gmanty/code/output_20s.mp4"
cap = cv2.VideoCapture(video_path)

# Crear sustractores de fondo
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

# Leer los primeros 50 frames para calcular el fondo (para absdiff)
background_frames = []
num_background_frames = 50

for i in range(num_background_frames):
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background_frames.append(gray_frame)

# Calcular la imagen de fondo usando la mediana
if background_frames:
    background = np.median(np.array(background_frames), axis=0).astype(dtype=np.uint8)
else:
    background = None

# Reiniciar el video para procesarlo de nuevo
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar MOG2
    fg_mog2 = mog2.apply(gray_frame)
    
    # Aplicar KNN
    fg_knn = knn.apply(gray_frame)
    
    # Aplicar absdiff si hay fondo calculado
    if background is not None:
        fg_absdiff = cv2.absdiff(gray_frame, background)
        _, fg_absdiff = cv2.threshold(fg_absdiff, 30, 255, cv2.THRESH_BINARY)
    else:
        fg_absdiff = np.zeros_like(gray_frame)

    # Mostrar resultados
    cv2.imshow("Original", frame)
    cv2.imshow("MOG2", fg_mog2)
    cv2.imshow("KNN", fg_knn)
    cv2.imshow("AbsDiff", fg_absdiff)
    
    # Salir con 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
