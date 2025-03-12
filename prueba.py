import cv2

# Establecer el códec de salida (en este caso, mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # O puedes usar 'XVID', 'H264', etc.

# Crear un VideoWriter para guardar el video
output_path = "output_video.mp4"
fps = 25  # Asegúrate de usar la misma tasa de fotogramas que el video de entrada
width = 3840  # Ancho de la resolución
height = 2160  # Altura de la resolución

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Abrir un archivo de video de entrada
vid = cv2.VideoCapture("/home/gms/Downloads/USCF1-140644-141144.mp4")
while True:
    ret, frame = vid.read()
    if not ret:
        break
    out.write(frame)  # Escribir el frame redimensionado o sin modificar

vid.release()
out.release()
