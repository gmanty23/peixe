import cv2

def extraer_frame(video_path, frame_num):
    # Abrir el video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    # Establecer el frame deseado
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # Leer ese frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: No se pudo leer el frame {frame_num}.")
        return

    # Mostrar el frame (opcional)
    cv2.imshow(f'Frame {frame_num}', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar el frame como imagen en la carpeta del video


    output_filename = f'frame_{frame_num}.jpg'
    cv2.imwrite(output_filename, frame)
    print(f"Frame {frame_num} guardado como {output_filename}")

    # Liberar el video
    cap.release()

# Ejemplo de uso:
# Cambia el path y el número de frame según necesites
video_path = '/home/gmanty/code/calculos_memoria/_temp_/USCL2-184715-185215_p1.mp4'
frame_num = 740  # por ejemplo, frame 100 (contando desde 0)
extraer_frame(video_path, frame_num)
