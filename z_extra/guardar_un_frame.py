import cv2

def guardar_frame(video_path, frame_num, output_path):
    # Abre el vídeo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el vídeo {video_path}")
        return
    
    # Mueve el puntero al frame deseado
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    # Lee el frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: no se pudo leer el frame {frame_num}")
        cap.release()
        return
    
    # Guarda el frame como imagen
    cv2.imwrite(output_path, frame)
    print(f"Frame {frame_num} guardado en {output_path}")
    
    cap.release()

# Ejemplo de uso
if __name__ == "__main__":
    video_path = "/home/gmanty/code/AnemoNAS/07-12-23/1812-2002/USCL2-190229-190729.mp4"
    frame_num = 220  # número de frame que quieres guardar
    output_path = "frame_220.png"
    
    guardar_frame(video_path, frame_num, output_path)
