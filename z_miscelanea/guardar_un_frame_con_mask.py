import cv2
import numpy as np

def guardar_frame_con_mascaras(video_path, frame_num, output_base_path, mask_path):
    # Bounding box
    x, y, w, h = 550, 960, 2225, 1186
    target_size = (1920, 1080)
    
    # Abre el vídeo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el vídeo {video_path}")
        return
    
    # Mueve al frame deseado
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: no se pudo leer el frame {frame_num}")
        cap.release()
        return
    
    # Recorta el frame
    frame_cropped = frame[y:y+h, x:x+w]
    
    # Redimensiona a 1920x1080
    frame_resized = cv2.resize(frame_cropped, target_size)
    
    # Carga la máscara (ya está en 1920x1080)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: no se pudo cargar la máscara {mask_path}")
        cap.release()
        return
    
    # Guarda el frame original
    cv2.imwrite(f"{output_base_path}_original.png", frame_resized)
    print(f"Guardado: {output_base_path}_original.png")
    
    # Aplica sombra translúcida
    translucent = frame_resized.copy()
    color = (0, 0, 200)  # Rojo BGR
    alpha = 0.4
    translucent[mask > 0] = (
        (1 - alpha) * translucent[mask > 0] + alpha * np.array(color)
    ).astype(np.uint8)
    cv2.imwrite(f"{output_base_path}_translucido.png", translucent)
    print(f"Guardado: {output_base_path}_translucido.png")
    
    # Aplica contorno
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured = translucent.copy()
    cv2.drawContours(contoured, contours, -1, (0, 0, 150), 2)
    cv2.imwrite(f"{output_base_path}_translucido_contorno.png", contoured)
    print(f"Guardado: {output_base_path}_translucido_contorno.png")
    
    # Aplica hachurado
    hatch_pattern = np.zeros_like(mask)
    spacing = 10
    for i in range(-mask.shape[1], mask.shape[0], spacing):
        cv2.line(hatch_pattern, (max(0, i), max(0, -i)), 
                 (min(mask.shape[1], mask.shape[1] + i), min(mask.shape[0], mask.shape[0] + i)), 
                 255, 1)
    
    hatch_mask = cv2.bitwise_and(hatch_pattern, mask)
    hatched = frame_resized.copy()
    hatched[hatch_mask > 0] = (0, 0, 255)
    cv2.imwrite(f"{output_base_path}_hachurado.png", hatched)
    print(f"Guardado: {output_base_path}_hachurado.png")
    
    cap.release()

# Ejemplo de uso
if __name__ == "__main__":
    video_path = "/home/gmanty/code/AnemoNAS/07-12-23/1812-2002/USCL2-190229-190729.mp4"
    frame_num = 220
    output_base_path = "frame_220"
    mask_path = "processing_GUI/procesamiento/zona_no_valida.png"

    
    guardar_frame_con_mascaras(video_path, frame_num, output_base_path, mask_path)
