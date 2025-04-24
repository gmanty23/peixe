import os
import cv2
import numpy as np

# === Configuración por defecto ===
ROOT_FOLDER = "/home/gmanty/code/Workspace_prueba"
DEFAULT_BLOBS_FOLDER = os.path.join(ROOT_FOLDER, "outputs_tracking/blobs_sin_clasificar")
DEFAULT_FRAMES_FOLDER = os.path.join(ROOT_FOLDER, "imagenes_og_re")
DEFAULT_OUTPUT_FOLDER = "/home/gmanty/code/peixe/tracking/classification/dataset_blobs"
DEFAULT_ALPHA = 0.5
LABELS_PATH = "tracking/classification/dataset_blobs/labels.csv"

def obtener_siguiente_id_con_prefijo(carpeta_clase, clase):
    archivos = [f for f in os.listdir(carpeta_clase) if f.startswith(f"mask_{clase}_") and f.endswith(".png")]
    if not archivos:
        return 0
    try:
        ultimo = max(
            int(f.split(f"mask_{clase}_")[1].split(".")[0])
            for f in archivos
        )
        return ultimo + 1
    except:
        return 0

def clasificacion_desde_blobs(blob_folder, frames_folder, output_folder, alpha=0.5):
    os.makedirs(os.path.join(output_folder, "individual"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "group"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "ruido"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "reflejo"), exist_ok=True)

    blob_paths = sorted([f for f in os.listdir(blob_folder) if f.endswith(".png")])
    frame_names = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg") or f.endswith(".png")])

    for blob_name in blob_paths:
        blob_path = os.path.join(blob_folder, blob_name)
        blob_img = cv2.imread(blob_path, cv2.IMREAD_COLOR)

        if blob_img is None:
            print(f"[WARN] No se pudo cargar el blob: {blob_path}")
            continue

        # Extraer frame ID del nombre del archivo
        try:
            parts = blob_name.split("_")
            frame_id = int(parts[1])
            frame_name = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg") or f.endswith(".png")])[frame_id]
        except:
            print(f"[ERROR] No se pudo inferir el frame para: {blob_name}")
            continue

        frame_path = os.path.join(frames_folder, frame_name)
        frame_img = cv2.imread(frame_path)

        if frame_img is None or frame_img.shape != blob_img.shape:
            print(f"[WARN] Tamaño incompatible o error en: {frame_path}")
            continue

        # Reconstruir la máscara desde el blob (asume fondo negro)
        mask = cv2.cvtColor(blob_img, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)

        # Crear imagen resaltada
        overlay = frame_img.copy()
        color = (0, 255, 0)  # Amarillo
        for c in range(3):
            overlay[:, :, c][mask > 0] = (
                alpha * color[c] + (1 - alpha) * overlay[:, :, c][mask > 0]
            ).astype(np.uint8)

        # Mostrar nombre
        cv2.putText(overlay, blob_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Inicialización de imágenes
        scale = 0.6
        overlay_resized = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
        original_resized = cv2.resize(frame_img, (0, 0), fx=scale, fy=scale)
        imagen_actual = overlay_resized
        modo_visualizacion = "blob"
        navegando = False
        frame_idx_actual = frame_id 

        # Mostrar por primera vez
        cv2.imshow("Clasifica: [i] Individual | [g] Grupo | [r] Ruido | [q] Salir", imagen_actual)

        while True:
            key = cv2.waitKey(0) & 0xFF

            #print(f"[DEBUG] Tecla pulsada: {key}")

            if key == 32:  # barra espaciadora → alterna overlay/original solo si en modo "blob"
                if navegando:
                    imagen_actual = overlay_resized
                    modo_visualizacion = "blob"
                    navegando = False
                    frame_idx_actual = frame_id  # reset al blob
                else:
                    imagen_actual = original_resized if np.array_equal(imagen_actual, overlay_resized) else overlay_resized
                cv2.imshow("Clasifica: [i] Individual | [g] Grupo | [r] Ruido | [q] Salir", imagen_actual)

            elif key == 81:  # ← flecha izquierda → mostrar frame anterior
                if frame_idx_actual > 0:
                    frame_idx_actual -= 1
                    path = os.path.join(frames_folder, frame_names[frame_idx_actual])
                    frame = cv2.imread(path)
                    if frame is not None:
                        imagen_actual = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                        modo_visualizacion = "frame"
                        navegando = True
                        cv2.imshow("Clasifica: [i] Individual | [g] Grupo | [r] Ruido | [q] Salir", imagen_actual)

            elif key == 83:  # → flecha derecha → mostrar frame siguiente
                if frame_idx_actual < len(frame_names) - 1:
                    frame_idx_actual += 1
                    path = os.path.join(frames_folder, frame_names[frame_idx_actual])
                    frame = cv2.imread(path)
                    if frame is not None:
                        imagen_actual = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                        modo_visualizacion = "frame"
                        navegando = True
                        cv2.imshow("Clasifica: [i] Individual | [g] Grupo | [r] Ruido | [q] Salir", imagen_actual)

            elif key in [ord("i"), ord("g"), ord("n"), ord("q"), ord("r")] and not navegando:
                break


        if key == ord("q"):
            print("Saliendo del clasificador.")
            break
        elif key == ord("r"):
            clase = "ruido"
        elif key == ord("i"):
            clase = "individual"
        elif key == ord("g"):
            clase = "group"
        # elif key == ord("r"):
        #     clase = "reflejo"
        else:
            print("Tecla no válida, saltando.")
            continue

        # Guardar imagen con nombre único por clase y prefijo claro
        carpeta_clase = os.path.join(output_folder, clase)
        nuevo_id = obtener_siguiente_id_con_prefijo(carpeta_clase, clase)
        nombre_archivo = f"mask_{clase}_{nuevo_id:06d}.png"
        destino = os.path.join(carpeta_clase, nombre_archivo)
        cv2.imwrite(destino, blob_img)
        print(f"Guardado en {destino}")

        # Guardar entrada en CSV con centroide
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0  # centroide indefinido si máscara vacía
        with open(LABELS_PATH, "a") as f:
            if os.path.getsize(LABELS_PATH) == 0:
                f.write("filename,class,cx,cy\n")
            f.write(f"{nombre_archivo},{clase},{cx},{cy}\n")

        os.remove(blob_path) # Eliminar blob original una vez clasificado

    cv2.destroyAllWindows()


# === Main ejecutable ===
if __name__ == "__main__":
    print(f"[INFO] Iniciando clasificador de blobs con visualización sobre imagen original...")
    print(f"[INFO] Carpeta de blobs: {DEFAULT_BLOBS_FOLDER}")
    print(f"[INFO] Carpeta de frames originales: {DEFAULT_FRAMES_FOLDER}")
    print(f"[INFO] Carpeta de salida: {DEFAULT_OUTPUT_FOLDER}")
    print(f"[INFO] Transparencia del resaltado: {DEFAULT_ALPHA}")
    clasificacion_desde_blobs(DEFAULT_BLOBS_FOLDER, DEFAULT_FRAMES_FOLDER, DEFAULT_OUTPUT_FOLDER, DEFAULT_ALPHA)
