import os
import cv2
import numpy as np

# === Configuración ===
OUTPUT_DIR = "dataset_blobs"
ALPHA = 0.5  # Transparencia del resaltado (0.0 a 1.0)
os.makedirs(os.path.join(OUTPUT_DIR, "individual"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "group"), exist_ok=True)

def clasificacion_interactiva(tracking_graph, img_paths):
    for frame_idx, blobs in tracking_graph.items():
        frame = cv2.imread(img_paths[frame_idx])
        
        for blob_idx, blob in enumerate(blobs):
            mask = blob["mask"].astype(bool)
            overlay = frame.copy()

            # Color de resaltado
            color = (0, 255, 255)  # Amarillo

            # Aplicar color al blob
            for c in range(3):
                overlay[:, :, c][mask] = (
                    ALPHA * color[c] + (1 - ALPHA) * overlay[:, :, c][mask]
                ).astype(np.uint8)

            # Texto con frame y blob ID
            cx, cy = blob["centroid"]
            text = f"F{frame_idx}_B{blob_idx}"
            cv2.putText(overlay, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Mostrar imagen
            cv2.imshow("Clasifica: [i] Individual | [g] Grupo | [s] Saltar | [q] Salir", overlay)
            key = cv2.waitKey(0) & 0xFF

            # Acción según tecla
            if key == ord("q"):
                print("Saliendo del clasificador.")
                cv2.destroyAllWindows()
                return

            elif key == ord("s"):
                continue  # saltar sin guardar

            elif key == ord("i"):
                clase = "individual"

            elif key == ord("g"):
                clase = "group"

            else:
                print("Tecla no válida, saltando.")
                continue

            # Guardar recorte resaltado
            filename = f"frame_{frame_idx:04d}_blob_{blob_idx:02d}.png"
            path = os.path.join(OUTPUT_DIR, clase, filename)
            cv2.imwrite(path, overlay)
            print(f"Guardado en: {path}")

    cv2.destroyAllWindows()
