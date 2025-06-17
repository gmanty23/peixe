import os

def renombrar_archivos_especiales(directorio):
    if not os.path.isdir(directorio):
        return
    
    for archivo in os.listdir(directorio):
        if archivo.endswith(".json") and ("(" in archivo or ")" in archivo or "," in archivo or " " in archivo):
            nuevo_nombre = archivo.replace("(", "").replace(")", "").replace(",", "_").replace(" ", "")
            path_actual = os.path.join(directorio, archivo)
            path_nuevo = os.path.join(directorio, nuevo_nombre)
            os.rename(path_actual, path_nuevo)
            # print(f"Renombrado: {path_actual} ➡ {path_nuevo}")

def procesar_subcarpetas(carpeta_trabajo):
    # Recorremos todas las subcarpetas
    for root, dirs, files in os.walk(carpeta_trabajo):
        # Procesar solo las carpetas de primer nivel dentro de la carpeta de trabajo
        for subcarpeta in dirs:
            subcarpeta_path = os.path.join(root, subcarpeta)
            print(f"\nProcesando subcarpeta: {subcarpeta_path}")
            
            # 1️⃣ Eliminar archivos en bbox_stats
            bbox_stats_path = os.path.join(subcarpeta_path, "bbox_stats")
            archivos_bbox_stats = [
                "distribucion_espacial_5.json",
                "distribucion_espacial_10.json",
                "distribucion_espacial_15.json",
                "distribucion_espacial_20.json"
            ]
            for archivo in archivos_bbox_stats:
                archivo_path = os.path.join(bbox_stats_path, archivo)
                if os.path.exists(archivo_path):
                    os.remove(archivo_path)
                    # print(f"Eliminado: {archivo_path}")
                else:
                    print(f"No encontrado (no eliminado): {archivo_path}")
            
            # Renombrar archivos con caracteres especiales en bbox_stats
            renombrar_archivos_especiales(bbox_stats_path)

            # 2️⃣ Eliminar archivos en mask_stats
            masks_stats_path = os.path.join(subcarpeta_path, "mask_stats")
            archivos_masks_stats = [
                "densidad_5.json",
                "densidad_10.json",
                "densidad_15.json",
                "densidad_20.json"
            ]
            for archivo in archivos_masks_stats:
                archivo_path = os.path.join(masks_stats_path, archivo)
                if os.path.exists(archivo_path):
                    os.remove(archivo_path)
                    # print(f"Eliminado: {archivo_path}")
                else:
                    print(f"No encontrado (no eliminado): {archivo_path}")

            # Renombrar archivos con caracteres especiales en mask_stats
            renombrar_archivos_especiales(masks_stats_path)

            # 3️⃣ Comprobar archivos en bbox
            bbox_path = os.path.join(subcarpeta_path, "bbox")
            archivos_bbox = [
                "output_dims.json",
                "recorte_yolo.json"
            ]
            for archivo in archivos_bbox:
                archivo_path = os.path.join(bbox_path, archivo)
                # if os.path.exists(archivo_path):
                #     print(f"✅ Encontrado: {archivo_path}")
                # else:
                #     print(f"❌ Faltante: {archivo_path}")
                if not os.path.exists(archivo_path):
                    print(f"❌ Faltante: {archivo_path}")

            # 4️⃣ Comprobar archivos en masks_rle
            masks_rle_path = os.path.join(subcarpeta_path, "masks_rle")
            archivos_masks_rle = [
                "output_dims.json",
                "recorte_morphology.json"
            ]
            for archivo in archivos_masks_rle:
                archivo_path = os.path.join(masks_rle_path, archivo)
                # if os.path.exists(archivo_path):
                #     print(f"✅ Encontrado: {archivo_path}")
                # else:
                #     print(f"❌ Faltante: {archivo_path}")
                if not os.path.exists(archivo_path):
                    print(f"❌ Faltante: {archivo_path}")
        
        # No seguir recorriendo en profundidad
        break

if __name__ == "__main__":
    # 🔹 Define aquí la carpeta de trabajo
    carpeta_trabajo = "/home/gmanty/code/AnemoNAS/07-12-23/1812-2002"

    if os.path.isdir(carpeta_trabajo):
        procesar_subcarpetas(carpeta_trabajo)
    else:
        print(f"❌ El path proporcionado no es una carpeta válida: {carpeta_trabajo}")
