import cv2
import numpy as np
import os
from tqdm import tqdm

# ==== CONFIGURA AQUÍ LAS RUTAS DE LAS CARPETAS ====
carpeta_original = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/masks_prueba"
carpeta_decodificada = "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/masks_rle_decode"
# ===================================================

def comparar_mascaras(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"⚠️ No se pudo leer: {img1_path} o {img2_path}")
        return None

    if img1.shape != img2.shape:
        print(f"❌ Tamaños distintos: {img1.shape} vs {img2.shape}")
        return None

    diff = cv2.absdiff(img1, img2)
    n_diferencias = np.count_nonzero(diff)
    total_pixeles = diff.size
    porcentaje = 100 * n_diferencias / total_pixeles

    return n_diferencias, porcentaje

if __name__ == "__main__":
    archivos = sorted([f for f in os.listdir(carpeta_original) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    totales = []
    fallos = 0

    for archivo in tqdm(archivos, desc="Comparando máscaras"):
        path1 = os.path.join(carpeta_original, archivo)
        path2 = os.path.join(carpeta_decodificada, os.path.splitext(archivo)[0] + ".png")  # asumimos salida en .png

        resultado = comparar_mascaras(path1, path2)
        if resultado is not None:
            n_diff, pct = resultado
            totales.append((archivo, n_diff, pct))
        else:
            fallos += 1

    # Resultados
    print(f"\n📊 Comparación completada: {len(totales)} imágenes correctas, {fallos} fallidas")

    if totales:
        promedio = np.mean([x[2] for x in totales])
        max_dif = max(totales, key=lambda x: x[2])
        print(f"✅ Porcentaje medio de diferencia: {promedio:.4f}%")
        print(f"🔍 Imagen con mayor diferencia: {max_dif[0]} ({max_dif[2]:.4f}%)")

        # Mostrar las peores 5
        peores = sorted(totales, key=lambda x: x[2], reverse=True)[:5]
        print("\nTop 5 imágenes con más diferencia:")
        for nombre, ndiff, pct in peores:
            print(f" - {nombre}: {ndiff} px diferentes ({pct:.4f}%)")
