import numpy as np
from moment_scripts.generar_input_moment import (
    cargar_distribucion_espacial_5,
    cargar_coef_agrupacion,
    cargar_densidad_local,
    cargar_entropia,
    cargar_distancia_centroide_grupal,
    cargar_centroide_grupal,
    cargar_velocidades,
    cargar_dispersion_velocidades,
    cargar_porcentaje_giros,
    cargar_media_y_std_giros,
    cargar_entropia_direcciones,
    cargar_polarizacion,
    cargar_varianza_espacial,
    cargar_entropia_binaria,
    cargar_exploracion
)

# === DISTRIBUCIÓN ESPACIAL ===
arr_dist_esp = cargar_distribucion_espacial_5(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/distribucion_espacial_5.json"
)
print("--- DISTRIBUCIÓN ESPACIAL ---")
print("Shape:", arr_dist_esp.shape)
print("Ejemplo (primer canal, primeros 5 frames):", arr_dist_esp[0, :5])

# === COEFICIENTE DE AGRUPACIÓN ===
arr_coef = cargar_coef_agrupacion(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/coef_agrupacion.json"
)
print("\n--- COEFICIENTE DE AGRUPACIÓN ---")
print("Shape:", arr_coef.shape)
print("Ejemplo (primeros 5 frames):", arr_coef[0, :5])

# === DENSIDAD LOCAL ===
arr_densidad_local= cargar_densidad_local(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/densidad_local.json"
)
print("\n--- DENSIDAD LOCAL ---")
print("Shape:", arr_densidad_local.shape)
print("Ejemplo Media(primeros 5 frames):", arr_densidad_local[0, :5])
print("Ejemplo STD (primeros 5 frames):", arr_densidad_local[1, :5])

# === ENTROPÍA ===
arr_entropia = cargar_entropia(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/entropia.json"
)
print("\n--- ENTROPÍA ---")
print("Shape:", arr_entropia.shape)
print("Ejemplo (primeros 5 frames):", arr_entropia[0, :5])

# === DISTANCIA CENTROIDE GRUPAL ===
arr_dist_centroide = cargar_distancia_centroide_grupal(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/distancia_centroide_grupal.json"
)
print("\n--- DISTANCIA CENTROIDE GRUPAL ---")
print("Shape:", arr_dist_centroide.shape)
print("Ejemplo Media (primeros 5 frames):", arr_dist_centroide[0, :5])
print("Ejemplo STD (primeros 5 frames):", arr_dist_centroide[1, :5])

# === CENTROIDE GRUPAL ===
arr_centroide = cargar_centroide_grupal(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/centroide_grupal.json"
)
print("\n--- CENTROIDE GRUPAL ---")
print("Shape:", arr_centroide.shape)
print("Ejemplo eje x(primeros 5 frames):", arr_centroide[0, :5])
print("Ejemplo eje y(primeros 5 frames):", arr_centroide[1, :5])

# === VELOCIDADES ===
arr_velocidades = cargar_velocidades(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/velocidades.json"
)
print("\n--- VELOCIDADES ---")
print("Shape:", arr_velocidades.shape)
print("Ejemplo (primeros 5 frames):", arr_velocidades[0, :5])

# === DISPERSIÓN DE VELOCIDADES ===
arr_dispersion_velocidades = cargar_dispersion_velocidades(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/dispersion_velocidades.json"
)
print("\n--- DISPERSIÓN DE VELOCIDADES ---")
print("Shape:", arr_dispersion_velocidades.shape)
print("Ejemplo (primeros 5 frames):", arr_dispersion_velocidades[0, :5])

# === PORCENTAJE DE GIROS ===
arr_porcentaje_giros = cargar_porcentaje_giros(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/angulo_cambio_direccion.json"
)
print("\n--- PORCENTAJE DE GIROS ---")
print("Shape:", arr_porcentaje_giros.shape)
print("Ejemplo (primeros 5 frames):", arr_porcentaje_giros[0, :5])

# === MEDIA Y DESVIACIÓN ESTÁNDAR DE GIROS ===
arr_media_std_giros = cargar_media_y_std_giros(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/angulo_cambio_direccion.json"
)
print("\n--- MEDIA Y DESVIACIÓN ESTÁNDAR DE GIROS ---")
print("Shape:", arr_media_std_giros.shape)
print("Ejemplo Media (primeros 5 frames):", arr_media_std_giros[0, :5])
print("Ejemplo STD (primeros 5 frames):", arr_media_std_giros[1, :5])

# === ENTROPÍA DE DIRECCIONES ===
arr_entropia_direcciones = cargar_entropia_direcciones(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/direcciones.json"
)
print("\n--- ENTROPÍA DE DIRECCIONES ---")
print("Shape:", arr_entropia_direcciones.shape)
print("Ejemplo (primeros 5 frames):", arr_entropia_direcciones[0, :5])

# === POLARIZACIÓN ===
arr_polarizacion = cargar_polarizacion(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/direcciones.json"
)
print("\n--- POLARIZACIÓN ---")
print("Shape:", arr_polarizacion.shape)
print("Ejemplo (primeros 5 frames):", arr_polarizacion[0, :5])

# === VARIANZA ESPACIAL ===
arr_varianza_espacial = cargar_varianza_espacial(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/mask_stats/varianza_espacial.json"
)
print("\n--- VARIANZA ESPACIAL ---")
print("Shape:", arr_varianza_espacial.shape)
print("Ejemplo (primeros 5 frames):", arr_varianza_espacial[0, :5])

# === ENTROPIA BINARIA ===
arr_entropia_binaria = cargar_entropia_binaria(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/mask_stats/entropia_binaria_64.json"
)
print("\n--- ENTROPÍA BINARIA ---")
print("Shape:", arr_entropia_binaria.shape)
print("Ejemplo (cambio de ventana):", arr_entropia_binaria[0, 62:67]) 
print("Ejemplo (final ultima ventana):", arr_entropia_binaria[0, -5:])  # Últimos 5 frames

# === EXPLOREACIÓN ===
arr_exploracion = cargar_exploracion(
    "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/exploracion.json"
)
print("\n--- EXPLORACIÓN ---")
print("Shape:", arr_exploracion.shape)
print("Ejemplo (cambio de ventana):", arr_exploracion[0, 62:67])
print("Ejemplo (ultimo frame):", arr_exploracion[0, -5:])  # Últimos 5 frames