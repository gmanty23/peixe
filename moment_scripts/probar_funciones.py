import numpy as np
from moment_scripts.generar_input_moment import (
    cargar_areas_blobs,
    cargar_distribucion_espacial,
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
    cargar_exploracion,
    cargar_velocidad_centroide,
    cargar_centro_masa,
    cargar_densidad,
    cargar_dispersion_px,
    cargar_velocidad_grupo,
    cargar_persistencia,
    
)

# == AREAS BLOBS ==
arr_areas_blobs = cargar_areas_blobs(
    "/home/gms/AnemoNAS/prueba_GUI/output_5s/bbox_stats/areas_blobs.json"
)
print("--- AREAS BLOBS ---")
print("Shape:", arr_areas_blobs.shape)
print("Ejemplo media(primeros 5 frames):", arr_areas_blobs[0, :5])
print("Ejemplo std(primeros 5 frames):", arr_areas_blobs[1, :5])


# === DISTRIBUCIÓN ESPACIAL ===
arr_dist_esp = cargar_distribucion_espacial(
    "/home/gms/AnemoNAS/prueba_GUI/output_5s/bbox_stats/distribucion_espacial_(2, 4).json"
)
print("--- DISTRIBUCIÓN ESPACIAL ---")
print("Shape:", arr_dist_esp.shape)
print("Ejemplo (primer canal, primeros 5 frames):", arr_dist_esp[0, :5])

# == VELOCIDAD CENTROIDE GRUPO ==
arr_vel_centroide = cargar_velocidad_centroide(
    "/home/gms/AnemoNAS/prueba_GUI/output_5s/bbox_stats/velocidad_centroide.json"
)
print("--- VELOCIDAD CENTROIDE GRUPO ---")
print("Shape:", arr_vel_centroide.shape)
print("Ejemplo (primeros 5 frames):", arr_vel_centroide[0, :5])


# # === COEFICIENTE DE AGRUPACIÓN ===
# arr_coef = cargar_coef_agrupacion(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/coef_agrupacion.json"
# )
# print("\n--- COEFICIENTE DE AGRUPACIÓN ---")
# print("Shape:", arr_coef.shape)
# print("Ejemplo (primeros 5 frames):", arr_coef[0, :5])

# # === DENSIDAD LOCAL ===
# arr_densidad_local= cargar_densidad_local(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/densidad_local.json"
# )
# print("\n--- DENSIDAD LOCAL ---")
# print("Shape:", arr_densidad_local.shape)
# print("Ejemplo Media(primeros 5 frames):", arr_densidad_local[0, :5])
# print("Ejemplo STD (primeros 5 frames):", arr_densidad_local[1, :5])

# # === ENTROPÍA ===
# arr_entropia = cargar_entropia(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/entropia.json"
# )
# print("\n--- ENTROPÍA ---")
# print("Shape:", arr_entropia.shape)
# print("Ejemplo (primeros 5 frames):", arr_entropia[0, :5])

# # === DISTANCIA CENTROIDE GRUPAL ===
# arr_dist_centroide = cargar_distancia_centroide_grupal(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/distancia_centroide_grupal.json"
# )
# print("\n--- DISTANCIA CENTROIDE GRUPAL ---")
# print("Shape:", arr_dist_centroide.shape)
# print("Ejemplo Media (primeros 5 frames):", arr_dist_centroide[0, :5])
# print("Ejemplo STD (primeros 5 frames):", arr_dist_centroide[1, :5])

# # === CENTROIDE GRUPAL ===
# arr_centroide = cargar_centroide_grupal(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/centroide_grupal.json"
# )
# print("\n--- CENTROIDE GRUPAL ---")
# print("Shape:", arr_centroide.shape)
# print("Ejemplo eje x(primeros 5 frames):", arr_centroide[0, :5])
# print("Ejemplo eje y(primeros 5 frames):", arr_centroide[1, :5])

# # === VELOCIDADES ===
# arr_velocidades = cargar_velocidades(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/velocidades.json"
# )
# print("\n--- VELOCIDADES ---")
# print("Shape:", arr_velocidades.shape)
# print("Ejemplo (primeros 5 frames):", arr_velocidades[0, :5])

# # === DISPERSIÓN DE VELOCIDADES ===
# arr_dispersion_velocidades = cargar_dispersion_velocidades(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/dispersion_velocidades.json"
# )
# print("\n--- DISPERSIÓN DE VELOCIDADES ---")
# print("Shape:", arr_dispersion_velocidades.shape)
# print("Ejemplo (primeros 5 frames):", arr_dispersion_velocidades[0, :5])

# # === PORCENTAJE DE GIROS ===
# arr_porcentaje_giros = cargar_porcentaje_giros(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/angulo_cambio_direccion.json"
# )
# print("\n--- PORCENTAJE DE GIROS ---")
# print("Shape:", arr_porcentaje_giros.shape)
# print("Ejemplo (primeros 5 frames):", arr_porcentaje_giros[0, :5])

# # === MEDIA Y DESVIACIÓN ESTÁNDAR DE GIROS ===
# arr_media_std_giros = cargar_media_y_std_giros(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/angulo_cambio_direccion.json"
# )
# print("\n--- MEDIA Y DESVIACIÓN ESTÁNDAR DE GIROS ---")
# print("Shape:", arr_media_std_giros.shape)
# print("Ejemplo Media (primeros 5 frames):", arr_media_std_giros[0, :5])
# print("Ejemplo STD (primeros 5 frames):", arr_media_std_giros[1, :5])

# # === ENTROPÍA DE DIRECCIONES ===
# arr_entropia_direcciones = cargar_entropia_direcciones(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/direcciones.json"
# )
# print("\n--- ENTROPÍA DE DIRECCIONES ---")
# print("Shape:", arr_entropia_direcciones.shape)
# print("Ejemplo (primeros 5 frames):", arr_entropia_direcciones[0, :5])

# # === POLARIZACIÓN ===
# arr_polarizacion = cargar_polarizacion(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/trayectorias_stats/direcciones.json"
# )
# print("\n--- POLARIZACIÓN ---")
# print("Shape:", arr_polarizacion.shape)
# print("Ejemplo (primeros 5 frames):", arr_polarizacion[0, :5])

# # === VARIANZA ESPACIAL ===
# arr_varianza_espacial = cargar_varianza_espacial(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/mask_stats/varianza_espacial.json"
# )
# print("\n--- VARIANZA ESPACIAL ---")
# print("Shape:", arr_varianza_espacial.shape)
# print("Ejemplo (primeros 5 frames):", arr_varianza_espacial[0, :5])

# # === ENTROPIA BINARIA ===
# arr_entropia_binaria = cargar_entropia_binaria(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/mask_stats/entropia_binaria_64.json"
# )
# print("\n--- ENTROPÍA BINARIA ---")
# print("Shape:", arr_entropia_binaria.shape)
# print("Ejemplo (cambio de ventana):", arr_entropia_binaria[0, 62:67]) 
# print("Ejemplo (final ultima ventana):", arr_entropia_binaria[0, -5:])  # Últimos 5 frames

# # === EXPLOREACIÓN ===
# arr_exploracion = cargar_exploracion(
#     "/home/gmanty/code/AnemoNAS/06-12-23/0926-1852/USCL2-092107-092607/bbox_stats/exploracion.json"
# )
# print("\n--- EXPLORACIÓN ---")
# print("Shape:", arr_exploracion.shape)
# print("Ejemplo (cambio de ventana):", arr_exploracion[0, 62:67])
# print("Ejemplo (ultimo frame):", arr_exploracion[0, -5:])  # Últimos 5 frames

# === CENTRO DE MASAS ===
arr_centro_masa = cargar_centro_masa(
    "/home/gms/AnemoNAS/prueba_GUI/output_5s/mask_stats/centro_masa.json"
)
print("\n--- CENTRO DE MASAS ---")
print("Shape:", arr_centro_masa.shape)
print("Ejemplo eje x(primeros 5 frames):", arr_centro_masa[0, :5])
print("Ejemplo eje y(primeros 5 frames):", arr_centro_masa[1, :5])

# === DENSIDAD ===
arr_densidad = cargar_densidad(
    "/home/gms/AnemoNAS/prueba_GUI/output_5s/mask_stats/densidad_(2, 4).json"
)
print("\n--- DENSIDAD ---")
print("Shape:", arr_densidad.shape)
print("Ejemplo (primeros 5 frames):", arr_densidad[0, :5])

# === DISPERSIÓN PX ===
arr_dispersion_px = cargar_dispersion_px(
    "/home/gms/AnemoNAS/prueba_GUI/output_5s/mask_stats/dispersion_64.json"
)
print("\n--- DISPERSIÓN PX ---")
print("Shape:", arr_dispersion_px.shape)
print("Ejemplo (primeros 5 frames):", arr_dispersion_px[0, :5])
print("Ejemplo (cambio de ventana):", arr_dispersion_px[0, 62:67])
print("Ejemplo (final ultima ventana):", arr_dispersion_px[0, -5:]) 

# === VELOCIDAD GRUPO ===
arr_velocidad_grupo = cargar_velocidad_grupo(
    "/home/gms/AnemoNAS/prueba_GUI/output_5s/mask_stats/velocidad_grupo.json"
)
print("\n--- VELOCIDAD GRUPO ---")
print("Shape:", arr_velocidad_grupo.shape)
print("Ejemplo (primeros 5 frames):", arr_velocidad_grupo[0, :5])

# === PERSISTENCIA ===
arr_persistencia = cargar_persistencia(
    "/home/gms/AnemoNAS/prueba_GUI/output_5s/trayectorias_stats/persistencia_espacial.json"
)
print("\n--- PERSISTENCIA ESPACIAL ---")
print("Shape:", arr_persistencia.shape)
print("Ejemplo (primeros 5 frames):", arr_persistencia[0, :5])
print("Ejemplo (cambio de ventana):", arr_persistencia[0, 62:67])