from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
                               QLineEdit, QFileDialog, QHBoxLayout, QMessageBox,
                               QGroupBox, QListWidget, QListWidgetItem, QFrame, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from processing_GUI.ventanas.ventana_resultado_visualizacion import VentanaResultadoVisualizacion
from processing_GUI.ventanas.ventana_resultado_distribucion import VentanaResultadoDistribucion
from processing_GUI.ventanas.ventana_resultado_bbox_stats import VentanaResultadoBBoxStats
from processing_GUI.ventanas.ventana_resultado_distancias import VentanaResultadoDistancias
from processing_GUI.ventanas.ventana_resultado_centroide_global import VentanaResultadoCentroideGrupal
from processing_GUI.ventanas.ventana_resultado_densidad import VentanaResultadoDensidad
from processing_GUI.ventanas.ventana_resultado_centro_masa import VentanaResultadoCentroMasa
from processing_GUI.ventanas.ventana_resultado_mask_stats import VentanaResultadoMaskStats
from processing_GUI.ventanas.ventana_resultado_persistencia import VentanaResultadoPersistencia
from processing_GUI.ventanas.ventana_resultado_trayectorias import VentanaResultadoTrayectorias
from processing_GUI.ventanas.ventana_resultado_robustez import VentanaResultadoRobustez
from processing_GUI.ventanas.ventana_resultado_tray_stats import VentanaResultadoTrayStats


import os
import json

class VentanaVisualizacion(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.ventana_inicio = parent
        self.ventana_resultado = None
        self.setWindowTitle("Herramienta de Visualización de Resultados")
        self.setMinimumSize(600, 800)
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Título general
        titulo = QLabel("Visualización de Resultados")
        titulo.setFont(QFont('Segoe UI', 18, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        # Sección 0: Selección de carpeta del vídeo procesado
        carpeta_layout = QHBoxLayout()
        self.lineedit_carpeta = QLineEdit()
        self.lineedit_carpeta.setPlaceholderText("Selecciona la carpeta del vídeo procesado")
        boton_examinar = QPushButton("Examinar")
        boton_examinar.clicked.connect(self.seleccionar_carpeta)
        carpeta_layout.addWidget(self.lineedit_carpeta)
        carpeta_layout.addWidget(boton_examinar)
        layout.addLayout(carpeta_layout)

        # Sección 1: Resultados de etiquetado
        group_etiquetado = QGroupBox("Visualización de Etiquetado")
        layout_etiquetado = QVBoxLayout()

        selector_layout = QHBoxLayout()
        label_tipo = QLabel("Tipo de visualización:")
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(["Máscaras", "BBoxes"])
        selector_layout.addWidget(label_tipo)
        selector_layout.addWidget(self.combo_tipo)
        layout_etiquetado.addLayout(selector_layout)

        boton_visualizar = QPushButton("Visualizar")
        boton_visualizar.clicked.connect(self.visualizar_resultados)
        layout_etiquetado.addWidget(boton_visualizar)

        group_etiquetado.setLayout(layout_etiquetado)
        layout.addWidget(group_etiquetado)

        # Sección 2: Resultados de postprocesado
        group_post = QGroupBox("Visualización de Estadísticas de Comportamiento")
        layout_post = QVBoxLayout()

        # BLOQUE 1: Distribución espacial
        frame_dist = QFrame()
        layout_dist = QHBoxLayout()
        label_dist = QLabel("Distribución espacial Bbox:")
        combo_grid = QComboBox()
        combo_grid.addItems(["5", "10", "15", "20"])
        boton_mostrar_dist = QPushButton("Mostrar")
        layout_dist.addWidget(label_dist)
        layout_dist.addWidget(combo_grid)
        layout_dist.addWidget(boton_mostrar_dist)
        frame_dist.setLayout(layout_dist)
        layout_post.addWidget(frame_dist)

        boton_mostrar_dist.clicked.connect(lambda: self.mostrar_distribucion(combo_grid))

        # BLOQUE 2: Estadísticas BBox
        frame_bbox = QFrame()
        layout_bbox = QHBoxLayout()
        label_bbox = QLabel("Estadísticas BBox:")
        boton_mostrar_bbox = QPushButton("Mostrar")
        layout_bbox.addWidget(label_bbox)
        layout_bbox.addStretch()
        layout_bbox.addWidget(boton_mostrar_bbox)
        frame_bbox.setLayout(layout_bbox)
        layout_post.addWidget(frame_bbox)

        boton_mostrar_bbox.clicked.connect(self.mostrar_estadisticas_bbox)

        # BLOQUE 3: Visualización de centroides agrupados
        frame_agrupados = QFrame()
        layout_agrupados = QHBoxLayout()
        label_agrupados = QLabel("Centroides agrupados (verde) vs aislados (rojo):")
        boton_mostrar_agrupados = QPushButton("Mostrar")
        layout_agrupados.addWidget(label_agrupados)
        layout_agrupados.addStretch()
        layout_agrupados.addWidget(boton_mostrar_agrupados)
        frame_agrupados.setLayout(layout_agrupados)
        layout_post.addWidget(frame_agrupados)

        boton_mostrar_agrupados.clicked.connect(self.mostrar_centroides_agrupados)

        # BLOQUE 4: Visualización de centroide grupal
        frame_centroide = QFrame()
        layout_centroide = QHBoxLayout()
        label_centroide = QLabel("Centroide grupal (círculo blanco por frame):")
        boton_mostrar_centroide = QPushButton("Mostrar")
        layout_centroide.addWidget(label_centroide)
        layout_centroide.addStretch()
        layout_centroide.addWidget(boton_mostrar_centroide)
        frame_centroide.setLayout(layout_centroide)
        layout_post.addWidget(frame_centroide)

        boton_mostrar_centroide.clicked.connect(self.mostrar_centroide_grupal)

        # BLOQUE 5: Densidad desde máscaras
        frame_densidad = QFrame()
        layout_densidad = QHBoxLayout()
        label_densidad = QLabel("Densidad por máscaras:")
        combo_grid_densidad = QComboBox()
        combo_grid_densidad.addItems(["5", "10", "15", "20"])
        boton_mostrar_densidad = QPushButton("Mostrar")
        layout_densidad.addWidget(label_densidad)
        layout_densidad.addWidget(combo_grid_densidad)
        layout_densidad.addWidget(boton_mostrar_densidad)
        frame_densidad.setLayout(layout_densidad)
        layout_post.addWidget(frame_densidad)

        boton_mostrar_densidad.clicked.connect(lambda: self.mostrar_densidad(combo_grid_densidad))

        # BLOQUE 6: Estadísticas desde máscaras
        frame_mask_stats = QFrame()
        layout_mask_stats = QHBoxLayout()
        label_mask = QLabel("Estadísticas Masks:")
        boton_mostrar_mask = QPushButton("Mostrar")
        layout_mask_stats.addWidget(label_mask)
        layout_mask_stats.addStretch()
        layout_mask_stats.addWidget(boton_mostrar_mask)
        frame_mask_stats.setLayout(layout_mask_stats)
        layout_post.addWidget(frame_mask_stats)

        boton_mostrar_mask.clicked.connect(self.mostrar_estadisticas_mask)

        # BLOQUE 7: Centro de masa desde máscaras
        frame_centro_masa = QFrame()
        layout_centro_masa = QHBoxLayout()
        label_centro_masa = QLabel("Centro de masa (máscaras):")
        boton_mostrar_centro_masa = QPushButton("Mostrar")
        layout_centro_masa.addWidget(label_centro_masa)
        layout_centro_masa.addStretch()
        layout_centro_masa.addWidget(boton_mostrar_centro_masa)
        frame_centro_masa.setLayout(layout_centro_masa)
        layout_post.addWidget(frame_centro_masa)

        boton_mostrar_centro_masa.clicked.connect(self.mostrar_centro_masa)

        # BLOQUE 8: Persistencia espacial por ventanas
        frame_persistencia = QFrame()
        layout_persistencia = QHBoxLayout()
        label_persistencia = QLabel("Persistencia por ventanas:")
        boton_mostrar_persistencia = QPushButton("Mostrar")
        layout_persistencia.addWidget(label_persistencia)
        layout_persistencia.addStretch()
        layout_persistencia.addWidget(boton_mostrar_persistencia)
        frame_persistencia.setLayout(layout_persistencia)
        layout_post.addWidget(frame_persistencia)

        boton_mostrar_persistencia.clicked.connect(self.mostrar_persistencia)

        # BLOQUE 9: Visualización de trayectorias por ID
        frame_trayectorias = QFrame()
        layout_trayectorias = QHBoxLayout()
        label_trayectorias = QLabel("Visualización de trayectorias por ID")
        boton_mostrar_trayectorias = QPushButton("Mostrar")
        layout_trayectorias.addWidget(label_trayectorias)
        layout_trayectorias.addStretch()
        layout_trayectorias.addWidget(boton_mostrar_trayectorias)
        frame_trayectorias.setLayout(layout_trayectorias)
        layout_post.addWidget(frame_trayectorias)

        boton_mostrar_trayectorias.clicked.connect(self.mostrar_trayectorias)

        # BLOQUE 10: Estadísticos de Robustez del Tracking
        frame_robustez = QFrame()
        layout_robustez = QHBoxLayout()
        label_robustez = QLabel("Estadísticos de Robustez del Tracking:")
        boton_mostrar_robustez = QPushButton("Mostrar")
        layout_robustez.addWidget(label_robustez)
        layout_robustez.addStretch()
        layout_robustez.addWidget(boton_mostrar_robustez)
        frame_robustez.setLayout(layout_robustez)
        layout_post.addWidget(frame_robustez)

        boton_mostrar_robustez.clicked.connect(self.mostrar_robustez_tracking)

        # BLOQUE: Velocidades de Peces
        frame_velocidades = QFrame()
        layout_velocidades = QHBoxLayout()
        label_velocidades = QLabel("Estadíscas Trayectorias:")
        boton_mostrar_velocidades = QPushButton("Mostrar")
        layout_velocidades.addWidget(label_velocidades)
        layout_velocidades.addStretch()
        layout_velocidades.addWidget(boton_mostrar_velocidades)
        frame_velocidades.setLayout(layout_velocidades)
        layout_post.addWidget(frame_velocidades)

        boton_mostrar_velocidades.clicked.connect(self.mostrar_velocidades)


        group_post.setLayout(layout_post)
        layout.addWidget(group_post)

        # Botón abajo a la izquierda
        layout_inferior = QHBoxLayout()
        self.boton_atras = QPushButton("Atrás")
        self.boton_atras.clicked.connect(self.volver_a_inicio)
        layout_inferior.addWidget(self.boton_atras)
        layout_inferior.addStretch()
        layout.addLayout(layout_inferior)

        self.setLayout(layout)

    def setup_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a5276;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)

    def seleccionar_carpeta(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta del vídeo procesado")
        if carpeta:
            self.lineedit_carpeta.setText(carpeta)

    def visualizar_resultados(self):
        carpeta = self.lineedit_carpeta.text()
        tipo = self.combo_tipo.currentText().lower()

        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.critical(self, "Error", "Debes seleccionar una carpeta válida")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")
        if not os.path.exists(ruta_video):
            QMessageBox.critical(self, "Error", f"No se encontró el vídeo: {ruta_video}")
            return

        if self.ventana_resultado:
            self.ventana_resultado.close()
            self.ventana_resultado = None

        self.ventana_resultado = VentanaResultadoVisualizacion(
            ruta_video=ruta_video,
            carpeta_base=carpeta,
            tipo_visualizacion=tipo,
            parent=self
        )
        self.ventana_resultado.show()
        self.ventana_resultado.raise_()
        self.ventana_resultado.activateWindow()

    def añadir_bloque_visualizacion(self):
        frame = QFrame()
        layout = QHBoxLayout()

        label = QLabel("Distribución espacial:")
        combo_grid = QComboBox()
        combo_grid.addItems(["5", "10", "15", "20"])
        boton_mostrar = QPushButton("Mostrar")
        boton_borrar = QPushButton("✖")

        layout.addWidget(label)
        layout.addWidget(combo_grid)
        layout.addWidget(boton_mostrar)
        layout.addWidget(boton_borrar)
        frame.setLayout(layout)

        item = QListWidgetItem()
        item.setSizeHint(frame.sizeHint())
        self.lista_estadisticas.addItem(item)
        self.lista_estadisticas.setItemWidget(item, frame)

        boton_mostrar.clicked.connect(lambda _, c=combo_grid: self.mostrar_distribucion(c))
        boton_borrar.clicked.connect(lambda: self.eliminar_item(item))

    def eliminar_item(self, item):
        row = self.lista_estadisticas.row(item)
        self.lista_estadisticas.takeItem(row)

    def mostrar_distribucion(self, combo_grid_widget):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        grid = combo_grid_widget.currentText()
        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")

        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return

        self.ventana_distribucion = VentanaResultadoDistribucion(
            ruta_video=ruta_video,
            carpeta_base=carpeta,
            grid_inicial=grid
        )
        self.ventana_distribucion.show()
        self.ventana_distribucion.raise_()
        self.ventana_distribucion.activateWindow()

    def mostrar_estadisticas_bbox(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")
        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return

        self.ventana_bbox_stats = VentanaResultadoBBoxStats(
            ruta_video=ruta_video,
            carpeta_base=carpeta
        )
        self.ventana_bbox_stats.show()
        self.ventana_bbox_stats.raise_()
        self.ventana_bbox_stats.activateWindow()

    def mostrar_centroides_agrupados(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")

        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return

        self.ventana_distancias = VentanaResultadoDistancias(
            ruta_video=ruta_video,
            carpeta_base=carpeta
        )
        self.ventana_distancias.show()
        self.ventana_distancias.raise_()
        self.ventana_distancias.activateWindow()

    def mostrar_centroide_grupal(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")

        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return

        self.ventana_centroide = VentanaResultadoCentroideGrupal(
            ruta_video=ruta_video,
            carpeta_base=carpeta
        )
        self.ventana_centroide.show()
        self.ventana_centroide.raise_()
        self.ventana_centroide.activateWindow()

    def mostrar_densidad(self, combo_grid_widget):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        grid = combo_grid_widget.currentText()
        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")

        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return

        self.ventana_densidad = VentanaResultadoDensidad(
            ruta_video=ruta_video,
            carpeta_base=carpeta,
            grid_inicial=grid
        )
        self.ventana_densidad.show()
        self.ventana_densidad.raise_()
        self.ventana_densidad.activateWindow()

    def mostrar_centro_masa(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")

        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return

        self.ventana_centro_masa = VentanaResultadoCentroMasa(
            ruta_video=ruta_video,
            carpeta_base=carpeta
        )
        self.ventana_centro_masa.show()
        self.ventana_centro_masa.raise_()
        self.ventana_centro_masa.activateWindow()

    def mostrar_estadisticas_mask(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")
        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return

        self.ventana_mask_stats = VentanaResultadoMaskStats(
            ruta_video=ruta_video,
            carpeta_base=carpeta
        )
        self.ventana_mask_stats.show()
        self.ventana_mask_stats.raise_()
        self.ventana_mask_stats.activateWindow()

    def mostrar_persistencia(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")
        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return

        self.ventana_persistencia = VentanaResultadoPersistencia(
            ruta_video=ruta_video,
            carpeta_base=carpeta
        )
        self.ventana_persistencia.show()
        self.ventana_persistencia.raise_()
        self.ventana_persistencia.activateWindow()

    def mostrar_trayectorias(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")
        if not os.path.exists(ruta_video):
            QMessageBox.warning(self, "Vídeo no encontrado", f"No se encontró: {ruta_video}")
            return
        ruta_trayectorias = os.path.join(carpeta, "trayectorias_stats", "trayectorias.json")
        if os.path.exists(ruta_video) and os.path.exists(ruta_trayectorias):
            
            self.ventana_trayectorias = VentanaResultadoTrayectorias(
                ruta_video=ruta_video,
                carpeta_base=carpeta
            )
            self.ventana_trayectorias.show()
            self.ventana_trayectorias.raise_()
            self.ventana_trayectorias.activateWindow()
            
        else:
            QMessageBox.warning(self, "Archivos faltantes", "No se encontró el video o el archivo de trayectorias.")

    def mostrar_robustez_tracking(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        self.ventana_robustez = VentanaResultadoRobustez(
            carpeta_base=carpeta
        )
        self.ventana_robustez.show()
        self.ventana_robustez.raise_()
        self.ventana_robustez.activateWindow()

    def mostrar_velocidades(self):
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.warning(self, "Carpeta no válida", "Debes seleccionar una carpeta primero.")
            return

        self.ventana_tray_stats = VentanaResultadoTrayStats(
            carpeta_base = carpeta
    )
        self.ventana_tray_stats.show()
        self.ventana_tray_stats.raise_()
        self.ventana_tray_stats.activateWindow()


    def volver_a_inicio(self):
        self.close()
        if self.ventana_inicio:
            self.ventana_inicio.show()

    
