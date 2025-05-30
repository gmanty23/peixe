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


import os
import json

class VentanaVisualizacion(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.ventana_inicio = parent
        self.ventana_resultado = None
        self.setWindowTitle("Visualización de Resultados")
        self.setMinimumSize(600, 600)
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
        label_dist = QLabel("Distribución espacial:")
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




    def volver_a_inicio(self):
        self.close()
        if self.ventana_inicio:
            self.ventana_inicio.show()

    
