from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
                               QLineEdit, QFileDialog, QHBoxLayout, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from processing_GUI.ventanas.ventana_resultado_visualizacion import VentanaResultadoVisualizacion
import os

class VentanaVisualizacion(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.ventana_inicio = parent
        self.ventana_resultado = None  # referencia persistente
        self.setWindowTitle("Visualización de Etiquetado")
        self.setMinimumSize(500, 250)
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Título
        titulo = QLabel("Visualización de Resultados")
        titulo.setFont(QFont('Segoe UI', 18, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        # Selector de tipo de visualización
        selector_layout = QHBoxLayout()
        label_tipo = QLabel("Tipo de visualización:")
        self.combo_tipo = QComboBox()
        self.combo_tipo.addItems(["Cutie", "Morfología", "YOLOv8"])
        selector_layout.addWidget(label_tipo)
        selector_layout.addWidget(self.combo_tipo)
        layout.addLayout(selector_layout)

        # Selector de carpeta del vídeo procesado
        carpeta_layout = QHBoxLayout()
        self.lineedit_carpeta = QLineEdit()
        self.lineedit_carpeta.setPlaceholderText("Selecciona la carpeta del vídeo procesado")
        boton_examinar = QPushButton("Examinar")
        boton_examinar.clicked.connect(self.seleccionar_carpeta)
        carpeta_layout.addWidget(self.lineedit_carpeta)
        carpeta_layout.addWidget(boton_examinar)
        layout.addLayout(carpeta_layout)

        # Botones
        botones_layout = QHBoxLayout()
        self.boton_atras = QPushButton("Atrás")
        self.boton_visualizar = QPushButton("Visualizar")
        self.boton_atras.clicked.connect(self.volver_a_inicio)
        self.boton_visualizar.clicked.connect(self.visualizar_resultados)
        botones_layout.addWidget(self.boton_atras)
        botones_layout.addStretch()
        botones_layout.addWidget(self.boton_visualizar)
        layout.addLayout(botones_layout)

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
        tipo = self.combo_tipo.currentText()

        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.critical(self, "Error", "Debes seleccionar una carpeta válida")
            return

        nombre_video = os.path.basename(carpeta)
        ruta_video = os.path.join(carpeta, f"{nombre_video}.mp4")
        if not os.path.exists(ruta_video):
            QMessageBox.critical(self, "Error", f"No se encontró el vídeo: {ruta_video}")
            return

        # Si ya había una ventana abierta, la cerramos y la descartamos
        if self.ventana_resultado:
            self.ventana_resultado.close()
            self.ventana_resultado = None

        # Crear nueva instancia con los valores actuales
        self.ventana_resultado = VentanaResultadoVisualizacion(
            ruta_video=ruta_video,
            carpeta_base=carpeta,
            tipo_visualizacion=tipo,
            parent=self
        )
        self.ventana_resultado.show()
        self.ventana_resultado.raise_()
        self.ventana_resultado.activateWindow()

    def volver_a_inicio(self):
        self.close()
        if self.ventana_inicio:
            self.ventana_inicio.show()
