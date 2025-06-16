from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QProgressBar, QMessageBox, QSizePolicy
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, QTimer
import os
from processing_GUI.procesamiento.analisis import generar_inputs_moment, EstadoProceso

class VentanaAnalisis(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.ventana_inicio = parent
        self.setWindowTitle("Herramienta de Análisis")
        self.setMinimumSize(600, 400)
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        titulo = QLabel("Análisis Series Temporales")
        titulo.setFont(QFont('Segoe UI', 18, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        grupo_inputs = QGroupBox("Generación Inputs MOMENT")
        layout_grupo = QVBoxLayout()

        # Selector de carpeta
        selector_layout = QHBoxLayout()
        self.lineedit_carpeta = QLineEdit()
        self.lineedit_carpeta.setPlaceholderText("Selecciona la carpeta de trabajo")
        boton_examinar = QPushButton("Examinar")
        boton_examinar.clicked.connect(self.seleccionar_carpeta)
        selector_layout.addWidget(self.lineedit_carpeta)
        selector_layout.addWidget(boton_examinar)
        layout_grupo.addLayout(selector_layout)

        # Botón generar
        self.boton_generar = QPushButton("Generar Inputs Compatibles con MOMENT")
        self.boton_generar.clicked.connect(self.generar_inputs)
        self.boton_generar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout_grupo.addWidget(self.boton_generar)

        
        # Estado y progreso
        self.etiqueta_estado_grupo = QLabel("Esperando instrucciones...")
        layout_grupo.addWidget(self.etiqueta_estado_grupo)
        
        self.barra_progreso_grupo = QProgressBar()
        layout_grupo.addWidget(self.barra_progreso_grupo)
        
        self.etiqueta_estado_grupo.setVisible(False)
        self.barra_progreso_grupo.setVisible(False)
        
        
        grupo_inputs.setLayout(layout_grupo) 
        layout.addWidget(grupo_inputs)

        # Botón volver
        self.boton_atras = QPushButton("Atrás")
        self.boton_atras.clicked.connect(self.volver_a_inicio)
        layout.addWidget(self.boton_atras, alignment=Qt.AlignLeft)

    def setup_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
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
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de trabajo")
        if carpeta:
            self.lineedit_carpeta.setText(carpeta)

    def generar_inputs(self):
        self.etiqueta_estado_grupo.setVisible(True)
        self.barra_progreso_grupo.setVisible(True)
        self.barra_progreso_grupo.setValue(0)
        self.etiqueta_estado_grupo.setText("Iniciando generación...")
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", "Debes seleccionar una carpeta válida.")
            return
        
        estado = EstadoProceso()
        estado.on_etapa = self.etiqueta_estado_grupo.setText
        estado.on_total_videos = self.barra_progreso_grupo.setMaximum
        estado.on_video_progreso = self.barra_progreso_grupo.setValue
        
        # Llamada a tu función de procesado
        generar_inputs_moment(carpeta)
        
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Generación", "Inputs de MOMENT generados correctamente.")

    
    def volver_a_inicio(self):
        self.close()
        if self.ventana_inicio:
            self.ventana_inicio.show()
