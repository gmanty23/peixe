import cv2
import os
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from processing_GUI.procesamiento.visualizacion import (
    cargar_mascara, cargar_bboxes_yolo, dibujar_bboxes,
    superponer_mascara, cargar_recorte, aplicar_recorte, cargar_output_dims
)

class VentanaResultadoVisualizacion(QMainWindow):
    def __init__(self, ruta_video, carpeta_base, tipo_visualizacion, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resultado de Visualización")
        self.setMinimumSize(960, 600)
        self.ruta_video = ruta_video
        self.carpeta_base = carpeta_base
        self.tipo_visualizacion = tipo_visualizacion.lower()
        self.parent = parent

        self.cap = cv2.VideoCapture(self.ruta_video)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.frame_actual_leido = -1
        self.frame_cache = None

        self.recorte = cargar_recorte(self.carpeta_base)

        self.timer = QTimer()
        self.timer.timeout.connect(self.mostrar_siguiente_frame)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        self.setup_ui()
        self.cargar_y_mostrar_frame(self.frame_idx)

    def setup_ui(self):
        self.label_frame = QLabel("Cargando vídeo...")
        self.label_frame.setAlignment(Qt.AlignCenter)
        self.label_frame.setMinimumSize(800, 450)
        self.layout.addWidget(self.label_frame)

        controls = QHBoxLayout()

        self.boton_anterior = QPushButton("⏮️")
        self.boton_siguiente = QPushButton("⏭️")
        self.boton_salto_atras = QPushButton("-10")
        self.boton_salto_adelante = QPushButton("+10")
        self.boton_play = QPushButton("▶️")
        self.boton_atras = QPushButton("Atrás")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.sliderMoved.connect(self.cargar_y_mostrar_frame)

        self.texto_estado = QLabel("Frame 0 / 0")

        self.boton_anterior.clicked.connect(self.frame_anterior)
        self.boton_siguiente.clicked.connect(self.frame_siguiente)
        self.boton_salto_atras.clicked.connect(lambda: self.saltar_frames(-10))
        self.boton_salto_adelante.clicked.connect(lambda: self.saltar_frames(10))
        self.boton_play.clicked.connect(self.toggle_reproduccion)
        self.boton_atras.clicked.connect(self.volver)

        controls.addWidget(self.boton_salto_atras)
        controls.addWidget(self.boton_anterior)
        controls.addWidget(self.boton_play)
        controls.addWidget(self.boton_siguiente)
        controls.addWidget(self.boton_salto_adelante)
        controls.addWidget(self.slider)
        controls.addWidget(self.texto_estado)
        controls.addStretch()
        controls.addWidget(self.boton_atras)

        self.layout.addLayout(controls)

    def cargar_y_mostrar_frame(self, idx):
        self.frame_idx = int(idx)

        if self.frame_idx == self.frame_actual_leido + 1:
            ret, frame = self.cap.read()
            if not ret:
                return
            self.frame_cache = frame.copy()
            self.frame_actual_leido = self.frame_idx
        elif self.frame_idx != self.frame_actual_leido:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                return
            self.frame_cache = frame.copy()
            self.frame_actual_leido = self.frame_idx

        frame = self.frame_cache.copy()

        if self.tipo_visualizacion == "máscaras":
            mask = cargar_mascara(self.carpeta_base, self.frame_idx)
            if mask is not None:
                if self.recorte:
                    frame = aplicar_recorte(frame, self.recorte)
                    output_dims = cargar_output_dims(os.path.join(self.carpeta_base, "masks_rle"))
                    frame = cv2.resize(frame, output_dims, interpolation=cv2.INTER_LINEAR)
                frame = superponer_mascara(frame, mask)

        elif self.tipo_visualizacion == "bboxes":
            if self.recorte:
                frame = aplicar_recorte(frame, self.recorte)
                output_dims = cargar_output_dims(os.path.join(self.carpeta_base, "bbox"))
                frame = cv2.resize(frame, output_dims, interpolation=cv2.INTER_LINEAR)
            bboxes = cargar_bboxes_yolo(self.carpeta_base, self.frame_idx)
            frame = dibujar_bboxes(frame, bboxes)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.label_frame.setPixmap(pixmap.scaled(
            self.label_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.slider.setValue(self.frame_idx)
        self.texto_estado.setText(f"Frame {self.frame_idx} / {self.total_frames - 1}")

    def frame_anterior(self):
        if self.frame_idx > 0:
            self.cargar_y_mostrar_frame(self.frame_idx - 1)

    def frame_siguiente(self):
        if self.frame_idx < self.total_frames - 1:
            self.cargar_y_mostrar_frame(self.frame_idx + 1)

    def saltar_frames(self, cantidad):
        nuevo_idx = max(0, min(self.total_frames - 1, self.frame_idx + cantidad))
        self.cargar_y_mostrar_frame(nuevo_idx)

    def toggle_reproduccion(self):
        if self.timer.isActive():
            self.timer.stop()
            self.boton_play.setText("▶️")
        else:
            self.timer.start(66)
            self.boton_play.setText("⏸️")

    def mostrar_siguiente_frame(self):
        if self.frame_idx < self.total_frames - 1:
            self.cargar_y_mostrar_frame(self.frame_idx + 1)
        else:
            self.toggle_reproduccion()

    def volver(self):
        self.timer.stop()
        self.close()
        if self.parent:
            self.parent.show()
