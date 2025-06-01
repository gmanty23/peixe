import os
import json
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSlider, QComboBox
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO

class VentanaResultadoDensidad(QWidget):
    def __init__(self, ruta_video, carpeta_base, grid_inicial="10", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mapa de Densidad desde Máscaras")
        self.setMinimumSize(800, 600)

        self.ruta_video = ruta_video
        self.carpeta_base = carpeta_base
        self.grid_actual = grid_inicial

        self.total_frames = self._contar_frames_video()
        self.current_frame_idx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.siguiente_frame)

        self.densidades = {}
        self.setup_ui()
        self.cargar_densidades()
        self.mostrar_frame()

    def _contar_frames_video(self):
        import cv2
        cap = cv2.VideoCapture(self.ruta_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Grid selector
        self.combo_grid = QComboBox()
        self.combo_grid.addItems(["5", "10", "15", "20"])
        self.combo_grid.setCurrentText(self.grid_actual)
        self.boton_actualizar = QPushButton("Actualizar grid")
        self.boton_actualizar.clicked.connect(self.actualizar_grid)

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("Grid:"))
        top_controls.addWidget(self.combo_grid)
        top_controls.addWidget(self.boton_actualizar)
        top_controls.addStretch()
        layout.addLayout(top_controls)

        # Imagen
        self.label_imagen = QLabel("Cargando...")
        self.label_imagen.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_imagen, stretch=1)

        # Navegación
        nav_layout = QHBoxLayout()
        self.boton_ini = QPushButton("⏮️")
        self.boton_atras = QPushButton("◀️")
        self.boton_play = QPushButton("▶️")
        self.boton_stop = QPushButton("⏸️")
        self.boton_adelante = QPushButton("▶️▶️")
        self.boton_saltar = QPushButton("+10")

        self.boton_ini.clicked.connect(lambda: self.ir_a_frame(0))
        self.boton_atras.clicked.connect(lambda: self.ir_a_frame(self.current_frame_idx - 1))
        self.boton_play.clicked.connect(self.reproducir)
        self.boton_stop.clicked.connect(self.pausar)
        self.boton_adelante.clicked.connect(lambda: self.ir_a_frame(self.current_frame_idx + 1))
        self.boton_saltar.clicked.connect(lambda: self.ir_a_frame(self.current_frame_idx + 10))

        nav_layout.addWidget(self.boton_ini)
        nav_layout.addWidget(self.boton_atras)
        nav_layout.addWidget(self.boton_play)
        nav_layout.addWidget(self.boton_stop)
        nav_layout.addWidget(self.boton_adelante)
        nav_layout.addWidget(self.boton_saltar)
        layout.addLayout(nav_layout)

        # Slider + contador
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.valueChanged.connect(self.ir_a_frame)
        layout.addWidget(self.slider)

        self.label_info = QLabel("Frame 0")
        self.label_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_info)

    def cargar_densidades(self):
        ruta_json = os.path.join(self.carpeta_base, "mask_stats", f"densidad_{self.grid_actual}.json")
        if not os.path.exists(ruta_json):
            self.densidades = {}
            return

        with open(ruta_json, "r") as f:
            data = json.load(f)
        self.densidades = data.get("densidad", {})

    def actualizar_grid(self):
        self.grid_actual = self.combo_grid.currentText()
        self.cargar_densidades()
        self.mostrar_frame()

    def ir_a_frame(self, idx):
        if isinstance(idx, int):
            self.current_frame_idx = max(0, min(idx, self.total_frames - 1))
        else:
            self.current_frame_idx = self.slider.value()
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)
        self.mostrar_frame()

    def siguiente_frame(self):
        if self.current_frame_idx + 1 < self.total_frames:
            self.ir_a_frame(self.current_frame_idx + 1)
        else:
            self.pausar()

    def reproducir(self):
        self.timer.start(150)

    def pausar(self):
        self.timer.stop()

    def mostrar_frame(self):
        frame_key = f"frame_{self.current_frame_idx:05d}"
        if frame_key not in self.densidades:
            self.label_imagen.setText("Sin datos para este frame")
            return

        densidad_array = np.array(self.densidades[frame_key])
        densidad_array = densidad_array.astype(np.float32)

        fig, ax = plt.subplots()
        im = ax.imshow(densidad_array, cmap='hot', interpolation='nearest', vmin=0.0, vmax=1.0)
        ax.set_title(f"Densidad Frame {self.current_frame_idx+1}")
        fig.colorbar(im)

        for i in range(densidad_array.shape[0]):
            for j in range(densidad_array.shape[1]):
                valor = densidad_array[i, j]
                ax.text(j, i, f"{valor:.2f}", ha='center', va='center', color='blue', fontsize=8)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        qimg = QImage.fromData(buf.read(), format='PNG')
        pixmap = QPixmap.fromImage(qimg)
        self.label_imagen.setPixmap(pixmap.scaled(
            self.label_imagen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        self.label_info.setText(f"Frame {self.current_frame_idx + 1} / {self.total_frames}")
