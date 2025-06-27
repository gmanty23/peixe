import os
import json
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSlider, QComboBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from io import BytesIO

class VentanaResultadoPersistencia(QWidget):
    def __init__(self, ruta_video, carpeta_base, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Evolución de Persistencia Espacial")
        self.setMinimumSize(800, 600)

        self.ruta_video = ruta_video
        self.carpeta_base = carpeta_base
        self.tamano_ventana = "64"

        self.persistencias = {}
        self.claves_ventanas = []
        self.current_idx = 0

        self.setup_ui()
        self.cargar_datos()
        self.mostrar_ventana()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Selector de tamaño de ventana
        self.combo_ventana = QComboBox()
        self.combo_ventana.addItems(["64", "128", "256", "512"])
        self.combo_ventana.setCurrentText(self.tamano_ventana)
        self.boton_actualizar = QPushButton("Actualizar")
        self.boton_actualizar.clicked.connect(self.actualizar_tamano)

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("Tamaño ventana:"))
        top_controls.addWidget(self.combo_ventana)
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
        self.boton_adelante = QPushButton("▶️")
        self.boton_fin = QPushButton("⏭️")

        self.boton_ini.clicked.connect(lambda: self.ir_a_ventana(0))
        self.boton_atras.clicked.connect(lambda: self.ir_a_ventana(self.current_idx - 1))
        self.boton_adelante.clicked.connect(lambda: self.ir_a_ventana(self.current_idx + 1))
        self.boton_fin.clicked.connect(lambda: self.ir_a_ventana(len(self.claves_ventanas) - 1))

        nav_layout.addWidget(self.boton_ini)
        nav_layout.addWidget(self.boton_atras)
        nav_layout.addWidget(self.boton_adelante)
        nav_layout.addWidget(self.boton_fin)
        layout.addLayout(nav_layout)

        # Slider + etiqueta
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.ir_a_ventana)
        layout.addWidget(self.slider)

        self.label_info = QLabel("Ventana 0")
        self.label_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_info)

    def actualizar_tamano(self):
        self.tamano_ventana = self.combo_ventana.currentText()
        self.cargar_datos()
        self.mostrar_ventana()

    def cargar_datos(self):
        archivo = os.path.join(self.carpeta_base, "mask_stats", f"persistencia_{self.tamano_ventana}.json")
        if not os.path.exists(archivo):
            self.persistencias = {}
            self.claves_ventanas = []
            return
        with open(archivo, "r") as f:
            self.persistencias = json.load(f)
        self.claves_ventanas = sorted(self.persistencias.keys())
        self.slider.setRange(0, len(self.claves_ventanas) - 1)
        self.current_idx = 0

    def ir_a_ventana(self, idx):
        if isinstance(idx, int):
            self.current_idx = max(0, min(idx, len(self.claves_ventanas) - 1))
        else:
            self.current_idx = self.slider.value()
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_idx)
        self.slider.blockSignals(False)
        self.mostrar_ventana()

    def mostrar_ventana(self):
        if not self.claves_ventanas:
            self.label_imagen.setText("No hay datos para mostrar.")
            return

        clave = self.claves_ventanas[self.current_idx]
        datos = self.persistencias[clave]["por_celda"]

        # reconstruir matriz
        celdas = list(datos.keys())
        max_i = max(int(k.split('_')[0]) for k in celdas)
        max_j = max(int(k.split('_')[1]) for k in celdas)
        matriz = np.zeros((max_i+1, max_j+1), dtype=np.float32)

        for clave_celda, info in datos.items():
            i, j = map(int, clave_celda.split('_'))
            matriz[i, j] = info.get("media", 0.0)

        fig, ax = plt.subplots()
        im = ax.imshow(matriz, cmap='Blues', interpolation='nearest', vmin=0.0)
        ax.set_title(f"Persistencia - {clave} - Ventana {self.tamano_ventana}")
        fig.colorbar(im)

        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                valor = matriz[i, j]
                if valor > 0:
                    ax.text(j, i, f"{valor:.1f}", ha='center', va='center', color='black', fontsize=8)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        qimg = QImage.fromData(buf.read(), format='PNG')
        pixmap = QPixmap.fromImage(qimg)
        self.label_imagen.setPixmap(pixmap.scaled(
            self.label_imagen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        self.label_info.setText(f"{clave} ({self.current_idx+1}/{len(self.claves_ventanas)})")
