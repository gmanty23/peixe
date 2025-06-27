import os
import json
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy
import math

class VentanaResultadoMaskStats(QWidget):
    def __init__(self, ruta_video, carpeta_base, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Estadísticos desde Máscaras")
        self.setMinimumSize(900, 600)

        self.ruta_video = ruta_video
        self.carpeta_base = carpeta_base

        self.setup_ui()
        self.cargar_datos()
        self.mostrar_grafico()

    def setup_ui(self):
        self.layout = QVBoxLayout(self)

        self.label_titulo = QLabel("Estadísticos calculados desde máscaras")
        self.label_titulo.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.label_titulo)

        self.canvas = FigureCanvas(Figure(figsize=(10, 5)))
        self.layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

    def cargar_datos(self):
        self.descriptores = []

        posibles = [
            ("Varianza espacial", "varianza_espacial.json", "varianza", "std"),
            ("Velocidad del grupo", "velocidad_grupo.json", "velocidad", None),
            ("Histograma de direcciones", "velocidad_grupo.json", None, None),
            ("Autocorrelación: valor central", "autocorrelacion.json", "valor_central", None),
            ("Autocorrelación: media por pixel activo", "autocorrelacion.json", "media_por_pixel_activo", None),
            ("Persistencia global (64)", "persistencia_64.json", "media_global", "std_global"),
            ("Dispersión temporal (64)", "dispersion_64.json", "porcentaje", None),
            ("Entropía binaria (64)", "entropia_binaria_64.json", "entropia", None),

        ]

        for nombre, archivo, clave_y, clave_std in posibles:
            if archivo is None:
                continue  # aún no implementado
            ruta = os.path.join(self.carpeta_base, "mask_stats", archivo)
            if os.path.exists(ruta):
                with open(ruta, "r") as f:
                    datos = json.load(f)
                self.descriptores.append((nombre, archivo, datos, clave_y, clave_std))

    def mostrar_grafico(self):
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.figure.clf()
        fig = self.canvas.figure
        fig.set_constrained_layout(True)

        total = len(self.descriptores)
        if total == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("No se encontraron archivos de estadísticos.")
            ax.axis("off")
            self.canvas.draw()
            return

        columnas = min(3, total)
        filas = math.ceil(total / columnas)

        for idx, (titulo, archivo, dic, clave_y, clave_std) in enumerate(self.descriptores):
            if archivo == "velocidad_grupo.json" and clave_y is None:
                # Hacer rose plot
                ax = fig.add_subplot(filas, columnas, idx + 1, polar=True)

                bins = [entry.get("bin") for entry in dic.values() if entry.get("bin") is not None]
                n_bins = 12
                counts = np.bincount(bins, minlength=n_bins)

                theta = np.linspace(0.0, 2 * np.pi, n_bins, endpoint=False)
                width = 2 * np.pi / n_bins

                ax.bar(theta, counts, width=width, bottom=0.0, align='center', alpha=0.7, color='blue', edgecolor='black')
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_title(titulo)
                continue
            ax = fig.add_subplot(filas, columnas, idx + 1)
            frames = sorted(dic.keys())
            x = list(range(len(frames)))

            y = [dic[f].get(clave_y, 0.0) for f in frames]
            ax.plot(x, y, label=clave_y.capitalize(), color='green')

            if clave_std:
                std = [dic[f].get(clave_std, 0.0) for f in frames]
                y = np.array(y)
                std = np.array(std)
                ax.fill_between(x, y - std, y + std, alpha=0.2, color='green', label="±1 STD")

            ax.set_title(titulo)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Valor")
            ax.grid(True)
            ax.legend(fontsize=8)

            if total > 4:
                ax.title.set_fontsize(10)
                ax.xaxis.label.set_fontsize(8)
                ax.yaxis.label.set_fontsize(8)
                ax.tick_params(labelsize=7)

        self.canvas.draw()
