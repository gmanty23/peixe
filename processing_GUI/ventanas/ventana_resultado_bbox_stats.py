import os
import json
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy
import math




class VentanaResultadoBBoxStats(QWidget):
    def __init__(self, ruta_video, carpeta_base, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Área media de blobs por frame")
        self.setMinimumSize(900, 600)

        self.ruta_video = ruta_video
        self.carpeta_base = carpeta_base

        self.setup_ui()
        self.cargar_datos()
        self.mostrar_grafico()

    def setup_ui(self):
        self.layout = QVBoxLayout(self)

        self.label_titulo = QLabel("Área media ± desviación típica por frame")
        self.label_titulo.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.label_titulo)

        self.canvas = FigureCanvas(Figure(figsize=(10, 5)))
        self.layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)

    def cargar_datos(self):
        ruta_json = os.path.join(self.carpeta_base, "bbox_stats", "areas_blobs.json")
        if not os.path.exists(ruta_json):
            self.datos = None
            return

        with open(ruta_json, "r") as f:
            self.datos = json.load(f)

    def mostrar_grafico(self):
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.figure.clf()
        fig = self.canvas.figure
        fig.set_constrained_layout(True)

        # Estadísticos posibles y su clave principal
        estadisticos = [
            ("Área media de blobs", "areas_blobs.json", "media", "std"),
            ("Distancia media entre centroides", "distancia_centroides.json", "media", "std"),
            ("Coeficiente de agrupación", "coef_agrupacion.json", "agrupacion", None),
            ("Entropía espacial", "entropia.json", "entropia", None),
            ("Índice de exploración", "exploracion.json", "por_ventana", None),
            ("Distancia al centroide grupal", "distancia_centroide_grupal.json", "media", "std"),
            ("Densidad local", "densidad_local.json", "densidad_media", "std")

        ]

        # Filtrar los que existan
        existentes = []
        for titulo, archivo, clave_y, clave_std in estadisticos:
            ruta = os.path.join(self.carpeta_base, "bbox_stats", archivo)
            if os.path.exists(ruta):
                with open(ruta, "r") as f:
                    datos = json.load(f)
                existentes.append((titulo, archivo, datos, clave_y, clave_std))

        total = len(existentes)

        if total == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("No se encontraron archivos de estadísticos.")
            ax.axis("off")
            self.canvas.draw()
            return

        # Calcular rejilla dinámica
        columnas = min(3, total)
        filas = math.ceil(total / columnas)

        for idx, (titulo, archivo, dic, clave_y, clave_std) in enumerate(existentes):
            print(f"[DEBUG] Cargando: {archivo}")
            ax = fig.add_subplot(filas, columnas, idx + 1)
            if archivo == "exploracion.json":
                print(f"[DEBUG] Procesando exploración: {titulo}")
                y_dict = dic.get("por_ventana", {})
                frames = sorted(y_dict.keys())
                x = list(range(len(frames)))
                y = [y_dict[k] for k in frames]
                # Extraer el valor global y añadirlo como texto sobre la gráfica
                valor_global = dic.get("global", None)
                if valor_global is not None:
                    texto_global = f"Índice global: {valor_global:.3f}"
                    ax.text(0.95, 0.95, texto_global,
                            transform=ax.transAxes,
                            ha='right', va='top',
                            fontsize=8, color='gray')
            else:
                print(f"[DEBUG] Procesando: {titulo}")
                frames = sorted(dic.keys())
                x = list(range(len(frames)))
                y = [dic[f].get(clave_y, 0.0) for f in frames]

            ax.plot(x, y, label=clave_y.capitalize(), color='blue')

            if clave_std:
                std = [dic[f].get(clave_std, 0.0) for f in frames]
                y = np.array(y)
                std = np.array(std)
                ax.fill_between(x, y - std, y + std, alpha=0.2, color='blue', label="±1 STD")

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



