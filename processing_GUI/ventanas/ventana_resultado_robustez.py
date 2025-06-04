import os
import json
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QComboBox, QTabWidget
)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class VentanaResultadoRobustez(QWidget):
    def __init__(self, carpeta_base, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Estadísticos de Robustez del Tracking")
        self.setMinimumSize(1000, 700)

        self.carpeta_base = carpeta_base
        self.path_longitudes = os.path.join(carpeta_base, "trayectorias_stats", "longitudes_continuas.json")
        self.path_histograma = os.path.join(carpeta_base, "trayectorias_stats", "histograma_distancias.json")

        self.data_longitudes = self._cargar_json(self.path_longitudes)
        self.data_histograma = self._cargar_json(self.path_histograma)

        self.ids_disponibles = sorted(int(i) for i in self.data_longitudes["por_id"].keys())

        self.setup_ui()

    def _cargar_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Media y std de longitudes por ID
        self.tabs.addTab(self._crear_tabla_longitudes(), "Longitudes por ID")

        # Tab 2: Nº de segmentos por ID
        self.tabs.addTab(self._crear_tabla_segmentos(), "Nº Segmentos por ID")

        # Tab 3: Rupturas por frame
        self.tabs.addTab(self._crear_grafico_rupturas(), "Rupturas por Frame")

        # Tab 4: Histograma de distancias (selector de ID)
        self.tabs.addTab(self._crear_histograma_distancias(), "Histograma de Distancias")

    def _crear_tabla_longitudes(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        ids = list(self.data_longitudes["por_id"].keys())
        medias = [self.data_longitudes["por_id"][id]["media"] for id in ids]
        stds = [self.data_longitudes["por_id"][id]["std"] for id in ids]

        ax.bar(range(len(ids)), medias, yerr=stds, capsize=5)
        ax.set_title("Media y STD de Longitudes por ID")
        ax.set_xlabel("ID")
        ax.set_ylabel("Longitud (frames)")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90)

        layout.addWidget(canvas)

        resumen = self.data_longitudes.get("resumen", {})
        lbl = QLabel(f"Media global: {resumen.get('media', 0):.2f} - STD global: {resumen.get('std', 0):.2f}")
        layout.addWidget(lbl)

        return widget

    def _crear_tabla_segmentos(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        ids = list(self.data_longitudes["por_id"].keys())
        segmentos = [len(self.data_longitudes["por_id"][id]["longitudes"]) for id in ids]

        ax.bar(range(len(ids)), segmentos)
        ax.set_title("Nº de Trayectorias Continuas por ID")
        ax.set_xlabel("ID")
        ax.set_ylabel("Segmentos")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90)

        layout.addWidget(canvas)
        return widget

    def _crear_grafico_rupturas(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        rupturas = self.data_longitudes.get("rupturas_por_frame", {})
        frames = sorted(rupturas.keys(), key=lambda x: int(x.split("_")[1]))
        valores = [rupturas[f] for f in frames]

        ax.plot(range(len(frames)), valores, marker='o')
        ax.set_title("Rupturas de Trayectorias por Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Rupturas")

        layout.addWidget(canvas)
        return widget

    def _crear_histograma_distancias(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        selector_layout = QHBoxLayout()
        self.combo_id = QComboBox()
        self.combo_id.addItem("-1 (Global)")
        for i in self.ids_disponibles:
            self.combo_id.addItem(str(i))
        boton = QPushButton("Actualizar")
        boton.clicked.connect(self.actualizar_histograma)

        selector_layout.addWidget(QLabel("ID:"))
        selector_layout.addWidget(self.combo_id)
        selector_layout.addWidget(boton)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        self.canvas_hist = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas_hist)
        self.actualizar_histograma()

        return widget
        
    def actualizar_histograma(self):
        id_text = self.combo_id.currentText().split()[0]
        id_key = "global" if id_text == "-1" else id_text

        if id_key == "global":
            datos = self.data_histograma.get("global", {})
            bins = self.data_histograma.get("bins", [])
        else:
            datos = self.data_histograma.get("por_id", {}).get(id_key, {})
            bin_size = self.data_histograma.get("bin_size", 5)
            hist = datos.get("histograma", [])
            bins = [i * bin_size for i in range(len(hist) + 1)]

        hist = datos.get("histograma", [])

        self.canvas_hist.figure.clf()
        ax = self.canvas_hist.figure.add_subplot(111)

        if bins and hist and len(hist) == len(bins) - 1:
            bin_width = bins[1] - bins[0] if len(bins) > 1 else 1
            ax.bar(bins[:-1], hist, width=bin_width, align='edge')
            ax.set_title(f"Histograma de Distancias - ID {id_key}")
            ax.set_xlabel("Distancia (px)")
            ax.set_ylabel("Frecuencia")
        else:
            ax.text(0.5, 0.5, "Sin datos o tamaño inconsistente", ha='center', va='center', transform=ax.transAxes)

        self.canvas_hist.draw()
