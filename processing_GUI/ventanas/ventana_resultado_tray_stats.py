import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget,
    QComboBox, QPushButton, QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class VentanaResultadoTrayStats(QWidget):
    def __init__(self, carpeta_base, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Velocidades de Peces")
        self.setMinimumSize(1000, 700)

        self.carpeta_base = carpeta_base
        self.path_velocidades = os.path.join(carpeta_base, "trayectorias_stats", "velocidades.json")
        self.path_dispersion = os.path.join(carpeta_base, "trayectorias_stats", "dispersion_velocidades.json")
        self.path_angulos = os.path.join(carpeta_base, "trayectorias_stats", "angulo_cambio_direccion.json")
        self.path_persistencia = os.path.join(carpeta_base, "trayectorias_stats", "persistencia_espacial.json")
        self.path_direcciones = os.path.join(carpeta_base, "trayectorias_stats", "direcciones.json")

        self.data_velocidades = self._cargar_json(self.path_velocidades)
        self.data_dispersion = self._cargar_json(self.path_dispersion)
        self.data_angulos = self._cargar_json(self.path_angulos)
        self.data_persistencia = self._cargar_json(self.path_persistencia)
        self.data_direcciones = self._cargar_json(self.path_direcciones)



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

        self.tabs.addTab(self._crear_tab_velocidad_global(), "Velocidad Media Global por Frame")
        self.tabs.addTab(self._crear_tab_velocidad_por_id(), "Velocidad Media por ID")
        self.tabs.addTab(self._crear_tab_velocidad_id_interactiva(), "Velocidad por Frame (ID seleccionable)")
        self.tabs.addTab(self._crear_tab_dispersion_velocidades(), "Dispersión de las velocidades")
        self.tabs.addTab(self._crear_tab_cambio_angular(), "Cambio angular por ID")
        self.tabs.addTab(self._crear_tab_media_angular(), "Media de Ángulos por Frame")
        self.tabs.addTab(self._crear_tab_giros_bruscos(), "Porcentaje de Giros Bruscos")
        self.tabs.addTab(self._crear_tab_persistencia_por_id(), "Persistencia Media por ID")
        self.tabs.addTab(self._crear_tab_persistencia_por_ventana(), "Persistencia por Ventana")
        self.tabs.addTab(self._crear_tab_mapa_persistencia(), "Mapa de Persistencia por Celda")
        self.tabs.addTab(self._crear_tab_persistencia_id_interactiva(), "Persistencia por ID (detalle)")
        self.tabs.addTab(self._crear_tab_direccion_media_std(), "Dirección Media por Frame")
        self.tabs.addTab(self._crear_tab_rosa_direcciones(), "Rosa de Direcciones Global") 
        self.tabs.addTab(self._crear_tab_entropia_direccional(), "Entropía Direccional por Frame")
        self.tabs.addTab(self._crear_tab_direccion_id_interactiva(), "Dirección por ID")
        self.tabs.addTab(self._crear_tab_polarizacion(), "Polarización por Frame")



    def _crear_tab_velocidad_global(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        datos_media = self.data_velocidades.get("media_por_frame", {})
        frames = sorted(datos_media.keys(), key=lambda f: int(f.split("_")[1]))
        medias = [datos_media[f] for f in frames]

        ax.plot(range(len(frames)), medias, color="blue", label="Media")
        ax.set_title("Velocidad Media Global por Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Velocidad (px/frame)")
        ax.legend()

        layout.addWidget(canvas)
        return widget

    def _crear_tab_velocidad_por_id(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        datos_por_id = self.data_velocidades.get("por_id", {})
        ids = sorted(datos_por_id.keys(), key=int)
        medias = [
            np.mean(list(datos_por_id[_id].values()))
            if datos_por_id[_id] else 0.0
            for _id in ids
        ]

        ax.bar(range(len(ids)), medias)
        ax.set_title("Velocidad Media por ID")
        ax.set_xlabel("ID")
        ax.set_ylabel("Velocidad (px/frame)")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90)

        layout.addWidget(canvas)
        return widget

    def _crear_tab_velocidad_id_interactiva(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.combo_id = QComboBox()
        ids = sorted(self.data_velocidades.get("por_id", {}).keys(), key=int)
        for _id in ids:
            self.combo_id.addItem(str(_id))

        boton_actualizar = QPushButton("Actualizar")
        boton_actualizar.clicked.connect(self.actualizar_grafico_id)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Selecciona ID:"))
        top_layout.addWidget(self.combo_id)
        top_layout.addWidget(boton_actualizar)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.canvas_id = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas_id)

        self.actualizar_grafico_id()
        return widget

    def actualizar_grafico_id(self):
        id_str = self.combo_id.currentText()
        datos_id = self.data_velocidades.get("por_id", {}).get(id_str, {})
        frames = sorted(datos_id.keys(), key=lambda f: int(f.split("_")[1]))
        velocidades = [datos_id[f] for f in frames]

        self.canvas_id.figure.clf()
        ax = self.canvas_id.figure.add_subplot(111)
        ax.plot(range(len(frames)), velocidades, color="green")
        ax.set_title(f"Velocidad por Frame - ID {id_str}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Velocidad (px/frame)")
        self.canvas_id.draw()

    def _crear_tab_dispersion_velocidades(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        datos_dispersion = self.data_dispersion.get("dispersion_por_frame", {})
        frames = sorted(datos_dispersion.keys(), key=lambda f: int(f.split("_")[1]))
        valores = [datos_dispersion[f] for f in frames]

        ax.plot(range(len(frames)), valores, color="purple")
        ax.set_title("Dispersión de Velocidades por Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Desviación típica (px/frame)")

        layout.addWidget(canvas)
        return widget

    def _crear_tab_cambio_angular(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.combo_id_ang = QComboBox()
        ids = sorted(self.data_angulos.get("por_id", {}).keys(), key=int)
        for _id in ids:
            self.combo_id_ang.addItem(str(_id))

        boton_actualizar = QPushButton("Actualizar")
        boton_actualizar.clicked.connect(self.actualizar_grafico_angular)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Selecciona ID:"))
        top_layout.addWidget(self.combo_id_ang)
        top_layout.addWidget(boton_actualizar)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.canvas_ang = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas_ang)

        self.actualizar_grafico_angular()
        return widget

    def actualizar_grafico_angular(self):
        id_str = self.combo_id_ang.currentText()
        datos_id = self.data_angulos.get("por_id", {}).get(id_str, {})
        frames = sorted(datos_id.keys(), key=lambda f: int(f.split("_")[1]))
        angulos = [datos_id[f] for f in frames]

        self.canvas_ang.figure.clf()
        ax = self.canvas_ang.figure.add_subplot(111)
        ax.plot(range(len(frames)), angulos, color="orange")
        ax.set_title(f"Cambio angular por Frame - ID {id_str}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Ángulo (grados)")
        self.canvas_ang.draw()

    def _crear_tab_media_angular(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        datos_media = self.data_angulos.get("media_por_frame", {})
        datos_std = self.data_angulos.get("std_por_frame", {})

        frames = sorted(datos_media.keys(), key=lambda f: int(f.split("_")[1]))
        medias = [datos_media[f] for f in frames]
        stds = [datos_std.get(f, 0) for f in frames]

        ax.plot(range(len(frames)), medias, color="darkred", label="Media")
        ax.fill_between(range(len(frames)), np.array(medias) - np.array(stds), np.array(medias) + np.array(stds), color="darkred", alpha=0.3, label="± STD")

        ax.set_title("Media de Ángulos por Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Media de Ángulo (grados)")
        ax.legend()

        layout.addWidget(canvas)
        return widget

    def _crear_tab_giros_bruscos(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        datos = self.data_angulos.get("porcentaje_giros_bruscos", {})
        frames = sorted(datos.keys(), key=lambda f: int(f.split("_")[1]))
        valores = [datos[f] for f in frames]

        ax.plot(range(len(frames)), valores, color="brown")
        ax.set_title("Porcentaje de Giros Bruscos por Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Proporción de IDs (>100º)")
        ax.set_ylim(0, 1)

        layout.addWidget(canvas)
        return widget

    def _crear_tab_persistencia_por_id(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        medias = self.data_persistencia.get("media_por_id", {})
        stds = self.data_persistencia.get("std_por_id", {})
        ids = sorted(medias.keys(), key=int)
        valores = [medias[i] for i in ids]
        errores = [stds.get(i, 0) for i in ids]

        ax.bar(range(len(ids)), valores, yerr=errores, capsize=4)
        ax.set_title("Persistencia Media por ID")
        ax.set_xlabel("ID")
        ax.set_ylabel("Duración media (frames)")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90)

        layout.addWidget(canvas)
        return widget

    def _crear_tab_persistencia_por_ventana(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        selector_layout = QHBoxLayout()
        self.combo_ventana = QComboBox()
        for w in ["32", "64", "128", "256", "512"]:
            self.combo_ventana.addItem(w)
        boton = QPushButton("Actualizar")
        boton.clicked.connect(self.actualizar_persistencia_ventana)
        selector_layout.addWidget(QLabel("Tamaño de ventana:"))
        selector_layout.addWidget(self.combo_ventana)
        selector_layout.addWidget(boton)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)

        self.canvas_persistencia = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas_persistencia)
        self._graficar_persistencia_ventana("32")

        return widget

    def actualizar_persistencia_ventana(self):
        tamaño = self.combo_ventana.currentText()
        self._graficar_persistencia_ventana(tamaño)

    def _graficar_persistencia_ventana(self, tamaño):
        self.canvas_persistencia.figure.clf()
        ax = self.canvas_persistencia.figure.add_subplot(111)

        datos = self.data_persistencia.get("por_ventana", {}).get(tamaño, {})
        medias = datos.get("media", [])
        stds = datos.get("std", [])

        x = range(len(medias))
        ax.plot(x, medias, label="Media", color="navy")
        ax.fill_between(x, np.array(medias) - np.array(stds), np.array(medias) + np.array(stds), alpha=0.3, color="navy", label="± STD")
        ax.set_title(f"Persistencia Media - Ventanas de {tamaño} frames")
        ax.set_xlabel("Ventana")
        ax.set_ylabel("Duración media (frames)")
        ax.legend()
        self.canvas_persistencia.draw()

    def _crear_tab_mapa_persistencia(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        data = self.data_persistencia.get("por_celda", {})

        grid = np.zeros((5, 5))
        for key, val in data.items():
            i, j = map(int, key.split("_"))
            grid[j, i] = val["media"]  # nota: j, i por convención de imagen

        im = ax.imshow(grid, cmap="viridis")
        plt.colorbar(im, ax=ax)
        ax.set_title("Mapa de Persistencia Media por Celda")
        ax.set_xlabel("Celda X")
        ax.set_ylabel("Celda Y")

        layout.addWidget(canvas)
        return widget

    
    def _crear_tab_persistencia_id_interactiva(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.combo_id_persist = QComboBox()
        ids = sorted(self.data_persistencia.get("por_id", {}).keys(), key=int)
        for _id in ids:
            self.combo_id_persist.addItem(str(_id))

        boton_actualizar = QPushButton("Actualizar")
        boton_actualizar.clicked.connect(self.actualizar_grafico_persist_id)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Selecciona ID:"))
        top_layout.addWidget(self.combo_id_persist)
        top_layout.addWidget(boton_actualizar)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.canvas_persist_id = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas_persist_id)

        self.actualizar_grafico_persist_id()
        return widget

    def actualizar_grafico_persist_id(self):
        id_str = self.combo_id_persist.currentText()
        estancias = self.data_persistencia.get("por_id", {}).get(id_str, [])
        duraciones = [e["duración"] for e in estancias]

        self.canvas_persist_id.figure.clf()
        ax = self.canvas_persist_id.figure.add_subplot(111)
        ax.plot(range(len(duraciones)), duraciones, marker='o', linestyle='-')
        ax.set_title(f"Secuencia de estancias - ID {id_str}")
        ax.set_xlabel("# Estancia")
        ax.set_ylabel("Duración (frames)")
        self.canvas_persist_id.draw()

    def _crear_tab_direccion_media_std(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        datos_media = self.data_direcciones.get("media_por_frame", {})
        datos_std = self.data_direcciones.get("std_por_frame", {})

        frames = sorted(datos_media.keys(), key=lambda x: int(x.split("_")[1]))
        x = list(range(len(frames)))
        media = [datos_media[f] for f in frames]
        std = [datos_std.get(f, 0) for f in frames]

        ax.plot(x, media, label="Media", color="green")
        ax.fill_between(x, np.array(media) - np.array(std), np.array(media) + np.array(std), alpha=0.3, label="± STD", color="green")
        ax.set_title("Dirección Media por Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Ángulo (grados)")
        ax.legend()

        layout.addWidget(canvas)
        return widget

    def _crear_tab_rosa_direcciones(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        canvas = FigureCanvas(fig)

        histo = self.data_direcciones.get("histograma_global", {})
        bins = histo.get("bins", [])
        frec = histo.get("frecuencias", [])

        if len(bins) > 1:
            bin_centers_deg = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
            bin_centers_rad = [np.deg2rad(deg) for deg in bin_centers_deg]
            width = np.deg2rad(bins[1] - bins[0])
            ax.bar(bin_centers_rad, frec, width=width, bottom=0.0, align='center', color='teal', edgecolor='black')
            ax.set_theta_zero_location("E")  # 0° en el Este (derecha)
            ax.set_theta_direction(-1)       # sentido horario
            ax.set_title("Rosa de Direcciones Global")
        layout.addWidget(canvas)
        return widget

    def _crear_tab_entropia_direccional(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        entropia = self.data_direcciones.get("entropia_por_frame", {})
        frames = sorted(entropia.keys(), key=lambda x: int(x.split("_")[1]))
        x = list(range(len(frames)))
        valores = [entropia[f] for f in frames]

        ax.plot(x, valores, color="purple")
        ax.set_title("Entropía Direccional por Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Entropía (bits)")

        layout.addWidget(canvas)
        return widget

    def _crear_tab_direccion_id_interactiva(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.combo_id_dir = QComboBox()
        ids = sorted(self.data_direcciones.get("por_id", {}).keys(), key=int)
        for _id in ids:
            self.combo_id_dir.addItem(str(_id))

        boton_actualizar = QPushButton("Actualizar")
        boton_actualizar.clicked.connect(self.actualizar_grafico_direccion_id)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Selecciona ID:"))
        top_layout.addWidget(self.combo_id_dir)
        top_layout.addWidget(boton_actualizar)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.canvas_dir_id = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas_dir_id)

        self.actualizar_grafico_direccion_id()
        return widget

    def actualizar_grafico_direccion_id(self):
        id_str = self.combo_id_dir.currentText()
        datos = self.data_direcciones.get("por_id", {}).get(id_str, {})
        frames = sorted(datos.keys(), key=lambda x: int(x.split("_")[1]))
        angulos = [datos[f] for f in frames]

        self.canvas_dir_id.figure.clf()
        ax = self.canvas_dir_id.figure.add_subplot(111)
        ax.plot(range(len(angulos)), angulos, marker='o', linestyle='-')
        ax.set_title(f"Dirección por Frame - ID {id_str}")
        ax.set_xlabel("# Movimiento")
        ax.set_ylabel("Ángulo (grados)")
        self.canvas_dir_id.draw()

    def _crear_tab_polarizacion(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        datos = self.data_direcciones.get("polarizacion_por_frame", {})
        frames = sorted(datos.keys(), key=lambda x: int(x.split("_")[1]))
        x = list(range(len(frames)))
        valores = [datos[f] for f in frames]

        ax.plot(x, valores, color="darkorange")
        ax.set_title("Polarización por Frame")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Polarización (0-1)")
        ax.set_ylim(0, 1.05)

        layout.addWidget(canvas)
        return widget


