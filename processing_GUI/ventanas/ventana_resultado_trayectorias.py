import os
import json
import cv2
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout,
    QComboBox, QCheckBox, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage

from processing_GUI.procesamiento.visualizacion import cargar_recorte, aplicar_recorte, cargar_output_dims, dibujar_bboxes


class VentanaResultadoTrayectorias(QMainWindow):
    def __init__(self, ruta_video, carpeta_base, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visualización de Trayectorias por ID")
        self.setMinimumSize(960, 600)

        self.ruta_video = ruta_video
        self.carpeta_base = carpeta_base
        self.parent = parent

        self.cap = cv2.VideoCapture(self.ruta_video)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No se pudo abrir el vídeo.")
            self.close()
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames == 0:
            QMessageBox.critical(self, "Error", "El vídeo no contiene frames.")
            self.close()
            return

        self.frame_idx = 0
        self.frame_actual_leido = -1
        self.frame_cache = None

        _, self.recorte_bbox = cargar_recorte(self.carpeta_base)
        self.output_dims = cargar_output_dims(os.path.join(self.carpeta_base, "bbox"))

        self.timer = QTimer()
        self.timer.timeout.connect(self.mostrar_siguiente_frame)

        self.bbox_data = self._cargar_json_bbox()
        if not self.bbox_data:
            QMessageBox.warning(self, "Advertencia", "No se encontró o no se pudo cargar el JSON de trayectorias.")

        self.colores_por_id = self._generar_colores_por_id()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        self.id_seleccionado = -1
        self.dibujar_trayectoria = True

        self.setup_ui()
        self.actualizar_parametros()
        self.cargar_y_mostrar_frame(self.frame_idx)

    def _cargar_json_bbox(self):
        ruta_json = os.path.join(self.carpeta_base, "trayectorias_stats", "trayectorias.json")
        if not os.path.exists(ruta_json):
            return {}
        try:
            with open(ruta_json, "r") as f:
                return json.load(f)
        except Exception as e:
            return {}

    def _generar_colores_por_id(self):
        colores = {}
        for id_val_str in self.bbox_data.keys():
            color = tuple(int(c) for c in np.random.randint(10, 255, 3))
            colores[id_val_str] = color
        return colores

    def setup_ui(self):
        self.label_frame = QLabel("Cargando vídeo...")
        self.label_frame.setAlignment(Qt.AlignCenter)
        self.label_frame.setMinimumSize(800, 450)
        self.layout.addWidget(self.label_frame)

        # Parámetros
        param_layout = QHBoxLayout()
        self.combo_id = QComboBox()
        ids = sorted(int(i) for i in self.bbox_data.keys())
        self.combo_id.addItem("-1 (Todos)")
        for id_val in ids:
            self.combo_id.addItem(str(id_val))
        self.checkbox_trayectoria = QCheckBox("Dibujar trayectorias")
        self.checkbox_trayectoria.setChecked(True)
        self.boton_actualizar = QPushButton("Actualizar")
        self.boton_actualizar.clicked.connect(self.actualizar_parametros)
        param_layout.addWidget(QLabel("ID a visualizar:"))
        param_layout.addWidget(self.combo_id)
        param_layout.addWidget(self.checkbox_trayectoria)
        param_layout.addWidget(self.boton_actualizar)
        param_layout.addStretch()
        self.layout.addLayout(param_layout)

        # Controles
        controls = QHBoxLayout()
        self.boton_anterior = QPushButton("⏮️")
        self.boton_siguiente = QPushButton("⏭️")
        self.boton_play = QPushButton("▶️")
        self.boton_atras = QPushButton("Atrás")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.total_frames - 1)
        self.slider.sliderMoved.connect(self.cargar_y_mostrar_frame)
        self.texto_estado = QLabel("Frame 0 / 0")

        self.boton_anterior.clicked.connect(self.frame_anterior)
        self.boton_siguiente.clicked.connect(self.frame_siguiente)
        self.boton_play.clicked.connect(self.toggle_reproduccion)
        self.boton_atras.clicked.connect(self.volver)

        controls.addWidget(self.boton_anterior)
        controls.addWidget(self.boton_play)
        controls.addWidget(self.boton_siguiente)
        controls.addWidget(self.slider)
        controls.addWidget(self.texto_estado)
        controls.addStretch()
        controls.addWidget(self.boton_atras)
        self.layout.addLayout(controls)


    def actualizar_parametros(self):
        self.id_seleccionado = int(self.combo_id.currentText().split()[0])
        self.dibujar_trayectoria = self.checkbox_trayectoria.isChecked()

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

        if self.recorte_bbox:
            frame = aplicar_recorte(frame, self.recorte_bbox)
            frame = cv2.resize(frame, self.output_dims, interpolation=cv2.INTER_LINEAR)


        frame_key = f"frame_{self.frame_idx:05d}"
        print(f"[DEBUG] Buscando en frame_key: {frame_key}")

        for id_val_str, frames in self.bbox_data.items():
            if self.id_seleccionado != -1 and int(id_val_str) != self.id_seleccionado:
                continue
            print(f"[CHECK] Claves de ID {id_val_str}")
            if frame_key not in frames:
                print(f"[DEBUG] ID {id_val_str} no tiene datos en {frame_key}")
                continue
            color = self.colores_por_id[id_val_str]
            print(f"[DRAW] Dibujando ID {id_val_str} en frame {frame_key}")
            x1, y1, x2, y2 = frames[frame_key]['bbox']
            frame = dibujar_bboxes(frame, [(int(x1), int(y1), int(x2), int(y2))], color, thickness=2)
            cv2.putText(frame, f"ID: {id_val_str}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if self.dibujar_trayectoria:
                puntos = []
                umbral_px = 50  # puedes ajustar este valor

                for i in range(self.frame_idx + 1):
                    key = f"frame_{i:05d}"
                    if key in frames:
                        cx, cy = frames[key]['centroide']
                        punto_actual = (int(cx), int(cy))

                        if puntos:
                            # calcular distancia al punto anterior
                            dx = punto_actual[0] - puntos[-1][0]
                            dy = punto_actual[1] - puntos[-1][1]
                            distancia = (dx**2 + dy**2) ** 0.5

                            if distancia > umbral_px:
                                puntos = []  # resetear trayectoria

                        puntos.append(punto_actual)

                        if len(puntos) >= 2:
                            cv2.line(frame, puntos[-2], puntos[-1], color, 2)


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
