from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QGridLayout,
                               QLineEdit, QGroupBox, QHBoxLayout, QCheckBox, QScrollArea, QProgressBar, QSpinBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import os
import json
from processing_GUI.procesamiento.postprocesado import EstadoProceso, procesar_bbox_stats, procesar_mask_stats, procesar_tray_stats
import shutil
from multiprocessing import Process, Queue, cpu_count
from functools import partial

class VentanaPostprocesado(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.ventana_inicio = parent
        self.setWindowTitle("Herramienta de Postprocesado")
        self.setMinimumSize(1200, 800)
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Título
        titulo = QLabel("Generación de Descriptores Estadísticos")
        titulo.setFont(QFont('Segoe UI', 18, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        # Selector de carpeta de trabajo
        selector_layout = QHBoxLayout()
        self.lineedit_carpeta = QLineEdit()
        self.lineedit_carpeta.setPlaceholderText("Selecciona la carpeta de trabajo")
        boton_examinar = QPushButton("Examinar")
        boton_examinar.clicked.connect(self.seleccionar_carpeta)
        selector_layout.addWidget(self.lineedit_carpeta)
        selector_layout.addWidget(boton_examinar)
        layout.addLayout(selector_layout)

        # Opciones adicionales
        opciones_layout = QHBoxLayout()


        self.spin_nucleos = QSpinBox()
        self.spin_nucleos.setRange(1, cpu_count())
        self.spin_nucleos.setValue(min(4, cpu_count()))

        opciones_layout.addWidget(QLabel("Núcleos:"))
        opciones_layout.addWidget(self.spin_nucleos)

        
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 8)
        self.spin_batch.setValue(2)

        opciones_layout.addWidget(QLabel("Batch size:"))
        opciones_layout.addWidget(self.spin_batch)

        layout.addLayout(opciones_layout)

        # Scroll area para los checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        contenido_scroll = QWidget()
        scroll_layout = QVBoxLayout(contenido_scroll)

        # Grupo: BBoxes / centroides
        self.group_bbox = QGroupBox("Estadísticos basados en BBoxes / Centroide")
        layout_bbox = QVBoxLayout()
        self.chk_distribucion = QCheckBox("Distribución espacial por frames")
        self.chk_area = QCheckBox("Área media de los blobs")
        self.chk_distancia = QCheckBox("Distancia entre centroides")
        self.chk_agrupacion = QCheckBox("Coeficiente de agrupación")
        self.chk_entropia = QCheckBox("Entropía espacial")
        self.chk_exploracion = QCheckBox("Índice de exploración")
        self.chk_dist_centroide_global = QCheckBox("Distancia al centroide global")
        self.chk_densidad_local = QCheckBox("Densidad local")
        self.chk_vel_centroide_global = QCheckBox("Velocidad y ángulo del centroide global")
        layout_bbox.addWidget(self.chk_distribucion)
        layout_bbox.addWidget(self.chk_area)
        layout_bbox.addWidget(self.chk_distancia)
        layout_bbox.addWidget(self.chk_agrupacion)
        layout_bbox.addWidget(self.chk_entropia)
        layout_bbox.addWidget(self.chk_exploracion)
        layout_bbox.addWidget(self.chk_dist_centroide_global)
        layout_bbox.addWidget(self.chk_densidad_local)
        layout_bbox.addWidget(self.chk_vel_centroide_global)
        self.group_bbox.setLayout(layout_bbox)
        scroll_layout.addWidget(self.group_bbox)

        # Grupo: Máscaras
        self.group_mask = QGroupBox("Estadísticos basados en Máscaras")
        layout_mask = QVBoxLayout()
        self.chk_densidad = QCheckBox("Histograma de densidad")
        self.chk_centro_masa = QCheckBox("Centro de masa del grupo")
        self.chk_varianza = QCheckBox("Varianza espacial")
        self.chk_vel_grupo = QCheckBox("Velocidad y ángulo del grupo")
        self.chk_persistencia_mask = QCheckBox("Persistencia en zona ")
        self.chk_dispersion_temp = QCheckBox("Dispersión temporal")
        self.chk_entropia_binaria = QCheckBox("Entropía binaria")
        layout_mask.addWidget(self.chk_densidad)
        layout_mask.addWidget(self.chk_centro_masa)
        layout_mask.addWidget(self.chk_varianza)
        layout_mask.addWidget(self.chk_vel_grupo)
        layout_mask.addWidget(self.chk_persistencia_mask)
        layout_mask.addWidget(self.chk_dispersion_temp)
        layout_mask.addWidget(self.chk_entropia_binaria)
        self.group_mask.setLayout(layout_mask)
        scroll_layout.addWidget(self.group_mask)

        # Grupo: Trayectorias
        self.group_tray = QGroupBox("Estadísticos basados en Trayectorias")
        layout_tray = QVBoxLayout()
        self.chk_trayectorias = QCheckBox("Recalcular trayectorias")
        self.chk_longitud = QCheckBox("Longitud media de trayectorias")
        self.hist_distancia = QCheckBox("Histograma de distancias entre frames")
        self.chk_velocidad = QCheckBox("Velocidad de trayectoria")
        self.chk_disp_velocidad = QCheckBox("Dispersión de la velocidad")
        self.chk_cambio_angular = QCheckBox("Cambio angular")
        self.chk_persistencia_espacial = QCheckBox("Persistencia espacial")
        self.chk_direccion = QCheckBox("Dirección")
        self.chk_polarizacion = QCheckBox("Índice de polarización")
        layout_tray.addWidget(self.chk_trayectorias)
        layout_tray.addWidget(self.chk_longitud)
        layout_tray.addWidget(self.hist_distancia)
        layout_tray.addWidget(self.chk_velocidad)
        layout_tray.addWidget(self.chk_disp_velocidad)
        layout_tray.addWidget(self.chk_cambio_angular)
        layout_tray.addWidget(self.chk_persistencia_espacial)
        layout_tray.addWidget(self.chk_direccion)
        self.group_tray.setLayout(layout_tray)
        scroll_layout.addWidget(self.group_tray)

        scroll_area.setWidget(contenido_scroll)
        layout.addWidget(scroll_area)

        # Barra global
        self.etapa_actual = QLabel("Esperando instrucciones...")
        layout.addWidget(self.etapa_actual)

        self.barra_progreso_etapas = QProgressBar()
        layout.addWidget(self.barra_progreso_etapas)

        self.layout_videos = QGridLayout()
        self.barras_por_video = {}

        contenedor_scroll_videos = QWidget()
        contenedor_scroll_videos.setLayout(self.layout_videos)

        scroll_videos = QScrollArea()
        scroll_videos.setWidgetResizable(True)
        scroll_videos.setWidget(contenedor_scroll_videos)

        self.group_videos = QGroupBox("Progreso por vídeo")
        group_layout = QVBoxLayout()
        group_layout.addWidget(scroll_videos)
        self.group_videos.setLayout(group_layout)
        layout.addWidget(self.group_videos)





        # Botones inferiores
        botones_layout = QHBoxLayout()
        self.boton_atras = QPushButton("Atrás")
        self.boton_atras.clicked.connect(self.volver_a_inicio)
        self.boton_calcular = QPushButton("Calcular Estadísticos")
        self.boton_calcular.clicked.connect(self.lanzar_postprocesado)
        botones_layout.addWidget(self.boton_atras)
        botones_layout.addStretch()
        botones_layout.addWidget(self.boton_calcular)
        layout.addLayout(botones_layout)

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
            QProgressBar#barraMini {
                max-width: 140px;
            }
        """)

    def seleccionar_carpeta(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de trabajo")
        if carpeta:
            self.lineedit_carpeta.setText(carpeta)

    def lanzar_postprocesado(self):
        for i in reversed(range(self.layout_videos.count())):
            widget = self.layout_videos.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.barras_por_video.clear()
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            QMessageBox.critical(self, "Error", "Debes seleccionar una carpeta válida")
            return
        
        # Estadísticos seleccionados
        estadisticos_bbox = []
        if self.chk_distribucion.isChecked():
            estadisticos_bbox.append("distribucion_espacial")
        if self.chk_area.isChecked():
            estadisticos_bbox.append("area_media")
        if self.chk_distancia.isChecked():
            estadisticos_bbox.append("distancia_centroides")
        if self.chk_agrupacion.isChecked():
            estadisticos_bbox.append("coeficiente_agrupacion")
        if self.chk_entropia.isChecked():
            estadisticos_bbox.append("entropia_espacial")
        # if self.chk_persistencia_bbox.isChecked():
        #     estadisticos_bbox.append("persistencia_espacial")
        if self.chk_exploracion.isChecked():
            estadisticos_bbox.append("indice_exploracion")
        if self.chk_dist_centroide_global.isChecked():
            estadisticos_bbox.append("distancia_centroide_global")
        if self.chk_densidad_local.isChecked():
            estadisticos_bbox.append("densidad_local")
        if self.chk_vel_centroide_global.isChecked():
            estadisticos_bbox.append("velocidad_centroide")
        #aqui los demas estadísticos de BBoxes

        estadisticos_mask = []
        if self.chk_densidad.isChecked():
            estadisticos_mask.append("histograma_densidad")
        if self.chk_centro_masa.isChecked():
            estadisticos_mask.append("centro_masa_grupo")
        if self.chk_varianza.isChecked():
            estadisticos_mask.append("varianza_espacial")
        if self.chk_vel_grupo.isChecked():
            estadisticos_mask.append("velocidad_grupo")
        if self.chk_persistencia_mask.isChecked():
            estadisticos_mask.append("persistencia_zona")
        if self.chk_dispersion_temp.isChecked():
            estadisticos_mask.append("dispersion_temporal")
        if self.chk_entropia_binaria.isChecked():
            estadisticos_mask.append("entropia_binaria")
        #aqui los demas estadísticos de máscaras

        estadisticos_tray = []
        if self.chk_trayectorias.isChecked():
            estadisticos_tray.append("recalcular_trayectorias")
        if self.chk_longitud.isChecked():
            estadisticos_tray.append("longitud_media_trayectorias")
        if self.hist_distancia.isChecked():
            estadisticos_tray.append("histograma_distancias")
        if self.chk_velocidad.isChecked():
            estadisticos_tray.append("velocidades")
        if self.chk_disp_velocidad.isChecked():
            estadisticos_tray.append("dispersion_velocidad")
        if self.chk_cambio_angular.isChecked():
            estadisticos_tray.append("cambio_angular")
        if self.chk_persistencia_espacial.isChecked():
            estadisticos_tray.append("persistencia_espacial")
        if self.chk_direccion.isChecked():
            estadisticos_tray.append("direccion")
        #aqui los demas estadísticos de trayectorias 



        if not estadisticos_bbox and not estadisticos_mask and not estadisticos_tray:
            QMessageBox.warning(self, "Aviso", "Debes seleccionar al menos un estadístico para continuar.")
            return

        # Obtener número de núcleos y formato
        num_procesos = self.spin_nucleos.value()
        batch_size = self.spin_batch.value()

        # Crear objeto de estado y conectar feedback
        estado = EstadoProceso()
        estado.on_etapa = self.etapa_actual.setText
        estado.on_error = lambda msg: QMessageBox.critical(self, "Error en el postprocesado", msg)

        estado.on_total_videos = self.barra_progreso_etapas.setMaximum
        estado.on_video_progreso = lambda idx: self.barra_progreso_etapas.setValue(idx + 1)
        # Buscar dimensiones de entrada desde el primer vídeo válido
        try:
            subcarpetas = [os.path.join(carpeta, d) for d in os.listdir(carpeta)
                        if os.path.isdir(os.path.join(carpeta, d))]
            output_dims_path_yolo = os.path.join(subcarpetas[0], "bbox", "output_dims.json")
            with open(output_dims_path_yolo, "r") as f:
                dims = json.load(f)
            dimensiones_entrada_yolo = dims["output_dims"]
        except Exception:
            QMessageBox.critical(self, "Error", "No se pudo leer output_dims_yolo.json")
            print("Ruta usada para output_dims_yolo:", output_dims_path_yolo)
            return
        try:
            subcarpetas = [os.path.join(carpeta, d) for d in os.listdir(carpeta)
                        if os.path.isdir(os.path.join(carpeta, d))]
            output_dims_path_morph = os.path.join(subcarpetas[0], "masks_rle", "output_dims.json")
            with open(output_dims_path_morph, "r") as f:
                dims = json.load(f)
            dimensiones_entrada_morph = dims["output_dims"]
        except Exception:
            QMessageBox.critical(self, "Error", "No se pudo leer output_dims_masks.json")
            print("Ruta usada para output_dims_masks:", output_dims_path_morph)
            return

        # Llamada al procesado de bbox
        # Reiniciar barras de progreso
        self.etapa_actual.setText("Iniciando postprocesado global...")
        self.barra_progreso_etapas.setValue(0)


        # Para cada subdirectorio en la carpeta de trabajo, si existe un video mp4 en la carpeta de trabajo con el mismo nombre que dicho subdirectorio, si este no esta copiado ya en el subdirectorio, se copiará al subdirectorio
        # Además, la barra de progreso se actualizará con el número de subdirectorios procesados
        self.etapa_actual.setText("Copiando vídeos a subdirectorios...")
        self.barra_progreso_etapas.setValue(0)
        subcarpetas = [os.path.join(carpeta, d) for d in os.listdir(carpeta)
                      if os.path.isdir(os.path.join(carpeta, d)) and not d.startswith('.')]
        self.barra_progreso_etapas.setMaximum(len(subcarpetas))     
        for idx, subcarpeta in enumerate(subcarpetas):
            video_path = os.path.join(carpeta, f"{os.path.basename(subcarpeta)}.mp4")
            if os.path.exists(video_path) and not os.path.exists(os.path.join(subcarpeta, f"{os.path.basename(subcarpeta)}.mp4")):
                # Copiar el video al subdirectorio
                try:
                    print(f"Copiando {video_path} a {subcarpeta}")
                    shutil.copy2(video_path, os.path.join(subcarpeta, f"{os.path.basename(subcarpeta)}.mp4"))

                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo copiar el video: {e}")
                    return
            self.barra_progreso_etapas.setValue(idx + 1)

        # Crear colas para recibir progreso y etapa actual
        cola_bbox = Queue()
        cola_mask = Queue()
        cola_tray = Queue()

        videos = [os.path.join(carpeta, d) for d in os.listdir(carpeta)
                    if os.path.isdir(os.path.join(carpeta, d)) and not d.startswith('.')]

        self.barra_progreso_etapas.setMaximum(len(videos))
        self.barra_progreso_etapas.setValue(0)
        self.etapa_actual.setText("Procesando vídeos...")
        activos = []

        def lanzar_video(idx, path_video):
            nombre = os.path.basename(path_video)
            colas = {}
            procesos = {}

            layout_columna = QVBoxLayout()
            self.barras_por_video[nombre] = {}

            # Añadir el título con el nombre del vídeo (recortado si es largo)
            titulo_video = QLabel(nombre)
            titulo_video.setAlignment(Qt.AlignCenter)
            titulo_video.setStyleSheet("font-weight: bold; font-size: 12px;")
            layout_columna.addWidget(titulo_video)
            
            for tipo, lista, funcion, dims in [
                ("bbox", estadisticos_bbox, procesar_bbox_stats, dimensiones_entrada_yolo),
                ("mask", estadisticos_mask, procesar_mask_stats, dimensiones_entrada_morph),
                ("tray", estadisticos_tray, procesar_tray_stats, dimensiones_entrada_yolo)
            ]:
                if lista:
                    q = Queue()
                    p = Process(target=funcion, args=(path_video, lista, num_procesos, dims, q))
                    p.start()
                    colas[tipo] = q
                    procesos[tipo] = p

                    label = QLabel(f"[{tipo.upper()}] Iniciando...")
                    barra = QProgressBar()
                    barra.setObjectName("barraMini")
                    barra.setMaximum(100)
                    layout_columna.addWidget(label)
                    layout_columna.addWidget(barra)
                    self.barras_por_video[nombre][tipo] = (label, barra)

            contenedor = QWidget()
            contenedor.setLayout(layout_columna)
            col_idx = idx  # cada columna es un vídeo distinto
            self.layout_videos.addWidget(contenedor, 0, col_idx)

            return {"nombre": nombre, "procesos": procesos, "colas": colas, "widget": contenedor}

        from PySide6.QtCore import QTimer
        idx_actual = 0
        videos_totales = len(videos)

        def actualizar():
            nonlocal idx_actual

            # Lanzar nuevos si caben
            while len(activos) < batch_size and idx_actual < videos_totales:
                activos.append(lanzar_video(idx_actual, videos[idx_actual]))
                idx_actual += 1

            # Procesar mensajes de los activos
            for v in activos[:]:
                terminado = True
                for tipo, q in v["colas"].items():
                    while not q.empty():
                        msg = q.get()
                        if isinstance(msg, tuple) and len(msg) == 2:
                            if msg[0] == "etapa":
                                self.barras_por_video[v["nombre"]][tipo][0].setText(f"[{tipo.upper()}] {msg[1]}")
                            elif msg[0] == "progreso":
                                self.barras_por_video[v["nombre"]][tipo][1].setValue(msg[1])
                    if v["procesos"][tipo].is_alive():
                        terminado = False

                if terminado:
                    for p in v["procesos"].values():
                        p.join()
                    v["widget"].setVisible(False)
                    activos.remove(v)
                    self.barra_progreso_etapas.setValue(self.barra_progreso_etapas.value() + 1)

            if idx_actual >= videos_totales and not activos:
                self.etapa_actual.setText("✅ Postprocesado completado.")
                timer.stop()

        timer = QTimer(self)
        timer.timeout.connect(actualizar)
        timer.start(300)



    


    def volver_a_inicio(self):
        self.close()
        if self.ventana_inicio:
            self.ventana_inicio.show()
