from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox,
                               QLineEdit, QGroupBox, QHBoxLayout, QCheckBox, QScrollArea, QProgressBar)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import os
import json
from processing_GUI.procesamiento.postprocesado import EstadoProceso, procesar_bbox_stats, procesar_mask_stats, procesar_tray_stats
import shutil

class VentanaPostprocesado(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.ventana_inicio = parent
        self.setWindowTitle("Análisis de comportamiento")
        self.setMinimumSize(700, 700)
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Título
        titulo = QLabel("Análisis de comportamiento de peces")
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

        from PySide6.QtWidgets import QSpinBox, QComboBox
        from multiprocessing import cpu_count

        self.spin_nucleos = QSpinBox()
        self.spin_nucleos.setRange(1, cpu_count())
        self.spin_nucleos.setValue(min(4, cpu_count()))

        opciones_layout.addWidget(QLabel("Núcleos:"))
        opciones_layout.addWidget(self.spin_nucleos)
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

        # Progreso
        self.etapa_actual = QLabel("Esperando instrucciones...")
        layout.addWidget(self.etapa_actual)

        self.barra_progreso_etapas = QProgressBar()
        layout.addWidget(self.barra_progreso_etapas)

        self.barra_progreso_especifica = QProgressBar()
        layout.addWidget(self.barra_progreso_especifica)


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
        """)

    def seleccionar_carpeta(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de trabajo")
        if carpeta:
            self.lineedit_carpeta.setText(carpeta)

    def lanzar_postprocesado(self):
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

        # Crear objeto de estado y conectar feedback
        estado = EstadoProceso()
        estado.on_etapa = self.etapa_actual.setText
        estado.on_progreso = self.barra_progreso_especifica.setValue
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
            QMessageBox.critical(self, "Error", "No se pudo leer output_dims.json")
            return
        try:
            subcarpetas = [os.path.join(carpeta, d) for d in os.listdir(carpeta)
                        if os.path.isdir(os.path.join(carpeta, d))]
            output_dims_path_morph = os.path.join(subcarpetas[0], "masks", "output_dims.json")
            with open(output_dims_path_morph, "r") as f:
                dims = json.load(f)
            dimensiones_entrada_morph = dims["output_dims"]
        except Exception:
            QMessageBox.critical(self, "Error", "No se pudo leer output_dims.json")
            return

        # Llamada al procesado de bbox
        # Reiniciar barras de progreso
        self.etapa_actual.setText("Iniciando postprocesado...")
        self.barra_progreso_etapas.setValue(0)
        self.barra_progreso_especifica.setValue(0)

        # Para cada subdirectorio en la carpeta de trabajo, si existe un video mp4 en la carpeta de trabajo con el mismo nombre que dicho subdirectorio, si este no esta copiado ya en el subdirectorio, se copiará al subdirectorio
        # Además, la barra de progreso se actualizará con el número de subdirectorios procesados
        self.etapa_actual.setText("Copiando vídeos a subdirectorios...")
        self.barra_progreso_etapas.setValue(0)
        self.barra_progreso_especifica.setValue(0)
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
            estado.on_video_progreso(idx)

        if estadisticos_bbox:
            print(f"Procesando BBoxes con los siguientes estadísticos: {estadisticos_bbox}")
            procesar_bbox_stats(
                carpeta_trabajo=carpeta,
                estadisticos_seleccionados=estadisticos_bbox,
                num_procesos=num_procesos,
                dimensiones_entrada=dimensiones_entrada_yolo,
                estado=estado
            )


        if estadisticos_mask:
            print(f"Procesando Máscaras con los siguientes estadísticos: {estadisticos_mask}")
            procesar_mask_stats(
                carpeta_trabajo=carpeta,
                estadisticos_seleccionados=estadisticos_mask,
                num_procesos=num_procesos,
                dimensiones_entrada=dimensiones_entrada_morph,
                estado=estado
            )

        if estadisticos_tray:
            print(f"Procesando Trayectorias con los siguientes estadísticos: {estadisticos_tray}")
            procesar_tray_stats(
                carpeta_trabajo=carpeta,
                estadisticos_seleccionados=estadisticos_tray,
                num_procesos=num_procesos,
                dimensiones_entrada=dimensiones_entrada_yolo,
                estado=estado
            )

    


    def volver_a_inicio(self):
        self.close()
        if self.ventana_inicio:
            self.ventana_inicio.show()
