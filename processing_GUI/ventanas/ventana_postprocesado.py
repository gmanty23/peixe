from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox,
                               QLineEdit, QGroupBox, QHBoxLayout, QCheckBox, QScrollArea, QProgressBar)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import os
import json
from processing_GUI.procesamiento.postprocesado import EstadoProceso, procesar_bbox_stats, procesar_mask_stats

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
        layout_bbox.addWidget(self.chk_distribucion)
        layout_bbox.addWidget(self.chk_area)
        layout_bbox.addWidget(self.chk_distancia)
        layout_bbox.addWidget(self.chk_agrupacion)
        layout_bbox.addWidget(self.chk_entropia)
        layout_bbox.addWidget(self.chk_exploracion)
        layout_bbox.addWidget(self.chk_dist_centroide_global)
        layout_bbox.addWidget(self.chk_densidad_local)
        self.group_bbox.setLayout(layout_bbox)
        scroll_layout.addWidget(self.group_bbox)

        # Grupo: Máscaras
        self.group_mask = QGroupBox("Estadísticos basados en Máscaras")
        layout_mask = QVBoxLayout()
        self.chk_densidad = QCheckBox("Histograma de densidad")
        self.chk_centro_masa = QCheckBox("Centro de masa del grupo")
        self.chk_varianza = QCheckBox("Varianza espacial")
        self.chk_vel_grupo = QCheckBox("Velocidad del grupo")
        self.chk_compact = QCheckBox("Compactidad del grupo")
        self.chk_autocorrelacion = QCheckBox("Autocorrelación espacial")
        self.chk_persistencia_mask = QCheckBox("Persistencia en zona (a partir de máscaras)")
        self.chk_tiempo_en_zona = QCheckBox("Tiempo en zona de interés")
        layout_mask.addWidget(self.chk_densidad)
        layout_mask.addWidget(self.chk_centro_masa)
        layout_mask.addWidget(self.chk_varianza)
        layout_mask.addWidget(self.chk_vel_grupo)
        layout_mask.addWidget(self.chk_compact)
        layout_mask.addWidget(self.chk_autocorrelacion)
        layout_mask.addWidget(self.chk_persistencia_mask)
        layout_mask.addWidget(self.chk_tiempo_en_zona)
        self.group_mask.setLayout(layout_mask)
        scroll_layout.addWidget(self.group_mask)

        # Grupo: Trayectorias
        self.group_tray = QGroupBox("Estadísticos basados en Trayectorias")
        layout_tray = QVBoxLayout()
        self.chk_longitud = QCheckBox("Longitud media de trayectorias")
        self.chk_velocidad = QCheckBox("Velocidad media por trayectoria")
        self.chk_direccion = QCheckBox("Dirección dominante del movimiento")
        self.chk_giros = QCheckBox("Cambios de dirección (zig-zag)")
        self.chk_orbital = QCheckBox("Circularidad / Patrón orbital")
        self.chk_polarizacion = QCheckBox("Índice de polarización")
        layout_tray.addWidget(self.chk_longitud)
        layout_tray.addWidget(self.chk_velocidad)
        layout_tray.addWidget(self.chk_direccion)
        layout_tray.addWidget(self.chk_giros)
        layout_tray.addWidget(self.chk_orbital)
        layout_tray.addWidget(self.chk_polarizacion)
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
        #aqui los demas estadísticos de BBoxes

        estadisticos_mask = []
        if self.chk_densidad.isChecked():
            estadisticos_mask.append("histograma_densidad")
        #aqui los demas estadísticos de máscaras

        if not estadisticos_bbox:
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
            output_dims_path = os.path.join(subcarpetas[0], "bbox", "output_dims.json")
            with open(output_dims_path, "r") as f:
                dims = json.load(f)
            dimensiones_entrada = dims["output_dims"]
        except Exception:
            QMessageBox.critical(self, "Error", "No se pudo leer output_dims.json")
            return

        # Llamada al procesado de bbox
        # Reiniciar barras de progreso
        self.etapa_actual.setText("Iniciando postprocesado...")
        self.barra_progreso_etapas.setValue(0)
        self.barra_progreso_especifica.setValue(0)

        procesar_bbox_stats(
            carpeta_trabajo=carpeta,
            estadisticos_seleccionados=estadisticos_bbox,
            num_procesos=num_procesos,
            dimensiones_entrada=dimensiones_entrada,
            estado=estado
        )

        procesar_mask_stats(
            carpeta_trabajo=carpeta,
            estadisticos_seleccionados=estadisticos_mask,
            num_procesos=num_procesos,
            dimensiones_entrada=dimensiones_entrada,
            estado=estado
        )

    


    def volver_a_inicio(self):
        self.close()
        if self.ventana_inicio:
            self.ventana_inicio.show()
