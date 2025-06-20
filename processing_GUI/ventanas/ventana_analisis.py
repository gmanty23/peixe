from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QProgressBar, QMessageBox, QSizePolicy, QListWidget, QListWidgetItem, QSpinBox, QDoubleSpinBox
)
from PySide6.QtGui import QFont, QIcon 
from PySide6.QtCore import Qt, QTimer, QThread, Signal
import os
import json
import glob
import torch
import subprocess
import tempfile


from processing_GUI.procesamiento.analisis import generar_inputs_moment, EstadoProceso

from PySide6.QtWidgets import QMessageBox


class VentanaAnalisis(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.ventana_inicio = parent
        self.setWindowTitle("Herramienta de Análisis")
        self.setMinimumSize(600, 400)
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        titulo = QLabel("Análisis Series Temporales - MOMENT")
        titulo.setFont(QFont('Segoe UI', 18, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        layout.addWidget(titulo)

        grupo_inputs = QGroupBox("Generación Inputs Compatibles")
        layout_grupo = QVBoxLayout()

        # Selector de carpeta
        selector_layout = QHBoxLayout()
        self.lineedit_carpeta = QLineEdit()
        self.lineedit_carpeta.setPlaceholderText("Selecciona la carpeta de trabajo")
        boton_examinar = QPushButton("Examinar")
        boton_examinar.clicked.connect(lambda: self.seleccionar_carpeta(self.lineedit_carpeta))
        selector_layout.addWidget(self.lineedit_carpeta)
        selector_layout.addWidget(boton_examinar)
        layout_grupo.addLayout(selector_layout)

        # Botón generar
        self.boton_generar = QPushButton("Generar Inputs Compatibles con MOMENT")
        self.boton_generar.clicked.connect(self.generar_inputs)
        self.boton_generar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout_grupo.addWidget(self.boton_generar)

        
        # Estado y progreso
        self.etiqueta_estado_grupo = QLabel("Esperando instrucciones...")
        layout_grupo.addWidget(self.etiqueta_estado_grupo)
        
        self.barra_progreso_grupo = QProgressBar()
        layout_grupo.addWidget(self.barra_progreso_grupo)
        
        self.etiqueta_estado_grupo.setVisible(False)
        self.barra_progreso_grupo.setVisible(False)
        
        
        grupo_inputs.setLayout(layout_grupo) 
        layout.addWidget(grupo_inputs)

         # -------- NUEVO GRUPO: CLASIFICACIÓN SUPERVISADA --------
        grupo_clasif = QGroupBox("Clasificación Supervisada")
        layout_clasif = QVBoxLayout()

        # Parámetros
        params_layout = QHBoxLayout()
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 1000)
        self.epoch_spin.setValue(30)
        self.epoch_spin.setPrefix("Epochs: ")
        params_layout.addWidget(self.epoch_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(64)
        self.batch_spin.setPrefix("Batch: ")
        params_layout.addWidget(self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setPrefix("LR: ")
        params_layout.addWidget(self.lr_spin)

        layout_clasif.addLayout(params_layout)

        # Selector de canales
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QListWidget.MultiSelection)
        with open("processing_GUI/procesamiento/clasificacion_supervisada/info_canales.json") as f:
            info = json.load(f)["info_canales"]
        for key, desc in info.items():
            item = QListWidgetItem(f"{key} - {desc}")
            item.setData(Qt.UserRole, int(key.split("_")[1]))
            self.channel_list.addItem(item)
        layout_clasif.addWidget(self.channel_list)

        # Selector carpeta entrada clasificación
        input_layout = QHBoxLayout()
        self.input_line_clasif = QLineEdit()
        self.input_line_clasif.setPlaceholderText("Selecciona la base de datos de entrada")
        input_btn_clasif = QPushButton("Examinar")
        input_btn_clasif.clicked.connect(lambda: self.seleccionar_carpeta(self.input_line_clasif))
        input_layout.addWidget(self.input_line_clasif)
        input_layout.addWidget(input_btn_clasif)
        layout_clasif.addLayout(input_layout)

        # Selector carpeta salida
        output_layout = QHBoxLayout()
        self.output_line = QLineEdit()
        self.output_line.setPlaceholderText("Selecciona carpeta de salida")
        output_btn = QPushButton("Examinar")
        output_btn.clicked.connect(lambda: self.seleccionar_carpeta(self.output_line))
        output_layout.addWidget(self.output_line)
        output_layout.addWidget(output_btn)
        layout_clasif.addLayout(output_layout)

        # Botón iniciar
        self.start_btn = QPushButton("Iniciar Clasificación")
        self.start_btn.clicked.connect(self.start_classification)
        layout_clasif.addWidget(self.start_btn)

        # Estado
        self.status_label = QLabel("Esperando instrucciones para clasificación...")
        layout_clasif.addWidget(self.status_label)

        grupo_clasif.setLayout(layout_clasif)
        layout.addWidget(grupo_clasif)

        # Botón volver
        self.boton_atras = QPushButton("Atrás")
        self.boton_atras.clicked.connect(self.volver_a_inicio)
        layout.addWidget(self.boton_atras, alignment=Qt.AlignLeft)

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
    
    def seleccionar_carpeta(self, target_lineedit):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta")
        if carpeta:
            target_lineedit.setText(carpeta)

    def generar_inputs(self):
        self.etiqueta_estado_grupo.setVisible(True)
        self.barra_progreso_grupo.setVisible(True)
        self.barra_progreso_grupo.setValue(0)
        self.etiqueta_estado_grupo.setText("Iniciando generación...")
        carpeta = self.lineedit_carpeta.text()
        if not carpeta or not os.path.exists(carpeta):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", "Debes seleccionar una carpeta válida.")
            return
        
        estado = EstadoProceso()
        estado.on_etapa = self.etiqueta_estado_grupo.setText
        estado.on_total_videos = self.barra_progreso_grupo.setMaximum
        estado.on_video_progreso = self.barra_progreso_grupo.setValue
        
        # Llamada a tu función de procesado
        generar_inputs_moment(carpeta, estado)
        
        
        self.barra_progreso_grupo.setValue(100)
        self.etiqueta_estado_grupo.setText("✅ Generación completada.")

    def volver_a_inicio(self):
        self.close()
        if self.ventana_inicio:
            self.ventana_inicio.show()

    def start_classification(self):
        input_dir = self.input_line_clasif.text()
        output_dir = self.output_line.text()


        selected = [item.data(Qt.UserRole) for item in self.channel_list.selectedItems()]

        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.critical(self, "Error", "Debes seleccionar una carpeta de entrada válida.")
            return

        if not output_dir or not os.path.exists(output_dir):
            QMessageBox.critical(self, "Error", "Debes seleccionar una carpeta de salida válida.")
            return

        if len(selected) < 1:
            QMessageBox.warning(self, "Selección de canales", "Debes seleccionar al menos un canal.")
            return


        out_name = "canales" + "_" + "_".join(map(str, selected))
        final_output = os.path.join(output_dir, out_name)
        params = {
            "epochs": self.epoch_spin.value(),
            "batch_size": self.batch_spin.value(),
            "learning_rate": self.lr_spin.value()
        }

        # Inicializa choice por defecto como None
        choice = None

        # Verifica si la carpeta ya existe
        if os.path.exists(final_output):
            model_dir = os.path.join(final_output, "models")
            if os.path.exists(model_dir):
                checkpoints = glob.glob(os.path.join(model_dir, "last_model.pt"))
                best_model = os.path.join(model_dir, "best_model*.pt")
                best_loss_model = os.path.join(model_dir, "best_loss_model*.pt")
                choice = None

                if checkpoints or os.path.exists(best_model) or os.path.exists(best_loss_model):
                    msg = QMessageBox(self)
                    msg.setWindowTitle("Modelo existente detectado")
                    msg.setText("Se han encontrado pesos guardados en la carpeta seleccionada.\n¿Qué deseas hacer?")
                    last_btn = msg.addButton("Cargar último checkpoint", QMessageBox.AcceptRole)
                    best_acc_btn = msg.addButton("Cargar best model (acc)", QMessageBox.AcceptRole)
                    best_loss_btn = msg.addButton("Cargar best model (loss)", QMessageBox.AcceptRole)
                    new_btn = msg.addButton("Empezar de cero", QMessageBox.RejectRole)
                    msg.exec()

                    if msg.clickedButton() == last_btn:
                        if checkpoints:
                            choice = max(checkpoints, key=os.path.getctime)
                        else:
                            QMessageBox.warning(self, "Aviso", "No hay checkpoint disponible. Empezando de cero.")
                    elif msg.clickedButton() == best_acc_btn:
                        if os.path.exists(best_model):
                            choice = best_model
                        else:
                            QMessageBox.warning(self, "Aviso", "No hay best model. Empezando de cero.")
                    elif msg.clickedButton() == best_loss_btn:
                        if os.path.exists(best_loss_model):
                            choice = best_loss_model
                        else:
                            QMessageBox.warning(self, "Aviso", "No hay best loss model. Empezando de cero.")
                    else:
                        choice = None

        self.status_label.setText("Lanzando MOMENT en modo clasificación supervisada...\nConsulta la terminal para detalles.")

        input_dirs = {
            0: os.path.join(input_dir, "activos"),
            1: os.path.join(input_dir, "alterados"),
            2: os.path.join(input_dir, "relajados")
        }

        config = {
            "input_dirs": input_dirs,
            "channels_to_use": selected,
            "params": params,
            "output_dir": final_output,
            "checkpoint_to_load": choice
        }

        # Guarda el config en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp:
            json.dump(config, tmp, indent=2)
            tmp_path = tmp.name

        self.status_label.setText("Lanzando MOMENT en modo clasificación supervisada (subproceso)...")

        # Lanza el subproceso
        subprocess.run(["python", "processing_GUI/procesamiento/clasificacion_supervisada/launcher_train.py", tmp_path])

        self.status_label.setText("✅ Procesamiento completado.")
        print("Clasificación supervisada completada. Revisa la carpeta de salida para los resultados.")