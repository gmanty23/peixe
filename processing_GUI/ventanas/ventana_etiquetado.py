import subprocess
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QProgressBar,
                              QLabel, QSpinBox, QPushButton, QFileDialog, QDoubleSpinBox, QScrollArea, 
                              QHBoxLayout, QLineEdit, QMessageBox, QGroupBox, QCheckBox, QComboBox, QStackedWidget)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont
import os
import subprocess
import shutil
import json
from processing_GUI.procesamiento.etiquetado_morfologia import procesar_videos_con_morfologia, EstadoProceso, extraer_frames
import cv2
from processing_GUI.procesamiento.etiquetado_yolo import procesar_yolo, limpiar_bboxes_txt_con_mascara

class WorkerThreadMorfología(QThread):
    progreso = Signal(int)
    etapa = Signal(str)
    error = Signal(str)
    progreso_video_actual = Signal(int)
    total_videos_progreso = Signal(int)

    def __init__(self, video_fondo, videos_dir, output_dir, percentil, grupo_size, nucleos, pipeline_ops, bbox_recorte):
        super().__init__()
        self.video_fondo = video_fondo
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.percentil = percentil
        self.grupo_size = grupo_size
        self.nucleos = nucleos
        self.pipeline_ops = pipeline_ops
        self.usar_imagen_fondo = False
        self.ruta_imagen_fondo = None
        self.resize_enabled = False
        self.resize_dims = (1920, 1080)
        self.bbox_recorte = bbox_recorte
        self.output_dims = (1920, 1080)

    def run(self):
        estado = EstadoProceso()
        estado.on_etapa = self.etapa.emit
        estado.on_progreso = self.progreso.emit
        estado.on_error = self.error.emit
        estado.on_total_videos = self.total_videos_progreso.emit
        estado.on_video_progreso = self.progreso_video_actual.emit

        procesar_videos_con_morfologia(
            self.video_fondo,
            self.videos_dir,
            self.output_dir,
            self.percentil,
            self.grupo_size,
            self.nucleos,
            self.pipeline_ops,
            estado,
            usar_imagen_fondo=self.usar_imagen_fondo,
            ruta_imagen_fondo=self.ruta_imagen_fondo,
            resize_enabled=self.resize_enabled,
            resize_dims=self.resize_dims,
            bbox_recorte=self.bbox_recorte,
            output_dims=self.output_dims
        )




class VentanaEtiquetado(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent  # Referencia a la ventana principal
        self.setWindowTitle("Herramienta de Etiquetado")
        self.setMinimumSize(600, 800)
        self.setup_ui()
        self.setup_styles()

    def volver_a_inicio(self):
        self.barra_progreso.setValue(0)
        self.etiqueta_etapa.setText("Esperando...")
        self.close()
        if self.parent_window:
            self.parent_window.show()
    
    def setup_ui(self):
        # Widget central y layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Selector de método de etiquetado
        method_layout = QHBoxLayout()
        method_label = QLabel("Método de etiquetado:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Cutie", "Morfología", "YOLOv8"])
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        main_layout.addLayout(method_layout)

        # Contenedor de parámetros de etiquetado
        self.method_params_container = QStackedWidget()
        self.method_params_container.addWidget(self.create_cutie_params())
        self.method_params_container.addWidget(self.create_morphology_params())
        self.method_params_container.addWidget(self.create_yolo_params())
        main_layout.addWidget(self.method_params_container)

        # Progreso y etiqueta de etapa
        self.etiqueta_etapa = QLabel("Esperando...")
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, 100)
        self.etiqueta_etapa.setVisible(False)
        self.barra_progreso.setVisible(False)
        
        main_layout.addWidget(self.etiqueta_etapa)
        main_layout.addWidget(self.barra_progreso)

        self.etiqueta_videos = QLabel("Progreso global:")
        self.barra_progreso_videos = QProgressBar()
        self.barra_progreso_videos.setRange(0, 100)
        self.etiqueta_videos.setVisible(False)
        self.barra_progreso_videos.setVisible(False)
        self.etiqueta_etapa.setVisible(False)
        self.barra_progreso.setVisible(False)
        main_layout.addWidget(self.etiqueta_videos)
        main_layout.addWidget(self.barra_progreso_videos)

        # Botones 
        buttons_layout = QHBoxLayout()
        self.back_button = QPushButton("Atrás")
        self.back_button.clicked.connect(self.volver_a_inicio)
        self.generate_button = QPushButton("Iniciar Etiquetado")
        self.generate_button.clicked.connect(self.iniciar_etiquetado)
        buttons_layout.addWidget(self.back_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.generate_button)
        main_layout.addLayout(buttons_layout)

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
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a5276;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QSpinBox {
                padding: 5px;
            }
        """)
        
    def create_directory_selector(self):
        dir_layout = QHBoxLayout()
        dir_label = QLabel("Directorio de trabajo:")
        self.dir_lineedit = QLineEdit()
        self.dir_lineedit.setPlaceholderText("Selecciona un directorio...")
        dir_button = QPushButton("Examinar...")
        dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.dir_lineedit)
        dir_layout.addWidget(dir_button)
        return dir_layout

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "Seleccionar Directorio de Trabajo",
            str(Path.home())
        )
        if dir_path:
            self.dir_lineedit.setText(dir_path)
    
    def iniciar_etiquetado(self):
        method = self.method_combo.currentText()


        if method == "Cutie":
            
            self.ejecutar_cutie()
            return
        elif method == "Morfología":
            self.ejecutar_morfologia()
            return
        elif method == "YOLOv8":
            self.ejecutar_yolo()
            return


    def ejecutar_cutie(self):
        num_objects = self.num_spinbox.value()
        extraer_frames_enabled = self.extraer_imagenes_checkbox.isChecked()
        resize_enabled = self.resize_checkbox.isChecked()
        resize_dims = (self.resize_width_spinbox.value(), self.resize_height_spinbox.value())
        bbox_recorte = None
        if self.checkbox_usar_recorte.isChecked():
            bbox_texto = self.bbox_lineedit.text()
            if bbox_texto:
                try:
                    bbox_recorte = tuple(map(int, bbox_texto.strip().split(',')))
                    assert len(bbox_recorte) == 4
                except Exception:
                    QMessageBox.critical(self, "Error", "La bounding box debe tener el formato: x, y, w, h")
                    return
            else:
                QMessageBox.critical(self, "Error", "Has activado recorte pero no has introducido una bounding box")
                return
        video_path = self.video_lineedit.text()
        if not video_path:
            QMessageBox.critical(self, "Error", "Debes seleccionar un video a procesar")
            return
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        carpeta_trabajo = os.path.dirname(video_path)
        workspace = os.path.join(carpeta_trabajo, video_name, "workspace")
        os.makedirs(workspace, exist_ok=True)
        frames_dir = os.path.join(workspace, "images")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Crear el jsond e output_dims
        output_dims_path = os.path.join(workspace, "output_dims.json")
        with open(output_dims_path, 'w') as f:
            json.dump(resize_dims, f)
            
        # Inicializar estado
        self.barra_progreso.setValue(0)
        self.barra_progreso.setVisible(True)
        self.etiqueta_etapa.setVisible(True)
        self.barra_progreso_videos.setVisible(True)
        self.etiqueta_videos.setVisible(True)

        estado = EstadoProceso()
        estado.on_etapa = self.etiqueta_etapa.setText
        estado.on_progreso = self.barra_progreso.setValue
        estado.on_error = lambda msg: QMessageBox.critical(self, "Error", msg)
        estado.on_total_videos = lambda total: self.barra_progreso_videos.setMaximum(total)
        estado.on_video_progreso = self.barra_progreso_videos.setValue
        
        # Extraer frames del video
        if extraer_frames_enabled:
            estado.on_etapa("Extrayendo frames del video...")
            imagenes = extraer_frames(video_path, frames_dir, resize_enabled, resize_dims, bbox_recorte, estado)
        reset_flag = self.reset_checkbox.isChecked()
        mask_guardadas_path = os.path.join(workspace, "masks_guardadas")
        if reset_flag:
            if os.path.exists(mask_guardadas_path):
                shutil.rmtree(mask_guardadas_path)
            file_path = os.path.join(workspace, "mask_counter.txt")
            with open(file_path, "w") as f:
                f.write(str("0"))
            
        cmd = f'python interactive_demo.py --num_objects "{num_objects}" --workspace "{workspace}"'
            
        # 1. Primero intentamos ejecución directa (sin terminal nuevo)
        if self._try_direct_execution(cmd):
            return
        # 2. Si falla, probamos con terminales gráficos
        if self._try_graphical_terminals(cmd):
            return         
        # 3. Último recurso: ejecución con captura de output
        self._execute_with_output(cmd)

    def _try_direct_execution(self, cmd):
        """Intenta ejecución directa sin terminal nuevo"""
        try:
            # Ejecutar en segundo plano sin bloquear
            subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/bash',
                start_new_session=True
            )
            return True
        except Exception:
            return False

    def _try_graphical_terminals(self, cmd):
        """Intenta con varios terminales gráficos"""
        terminals = [
            ('xterm', ['xterm', '-hold', '-e', cmd]),
            ('konsole', ['konsole', '-e', 'bash', '-c', f'{cmd}; exec bash']),
            ('gnome-terminal', ['gnome-terminal', '--', 'bash', '-c', f'{cmd}; exec bash']),
            ('xfce4-terminal', ['xfce4-terminal', '-x', 'bash', '-c', f'{cmd}; exec bash']),
            ('lxterminal', ['lxterminal', '-e', 'bash', '-c', f'{cmd}; exec bash'])
        ]
        
        for name, command in terminals:
            if self._try_execute(command):
                print(f"Éxito usando terminal: {name}")
                return True
        return False

    def _try_execute(self, command):
        """Intenta ejecutar un comando y devuelve True si tiene éxito"""
        try:
            subprocess.Popen(command)
            return True
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def _execute_with_output(self, cmd):
        """Ejecuta el comando directamente y muestra el resultado"""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable='/bin/bash'
            )
            
            output = f"Comando: {cmd}\n\nSalida:\n{result.stdout}"
            if result.stderr:
                output += f"\n\nErrores:\n{result.stderr}"
                
            QMessageBox.information(self, "Resultado", output)
            
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Error al ejecutar:\n{cmd}\n\n"
                f"Código: {e.returncode}\n\n"
                f"Error:\n{e.stderr}\n\n"
                f"Salida:\n{e.stdout}"
            )
            QMessageBox.critical(self, "Error de Ejecución", error_msg)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Grave",
                f"No se pudo ejecutar el comando:\n{cmd}\n\n"
                f"Error: {str(e)}\n\n"
                "Solución recomendada:\n"
                "1. Ejecuta manualmente en una terminal:\n"
                f"{cmd}\n\n"
                "2. O instala un terminal gráfico como xterm:\n"
                "sudo apt install xterm"
            )
    
    def volver_a_inicio(self):
        self.close()
        if self.parent_window:
            self.parent_window.show()

    def create_cutie_params(self):
        """Crea el layout para los parámetros de Cutie"""
        cutie_widget = QWidget()
        cutie_layout = QVBoxLayout(cutie_widget)
        
        # Grupo de configuración de extracción
        extract_group = QGroupBox("Configuración de extracción")
        extract_layout = QVBoxLayout()
        extract_group.setLayout(extract_layout)
        
        # Checkbox para extraer imagenes o no
        self.extraer_imagenes_checkbox = QCheckBox("Extraer imágenes del video")
        self.extraer_imagenes_checkbox.setChecked(True)
        self.extraer_imagenes_checkbox.stateChanged.connect(self.actualizar_estado_dependientes)

        extract_layout.addWidget(self.extraer_imagenes_checkbox)
        
        # Seleccion de recorte
        self.checkbox_usar_recorte = QCheckBox("Aplicar recorte")
        self.checkbox_usar_recorte.setChecked(True)
        self.checkbox_usar_recorte.stateChanged.connect(self.actualizar_estado_dependientes)

        extract_layout.addWidget(self.checkbox_usar_recorte)
        
        bbox_layout = QHBoxLayout()
        bbox_label = QLabel("Bounding box (x, y, w, h):")
        self.bbox_lineedit = QLineEdit()
        self.bbox_lineedit.setPlaceholderText("Ej: 100, 200, 300, 400")
        self.bbox_lineedit.setText("550, 960, 2225, 1186")
        self.bbox_lineedit.setEnabled(True)
        
        bbox_layout.addWidget(bbox_label)
        bbox_layout.addWidget(self.bbox_lineedit)
        extract_layout.addLayout(bbox_layout)
        
        # Redimensionado de imágenes
        resize_layout = QHBoxLayout()
        self.resize_checkbox = QCheckBox("Activar redimensionado (ancho x alto)")
        self.resize_checkbox.setChecked(True)
        self.resize_checkbox.stateChanged.connect(self.actualizar_estado_dependientes)

        self.resize_width_spinbox = QSpinBox()
        self.resize_height_spinbox = QSpinBox()
        self.resize_width_spinbox.setRange(100, 10000)
        self.resize_width_spinbox.setValue(1920)
        self.resize_height_spinbox.setRange(100, 10000)
        self.resize_height_spinbox.setValue(1080)
        resize_layout.addWidget(self.resize_checkbox)
        resize_layout.addWidget(self.resize_width_spinbox)
        resize_layout.addWidget(QLabel("×"))
        resize_layout.addWidget(self.resize_height_spinbox)
        extract_layout.addLayout(resize_layout)
        
        cutie_layout.addWidget(extract_group)

        #Grupo de configuración de Cutie
        config_cutie_group = QGroupBox("Configuración de Cutie")
        config_cutie_layout = QVBoxLayout()
        config_cutie_group.setLayout(config_cutie_layout)
        
        # Sección para el reinicio del contador de máscaras guardadas
        reset_layout = QHBoxLayout()
        reset_label = QLabel("Reiniciar contador de máscaras guardadas:")
        self.reset_checkbox = QCheckBox()
        reset_layout.addWidget(reset_label)
        reset_layout.addWidget(self.reset_checkbox)
        config_cutie_layout.addLayout(reset_layout)
        
        
        # Sección para el número de objetos
        num_layout = QHBoxLayout()
        num_label = QLabel("Número de objetos:")
        self.num_spinbox = QSpinBox()
        self.num_spinbox.setRange(1, 100)
        self.num_spinbox.setValue(3)
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_spinbox)
        config_cutie_layout.addLayout(num_layout)
        
        cutie_layout.addWidget(config_cutie_group)

        
        # Sección para el video a procesar
        video_layout = QHBoxLayout()
        video_label = QLabel("Video a procesar:")
        self.video_lineedit = QLineEdit()
        self.video_lineedit.setPlaceholderText("Selecciona un video...")
        video_button = QPushButton("Examinar...")
        video_button.clicked.connect(self.select_video)
        video_layout.addWidget(video_label)
        video_layout.addWidget(self.video_lineedit)
        video_layout.addWidget(video_button)
        cutie_layout.addLayout(video_layout)
        
        self.actualizar_estado_dependientes()

        


        return cutie_widget   

    def add_morphology_operation(self):
        opertaion_layout = QHBoxLayout()

        # Combo de tipo de operación
        op_combo = QComboBox()
        op_combo.addItems(["Erosión", "Dilatación", "Apertura", "Cierre"])

        # Tamaños del kernel
        spin_x = QSpinBox()
        spin_y = QSpinBox()
        spin_x.setRange(1, 100)
        spin_y.setRange(1, 100)
        spin_x.setValue(3)
        spin_y.setValue(3)

        # Botón de eliminar operación
        remove_button = QPushButton("❌")
        remove_button.setFixedWidth(30)

        # Wdget contenedor para poder eliinarlo visualmente
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.addWidget(op_combo)
        container_layout.addWidget(spin_x)
        container_layout.addWidget(spin_y)
        container_layout.addWidget(remove_button)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Lógica de eliminación
        def remove_operation():
            self.morph_entries = [e for e in self.morph_entries if e["widget"] != container]
            self.morph_ops_layout.removeWidget(container)
            container.deleteLater()

        remove_button.clicked.connect(remove_operation)

        # Guardar la operación
        self.morph_entries.append({
            "widget": container,
            "op": op_combo,
            "kernel_x": spin_x,
            "kernel_y": spin_y
        })
        self.morph_ops_layout.addWidget(container)

    def select_background_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen de fondo",
            str(Path.home()),
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.fondo_lineedit.setText(file_path)

    def create_morphology_params(self):
        """Crea el layout para los parámetros de Morfología"""
        morphology_widget = QWidget()
        morphology_layout = QVBoxLayout(morphology_widget)

        # Grupo de configuración de extracción
        extract_group = QGroupBox("Configuración de extracción")
        extract_layout = QVBoxLayout()
        extract_group.setLayout(extract_layout)
        
        # Selección del tamaño de salida
        output_resize_layout = QHBoxLayout()
        output_resize_label = QLabel("Tamaño de salida (ancho x alto):")
        self.output_width_spinbox = QSpinBox()
        self.output_height_spinbox = QSpinBox()
        self.output_width_spinbox.setRange(50, 4000)
        self.output_height_spinbox.setRange(50, 4000)
        self.output_width_spinbox.setValue(1920)
        self.output_height_spinbox.setValue(1080)
        output_resize_layout.addWidget(output_resize_label)
        output_resize_layout.addWidget(self.output_width_spinbox)
        output_resize_layout.addWidget(QLabel("×"))
        output_resize_layout.addWidget(self.output_height_spinbox)
        extract_layout.addLayout(output_resize_layout)

        # Seleccion de recorte
        self.checkbox_usar_recorte = QCheckBox("Aplicar recorte")
        self.checkbox_usar_recorte.setChecked(True)
        extract_layout.addWidget(self.checkbox_usar_recorte)

        bbox_layout = QHBoxLayout()
        bbox_label = QLabel("Bounding box (x, y, w, h):")
        self.bbox_lineedit = QLineEdit()
        self.bbox_lineedit.setPlaceholderText("Ej: 100, 200, 300, 400")
        self.bbox_lineedit.setText("550, 960, 2225, 1186")
        self.bbox_lineedit.setEnabled(True)

        bbox_layout.addWidget(bbox_label)
        bbox_layout.addWidget(self.bbox_lineedit)
        extract_layout.addLayout(bbox_layout)

        # Selección de núcleos
        cores_layout = QHBoxLayout()
        cores_label = QLabel("Número de núcleos:")
        self.cores_spinbox = QSpinBox()
        self.cores_spinbox.setRange(1, os.cpu_count())
        self.cores_spinbox.setValue(7)
        cores_layout.addWidget(cores_label)
        cores_layout.addWidget(self.cores_spinbox)
        extract_layout.addLayout(cores_layout)

        # Redimensionado de imágenes
        resize_layout = QHBoxLayout()

        self.resize_checkbox = QCheckBox("Activar redimensionado")
        self.resize_checkbox.setChecked(True)
        self.resize_width_spinbox = QSpinBox()
        self.resize_height_spinbox = QSpinBox()

        self.resize_width_spinbox.setRange(100, 10000)
        self.resize_width_spinbox.setValue(1920)
        self.resize_height_spinbox.setRange(100, 10000)
        self.resize_height_spinbox.setValue(1080)

        resize_layout.addWidget(self.resize_checkbox)
        resize_layout.addWidget(QLabel("Ancho:"))
        resize_layout.addWidget(self.resize_width_spinbox)
        resize_layout.addWidget(QLabel("Alto:"))
        resize_layout.addWidget(self.resize_height_spinbox)

        extract_layout.addLayout(resize_layout)
        
        morphology_layout.addWidget(extract_group)

        # Selección de los parámetros de cálculo de fondo
        fondo_group = QGroupBox("Cálculo de fondo")
        fondo_layout = QVBoxLayout(fondo_group)

        perc_layout = QHBoxLayout()
        perc_label = QLabel("Percentil de mediana:")
        self.percentile_spinbox = QDoubleSpinBox()
        self.percentile_spinbox.setRange(0, 100)
        self.percentile_spinbox.setValue(65)
        self.percentile_spinbox.setSingleStep(1.0)
        perc_layout.addWidget(perc_label)
        perc_layout.addWidget(self.percentile_spinbox)
        fondo_layout.addLayout(perc_layout)

        mediana_group_layout = QHBoxLayout()
        mediana_group_label = QLabel("Tamaño de grupo para fondo:")
        self.mediana_group_size_spinbox = QSpinBox()
        self.mediana_group_size_spinbox.setRange(1, 500)
        self.mediana_group_size_spinbox.setValue(60)
        mediana_group_layout.addWidget(mediana_group_label)
        mediana_group_layout.addWidget(self.mediana_group_size_spinbox)
        fondo_layout.addLayout(mediana_group_layout)

        mediana_img_layout = QHBoxLayout()
        mediana_img_label = QLabel("Imagen de fondo (opcional):")
        self.fondo_lineedit = QLineEdit()
        self.fondo_lineedit.setPlaceholderText("Selecciona una imagen...")
        mediana_img_button = QPushButton("Examinar...")
        mediana_img_button.clicked.connect(self.select_background_image)
        mediana_img_layout.addWidget(mediana_img_label)
        mediana_img_layout.addWidget(self.fondo_lineedit)
        mediana_img_layout.addWidget(mediana_img_button)
        fondo_layout.addLayout(mediana_img_layout)

        fondo_group.setLayout(fondo_layout)
        morphology_layout.addWidget(fondo_group)

        # Selección de las caracterización del umbral
        umbral_layout = QVBoxLayout()

        self.umbral_combo = QComboBox()
        self.umbral_combo.addItems(["Global", "Adaptativo"])
        self.umbral_combo.currentTextChanged.connect(self.actualizar_parametros_umbral)
        umbral_layout.addWidget(QLabel("Tipo de umbral:"))
        umbral_layout.addWidget(self.umbral_combo)

        # Parámetros para el umbral global y adaptativo
        self.umbral_global_widget = QWidget()
        umbral_global_layout = QHBoxLayout(self.umbral_global_widget)
        umbral_label = QLabel("Valor de umbral:")
        self.umbral_spinbox = QDoubleSpinBox()
        self.umbral_spinbox.setRange(0, 255)
        self.umbral_spinbox.setValue(15)
        self.umbral_spinbox.setSingleStep(1)
        umbral_global_layout.addWidget(umbral_label)
        umbral_global_layout.addWidget(self.umbral_spinbox)

        self.umbral_adaptativo_widget = QWidget()
        umbral_adaptativo_layout = QHBoxLayout(self.umbral_adaptativo_widget)
        self.block_size_spinbox = QSpinBox()
        self.block_size_spinbox.setRange(3, 999)
        self.block_size_spinbox.setSingleStep(2)
        self.block_size_spinbox.setValue(11)
        self.C_spinbox = QSpinBox()
        self.C_spinbox.setRange(0, 50)
        self.C_spinbox.setValue(2)
        # Selector de método adaptativo
        self.adaptive_method_combo = QComboBox()
        self.adaptive_method_combo.addItems(["Media", "Gaussiana"])
        umbral_adaptativo_layout.addWidget(QLabel("Método:"))
        umbral_adaptativo_layout.addWidget(self.adaptive_method_combo)
        umbral_adaptativo_layout.addWidget(QLabel("Block size:"))
        umbral_adaptativo_layout.addWidget(self.block_size_spinbox)
        umbral_adaptativo_layout.addWidget(QLabel("C:"))
        umbral_adaptativo_layout.addWidget(self.C_spinbox)

        umbral_layout.addWidget(self.umbral_global_widget)
        umbral_layout.addWidget(self.umbral_adaptativo_widget)

        fondo_layout.addLayout(umbral_layout)
        self.actualizar_parametros_umbral("Global")

        # Creación pipeline operaciones morfológicas
        morph_group = QGroupBox("Operaciones morfológicas")
        morph_layout = QVBoxLayout()

        # Scroll para operaciones morfológicas
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_cotent = QWidget()
        self.morph_ops_layout = QVBoxLayout(scroll_cotent)
        self.morph_entries = []  # Lista para guardar las entradas de operaciones morfológicas

        scroll_area.setWidget(scroll_cotent)
        morph_layout.addWidget(scroll_area)

        # Botón para añadir operaciones morfológicas
        add_button = QPushButton("➕ Añadir operación morfológica")
        add_button.clicked.connect(self.add_morphology_operation)
        morph_layout.addWidget(add_button)

        morph_group.setLayout(morph_layout)
        morphology_layout.addWidget(morph_group)

        # Sección para el directorio de salida
        morphology_layout.addLayout(self.create_directory_selector())


        return morphology_widget  

    def ejecutar_morfologia(self):
        workspace = self.dir_lineedit.text()
        if not workspace:
            QMessageBox.critical(self, "Error", "Debes seleccionar un directorio de trabajo")
            return

        fondo_manual = self.fondo_lineedit.text()
        if fondo_manual and os.path.exists(fondo_manual):
            usar_imagen_fondo = True
            video_fondo = None
        else:
            usar_imagen_fondo = False
            video_fondo = QFileDialog.getOpenFileName(self, "Selecciona el video para fondo", str(Path.home()), "Videos (*.mp4 *.avi)")[0]
            if not video_fondo:
                QMessageBox.critical(self, "Error", "Debes seleccionar un video o adjuntar una imagen de fondo")
                return

        output_dir = workspace
        videos_dir = workspace
        percentil = self.percentile_spinbox.value()
        grupo_size = self.mediana_group_size_spinbox.value()
        nucleos = self.cores_spinbox.value()
        output_dims = (self.output_width_spinbox.value(),self.output_height_spinbox.value())

        bbox_recorte = None
        if self.checkbox_usar_recorte.isChecked():
            bbox_texto = self.bbox_lineedit.text()
            if bbox_texto:
                try:
                    bbox_recorte = tuple(map(int, bbox_texto.strip().split(',')))
                    assert len(bbox_recorte) == 4
                except Exception:
                    QMessageBox.critical(self, "Error", "La bounding box debe tener el formato: x, y, w, h")
                    return
            else:
                QMessageBox.critical(self, "Error", "Has activado recorte pero no has introducido una bounding box")
                return

        print(">>> ROI devuelta por seleccionar_bbox:", bbox_recorte)

        pipeline_ops = []
        resize_enabled = self.resize_checkbox.isChecked()
        resize_dims = (self.resize_width_spinbox.value(), self.resize_height_spinbox.value())
        # Insertar umbral como paso inicial del pipeline
        tipo_umbral = self.umbral_combo.currentText()
        adaptive_method_text = self.adaptive_method_combo.currentText()
        adaptive_method = "mean" if adaptive_method_text == "Media" else "gaussian"
        if tipo_umbral == "Global":
            pipeline_ops.append({
                "op": "umbral",
                "valor": self.umbral_spinbox.value()
            })
        else:
            pipeline_ops.append({
                "op": "adaptativo",
                "block_size": self.block_size_spinbox.value(),
                "C": self.C_spinbox.value(),
                "method": adaptive_method
            })
        # Añadir operaciones morfológicas
        for entry in self.morph_entries:
            op = entry["op"].currentText().lower()
            kx = entry["kernel_x"].value()
            ky = entry["kernel_y"].value()
            pipeline_ops.append({"op": op, "kernel": [kx, ky]})

        self.worker = WorkerThreadMorfología(
            video_fondo or "",
            videos_dir,
            output_dir,
            percentil,
            grupo_size,
            nucleos,
            pipeline_ops,
            bbox_recorte
        )
        self.worker.usar_imagen_fondo = usar_imagen_fondo
        self.worker.ruta_imagen_fondo = fondo_manual if usar_imagen_fondo else None
        self.worker.resize_enabled = resize_enabled
        self.worker.resize_dims = resize_dims
        self.worker.output_dims = output_dims

        self.worker.progreso.connect(self.barra_progreso.setValue)
        self.worker.etapa.connect(self.etiqueta_etapa.setText)
        self.worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
        self.worker.progreso_video_actual.connect(self.barra_progreso_videos.setValue)
        self.worker.total_videos_progreso.connect(self.barra_progreso_videos.setMaximum)

        self.barra_progreso.setValue(0)
        self.barra_progreso_videos.setValue(0)
        self.etiqueta_etapa.setText("Iniciando...")
        self.etiqueta_videos.setVisible(True)
        self.barra_progreso_videos.setVisible(True)
        self.barra_progreso.setVisible(True)
        self.etiqueta_etapa.setVisible(True)

        self.worker.start()
        self.worker.finished.connect(self.resetear_barras_progreso)

       
    def create_yolo_params(self):
        """Crea el layout para los parámetros de YOLOv8"""
        yolo_widget = QWidget()
        yolo_layout = QVBoxLayout(yolo_widget)
        
        # Grupo de configuración de extracción
        extract_group = QGroupBox("Configuración de extracción")
        extract_layout = QVBoxLayout()
        extract_group.setLayout(extract_layout)

        # Selección del tamaño de salida
        output_resize_layout = QHBoxLayout()
        output_resize_label = QLabel("Tamaño de salida (ancho x alto):")
        self.yolo_output_width_spinbox = QSpinBox()
        self.yolo_output_height_spinbox = QSpinBox()
        self.yolo_output_width_spinbox.setRange(50, 4000)
        self.yolo_output_height_spinbox.setRange(50, 4000)
        self.yolo_output_width_spinbox.setValue(1920)
        self.yolo_output_height_spinbox.setValue(1080)
        output_resize_layout.addWidget(output_resize_label)
        output_resize_layout.addWidget(self.yolo_output_width_spinbox)
        output_resize_layout.addWidget(QLabel("×"))
        output_resize_layout.addWidget(self.yolo_output_height_spinbox)
        extract_layout.addLayout(output_resize_layout)

        # Recorte manual
        self.yolo_checkbox_usar_recorte = QCheckBox("Aplicar recorte")
        self.yolo_checkbox_usar_recorte.setChecked(True)
        extract_layout.addWidget(self.yolo_checkbox_usar_recorte)

        yolo_bbox_layout = QHBoxLayout()
        yolo_bbox_label = QLabel("Bounding box (x, y, w, h):")
        self.yolo_bbox_lineedit = QLineEdit()
        self.yolo_bbox_lineedit.setPlaceholderText("Ej: 100, 200, 300, 400")
        self.yolo_bbox_lineedit.setText("550, 960, 2225, 1186")
        yolo_bbox_layout.addWidget(yolo_bbox_label)
        yolo_bbox_layout.addWidget(self.yolo_bbox_lineedit)
        extract_layout.addLayout(yolo_bbox_layout)
        
        yolo_layout.addWidget(extract_group)
        
        #Grupo de configuración de YOLOv8
        config_yolo_group = QGroupBox("Configuración de YOLOv8")
        config_yolo_layout = QVBoxLayout()
        config_yolo_group.setLayout(config_yolo_layout)

        # Tamaño de imagen
        resize_layout = QHBoxLayout()
        resize_label = QLabel("Tamaño (imgsz):")
        self.yolo_resize_combo = QComboBox()
        self.yolo_resize_combo.addItems(["640", "1024"])
        resize_layout.addWidget(resize_label)
        resize_layout.addWidget(self.yolo_resize_combo)
        config_yolo_layout.addLayout(resize_layout)
        
        yolo_layout.addWidget(config_yolo_group)

        # Directorio de salida
        yolo_layout.addLayout(self.create_directory_selector())

        return yolo_widget

    def ejecutar_yolo(self):


        videos_dir = self.dir_lineedit.text()
        if not os.path.exists(videos_dir):
            QMessageBox.critical(self, "Error", "Debes seleccionar un directorio válido con vídeos.")
            return

        imgsz = int(self.yolo_resize_combo.currentText())
        if imgsz == 640:
            model_path = "processing_GUI/procesamiento/yolo_models/best_640.pt"
        elif imgsz == 1024:
            model_path = "processing_GUI/procesamiento/yolo_models/best_1024.pt"
        else:
            QMessageBox.critical(self, "Error", "Tamaño no válido.")
            return

        output_dims = (self.yolo_output_width_spinbox.value(),self.yolo_output_height_spinbox.value())

        # Seleccionar recorte si procede
        bbox_recorte = None
        if self.yolo_checkbox_usar_recorte.isChecked():
            bbox_texto = self.yolo_bbox_lineedit.text()
            if bbox_texto:
                try:
                    bbox_recorte = tuple(map(int, bbox_texto.strip().split(',')))
                    assert len(bbox_recorte) == 4
                except Exception:
                    QMessageBox.critical(self, "Error", "La bounding box debe tener el formato: x, y, w, h")
                    return
            else:
                QMessageBox.critical(self, "Error", "Has activado recorte pero no has introducido una bounding box")
                return



        # Inicializar estado
        self.barra_progreso.setValue(0)
        self.barra_progreso.setVisible(True)
        self.etiqueta_etapa.setVisible(True)
        self.barra_progreso_videos.setVisible(True)
        self.etiqueta_videos.setVisible(True)

        estado = EstadoProceso()
        estado.on_etapa = self.etiqueta_etapa.setText
        estado.on_progreso = self.barra_progreso.setValue
        estado.on_error = lambda msg: QMessageBox.critical(self, "Error", msg)
        estado.on_total_videos = lambda total: self.barra_progreso_videos.setMaximum(total)
        estado.on_video_progreso = self.barra_progreso_videos.setValue

        # Listar vídeos
        videos = [f for f in os.listdir(videos_dir) if f.endswith(('.avi', '.mp4'))]
        total = len(videos)
        estado.emitir_total_videos(total)

        for i, video_file in enumerate(videos):
            estado.emitir_video_progreso(i)

            nombre_base = Path(video_file).stem
            ruta_video = os.path.join(videos_dir, video_file)
            carpeta_video = os.path.join(videos_dir, nombre_base)
            os.makedirs(carpeta_video, exist_ok=True)
            salida_video = os.path.join(carpeta_video, "bbox")
            os.makedirs(salida_video, exist_ok=True)

            output_dims_info = {
                "output_dims": [output_dims[0], output_dims[1]]
            }
            with open(os.path.join(salida_video, "output_dims.json"), "w") as f:
                json.dump(output_dims_info, f, indent=4)

            if bbox_recorte:
                # Obtener dimensiones reales del vídeo
                cap = cv2.VideoCapture(ruta_video)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    QMessageBox.critical(self, "Error", f"No se pudo leer el vídeo {ruta_video} para calcular tamaño.")
                    continue
                h_img, w_img = frame.shape[:2]

                x, y, w, h = bbox_recorte
                recorte_info = {
                    "left": x,
                    "top": y,
                    "right": w_img - (x + w),
                    "bottom": h_img - (y + h),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "shape": [h_img, w_img]
                }
                with open(os.path.join(salida_video, "recorte_yolo.json"), "w") as f:
                    json.dump(recorte_info, f, indent=4)


            # Preprocesado
            imagenes_path, recorte_margenes, padding_info = procesar_yolo(ruta_video, salida_video, imgsz, bbox_recorte, estado, output_dims = output_dims)
            if imagenes_path is None:
                continue

            # Ejecutar comando YOLO
            cmd = f"yolo task=detect mode=predict model='{model_path}' source='{imagenes_path}' imgsz={imgsz} name=etiquetado_yolo project='{salida_video}' save_txt=True"

            self.etiqueta_etapa.setText("YOLO en marcha. Revisa la consola para ver el progreso...")
            self.barra_progreso.setRange(0, 0)  # barra indeterminada

            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                QMessageBox.critical(self, "Error en YOLO", f"Ocurrió un error al ejecutar YOLO:\n{e}")
                continue  # saltar al siguiente vídeo

            # Ruta de la carpeta temporal que crea YOLO dentro de bbox_yolo/
            etiquetado_yolo_dir = os.path.join(salida_video, "etiquetado_yolo")
            labels_dir_origen = os.path.join(etiquetado_yolo_dir, "labels")
            labels_dir_destino = os.path.join(salida_video, "labels")

            # Si existen labels, los movemos
            if os.path.exists(labels_dir_origen):
                if os.path.exists(labels_dir_destino):
                    shutil.rmtree(labels_dir_destino)
                shutil.move(labels_dir_origen, labels_dir_destino)

            # Eliminar toda la carpeta intermedia
            if os.path.exists(etiquetado_yolo_dir):
                shutil.rmtree(etiquetado_yolo_dir)
                
            # Normalizar coordenadas YOLO reescalando a output_dims
            labels_dir = os.path.join(salida_video, "labels")
            for archivo in os.listdir(labels_dir):
                if not archivo.endswith(".txt"):
                    continue
                ruta_txt = os.path.join(labels_dir, archivo)
                with open(ruta_txt, "r") as f:
                    lineas = f.readlines()

                abs_lines = []

                # Calcula el tamaño real de la imagen sin padding
                frame_w = imgsz - padding_info["left"] - padding_info["right"]
                frame_h = imgsz - padding_info["top"] - padding_info["bottom"]

                # Calcula los factores de escalado desde esa imagen directamente a output_dims
                fx = output_dims[0] / frame_w
                fy = output_dims[1] / frame_h

                for linea in lineas:
                    clase, x_rel, y_rel, w_rel, h_rel = map(float, linea.strip().split())

                    # Coordenadas absolutas en imagen 1024×1024 (con padding)
                    x_abs = x_rel * imgsz
                    y_abs = y_rel * imgsz
                    w_abs = w_rel * imgsz
                    h_abs = h_rel * imgsz

                    # Eliminar padding → imagen sin bordes negros
                    x_abs -= padding_info["left"]
                    y_abs -= padding_info["top"]

                    # Convertir a XYXY en imagen sin padding
                    x1 = x_abs - w_abs / 2
                    y1 = y_abs - h_abs / 2
                    x2 = x_abs + w_abs / 2
                    y2 = y_abs + h_abs / 2

                    # Escalar a output_dims directamente
                    x1 *= fx
                    x2 *= fx
                    y1 *= fy
                    y2 *= fy

                    abs_lines.append(f"{int(clase)} {int(x1)} {int(y1)} {int(x2)} {int(y2)}")

                with open(ruta_txt, "w") as f_abs:
                    f_abs.write("\n".join(abs_lines))


            # Eliminar las bbox fuera de la pecera
            limpiar_bboxes_txt_con_mascara(labels_dir, estado)


            # Finalizar este vídeo
            self.barra_progreso.setRange(0, 100)
            self.barra_progreso.setValue(100)

        estado.emitir_etapa("Detección completada.")
        self.barra_progreso_videos.setValue(total)


        self.resetear_barras_progreso()

    def on_method_changed(self, metodo):
        self.method_params_container.setCurrentIndex(metodo)
        if metodo == 0:
            self.barra_progreso.hide()
            self.etiqueta_etapa.hide()
        elif metodo == 1:
            self.barra_progreso.hide()
            self.etiqueta_etapa.hide()
        elif metodo == 2:
            self.barra_progreso.hide()
            self.etiqueta_etapa.hide()

    def resetear_barras_progreso(self):
        self.barra_progreso.setValue(0)
        self.barra_progreso_videos.setValue(0)
        self.etiqueta_etapa.setText("Esperando...")
        self.etiqueta_videos.setVisible(False)
        self.etiqueta_etapa.setVisible(False)
        self.barra_progreso_videos.setVisible(False)
        self.barra_progreso.setVisible(False)
            
    def actualizar_parametros_umbral(self,tipo):
        if tipo == "Global":
            self.umbral_global_widget.show()
            self.umbral_adaptativo_widget.hide()
        elif tipo == "Adaptativo":
            self.umbral_global_widget.hide()
            self.umbral_adaptativo_widget.show()
            
    def seleccionar_bbox(self, frame, titulo="Seleccionar ROI"):
        from processing_GUI.ventanas.ventana_seleccion_ROI import VentanaSeleccionROI
        from PySide6.QtWidgets import QDialog

        ventana_roi = VentanaSeleccionROI(frame, titulo)
        if ventana_roi.exec() == QDialog.Accepted and ventana_roi.roi is not None:
            return ventana_roi.roi
        else:
            return None
        
    def select_video(self):
        """Abre un diálogo para seleccionar un video y actualiza el campo de texto"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar video",
            str(Path.home()),
            "Videos (*.mp4 *.avi *.mov)"
        )
        if file_path:
            self.video_lineedit.setText(file_path)
            return True
        else:
            QMessageBox.warning(self, "Selección de Video", "No se seleccionó ningún video.")
            return False
        
    def actualizar_estado_dependientes(self):
        # print("Entro aqui")
        extraer = self.extraer_imagenes_checkbox.isChecked()
        self.checkbox_usar_recorte.setEnabled(extraer)
        usar_recorte = extraer and self.checkbox_usar_recorte.isChecked()
        self.bbox_lineedit.setEnabled(usar_recorte)

        self.resize_checkbox.setEnabled(extraer)
        activar_resize = extraer and self.resize_checkbox.isChecked()
        self.resize_width_spinbox.setEnabled(activar_resize)
        self.resize_height_spinbox.setEnabled(activar_resize)

        