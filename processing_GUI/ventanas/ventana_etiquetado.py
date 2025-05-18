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
from processing_GUI.procesamiento.etiquetado_morfologia import procesar_videos_con_morfologia, EstadoProceso

class WorkerThreadMorfología(QThread):
    progreso = Signal(int)
    etapa = Signal(str)
    error = Signal(str)
    progreso_video_actual = Signal(int)
    total_videos_progreso = Signal(int)

    def __init__(self, video_fondo, videos_dir, output_dir, percentil, grupo_size, umbral, nucleos, pipeline_ops):
        super().__init__()
        self.video_fondo = video_fondo
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.percentil = percentil
        self.grupo_size = grupo_size
        self.umbral = umbral
        self.nucleos = nucleos
        self.pipeline_ops = pipeline_ops
        self.usar_imagen_fondo = False
        self.ruta_imagen_fondo = None
        self.resize_enabled = False
        self.resize_dims = (1920, 1080)

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
            self.umbral,
            self.nucleos,
            self.pipeline_ops,
            estado,
            usar_imagen_fondo=self.usar_imagen_fondo,
            ruta_imagen_fondo=self.ruta_imagen_fondo,
            resize_enabled=self.resize_enabled,
            resize_dims=self.resize_dims
        )




class VentanaEtiquetado(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent  # Referencia a la ventana principal
        self.setWindowTitle("Herramienta de Etiquetado")
        self.setMinimumSize(500, 250)
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
        workspace = self.dir_lineedit.text()

        if not workspace:
            QMessageBox.critical(self, "Error", "Debes seleccionar un directorio de trabajo")
            return

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
        workspace = self.dir_lineedit.text()
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

        # Sección para el reinicio del contador de máscaras guardadas
        reset_layout = QHBoxLayout()
        reset_label = QLabel("Reiniciar contador de máscaras guardadas:")
        self.reset_checkbox = QCheckBox()
        reset_layout.addWidget(reset_label)
        reset_layout.addWidget(self.reset_checkbox)
        cutie_layout.addLayout(reset_layout)
        
        
        # Sección para el número de objetos
        num_layout = QHBoxLayout()
        num_label = QLabel("Número de objetos:")
        self.num_spinbox = QSpinBox()
        self.num_spinbox.setRange(1, 100)
        self.num_spinbox.setValue(3)
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_spinbox)
        cutie_layout.addLayout(num_layout)
        
        # Sección para el directorio
        cutie_layout.addLayout(self.create_directory_selector())

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

        # Redimensionado de imágenes
        resize_group = QGroupBox("Redimensionar imágenes (opcional)")
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

        resize_group.setLayout(resize_layout)
        morphology_layout.addWidget(resize_group)

        # Selección de núcleos
        cores_layout = QHBoxLayout()
        cores_label = QLabel("Número de núcleos:")
        self.cores_spinbox = QSpinBox()
        self.cores_spinbox.setRange(1, os.cpu_count())
        self.cores_spinbox.setValue(7)
        cores_layout.addWidget(cores_label)
        cores_layout.addWidget(self.cores_spinbox)
        morphology_layout.addLayout(cores_layout)

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

        # Umbral de diferencia
        umbral_layout = QHBoxLayout()
        umbral_label = QLabel("Umbral de diferencia:")
        self.umbral_spinbox = QDoubleSpinBox()
        self.umbral_spinbox.setRange(0, 255)
        self.umbral_spinbox.setValue(15)
        self.umbral_spinbox.setSingleStep(1)
        umbral_layout.addWidget(umbral_label)
        umbral_layout.addWidget(self.umbral_spinbox)
        morphology_layout.addLayout(umbral_layout)

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
        umbral = self.umbral_spinbox.value()
        nucleos = self.cores_spinbox.value()

        pipeline_ops = []
        resize_enabled = self.resize_checkbox.isChecked()
        resize_dims = (self.resize_width_spinbox.value(), self.resize_height_spinbox.value())
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
            umbral,
            nucleos,
            pipeline_ops
        )
        self.worker.usar_imagen_fondo = usar_imagen_fondo
        self.worker.ruta_imagen_fondo = fondo_manual if usar_imagen_fondo else None
        self.worker.resize_enabled = resize_enabled
        self.worker.resize_dims = resize_dims

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
        

        return yolo_widget

    def ejecutar_yolo(self):
        QMessageBox.information(self, "YOLOv8", "Este método aún no está implementado.")
        return

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
            
            
    