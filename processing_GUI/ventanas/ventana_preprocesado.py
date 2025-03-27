from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QCheckBox, QLineEdit, QHBoxLayout, QProgressBar, QGridLayout
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QImage, QPixmap
from processing_GUI.procesamiento.preprocesado import  extraer_imagenes, redimensionar_imagenes, atenuar_fondo_imagenes
import debugpy
import cv2

class WorkerThread(QThread):
    # Señales para el feedback del progreso del procesado
    etapa_actual_signal = Signal(str) # Nombre de la etapa actual
    progreso_general_signal = Signal(int) # Progreso general (0-100)
    progreso_especifico_signal = Signal(int) # Progreso de la etapa actual (0-100)

    def __init__(self, video_path, output_path,redimensionar_flag, width, height, atenuar_fondo_flag, sizeGrupo, factor_at, umbral_dif, apertura_flag, cierre_flag, apertura_kernel_size, cierre_kernel_size):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.cancelar_flag = False
        self.redimensionar_flag = redimensionar_flag
        self.atenuar_fondo_flag = atenuar_fondo_flag
        self.sizeGrupo = sizeGrupo
        self.factor_at = factor_at
        self.umbral_dif = umbral_dif
        self.apertura_flag = apertura_flag
        self.cierre_flag = cierre_flag
        self.apertura_kernel_size = apertura_kernel_size
        self.cierre_kernel_size = cierre_kernel_size


    def run(self):
        print("Iniciando preprocesado...")
        
        try:
            # Etapa 1: Extraer imágenes del video 
            self.etapa_actual_signal.emit("Extrayendo imágenes del video...")
            if self.cancelar_flag:
                self.etapa_actual_signal.emit("Preprocesado cancelado.")
                return
            if self.video_path is None:
                    self.etapa_actual_signal.emit("Error al extraer las imágenes.")
                    return
            images_path = extraer_imagenes(self.video_path, self.output_path, progress_callback=lambda p: self.progreso_especifico_signal.emit(p))
            if images_path is None:
                    self.etapa_actual_signal.emit("Error al extraer las imágenes.")
                    return
            self.progreso_general_signal.emit(33)
            
            # Paso 2: Redimensionar (si está habilitado)
            if self.redimensionar_flag:
                if self.cancelar_flag:
                    self.etapa_actual_signal.emit("Preprocesado cancelado.")
                    return
                if images_path is None:
                    self.etapa_actual_signal.emit("Error en el directorio de las imágenes.")
                    return
                self.etapa_actual_signal.emit("Reduciendo la resolución de las imágenes...")
                images_path = redimensionar_imagenes(images_path, self.output_path,self.width, self.height, progress_callback=lambda p: self.progreso_especifico_signal.emit(p))  
            self.progreso_general_signal.emit(66)

            # Paso 3: Atenuación de fondo
            if self.atenuar_fondo_flag: 
                if self.cancelar_flag:
                    self.etapa_actual_signal.emit("Preprocesado cancelado.")
                    return
                if images_path is None:
                    self.etapa_actual_signal.emit("Error en el directoro de las imágenes")
                self.etapa_actual_signal.emit("Iniciando la atenuación del fondo...")
                debugpy.breakpoint()
                atenuar_fondo_imagenes(images_path, self.output_path, self.sizeGrupo, self.factor_at, self.umbral_dif, self.apertura_flag, self.cierre_flag, self.apertura_kernel_size, self.cierre_kernel_size, progress_callback=lambda p: self.progreso_especifico_signal.emit(p))
            self.progreso_general_signal.emit(100)
            


            self.etapa_actual_signal.emit(f"Proceso finalizado.")
            print("Preprocesado finalizado.")
            
        except Exception as e:
            print(f"Error en el hilo de trabajo: {e}")
        
    def cancelar(self):
        self.cancelar_flag = True


class VentanaPreprocesado(QWidget):
    
    def __init__(self):
        super().__init__()
        
        # 
        self.settings = QSettings("preprocesado", "preprocesado")
        
        # Título de la ventana
        self.setWindowTitle("Ventana de preprocesado")
        self.setGeometry(100*2, 100*2, 400*2, 400*2)
        
        # Etiqueta para mostrar el nombre del archivo seleccionado
        self.label_video_input = QLabel("No se ha seleccionado ningún archivo", self)
        self.label_video_input.setAlignment(Qt.AlignCenter)
        
        # Etiqueta para mostrar la carpeta de salida seleccionada
        self.label_output = QLabel("No se ha seleccionado ninguna carpeta de salida.", self)
        self.label_output.setAlignment(Qt.AlignCenter)
        
        # Botón para seleccionar el archivo
        self.boton_seleccionar_video = QPushButton("Seleccionar archivo", self)
        self.boton_seleccionar_video.clicked.connect(self.seleccionar_video)
        
        # Botón para seleccionar la carpeta de salida
        self.boton_seleccionar_output = QPushButton("Seleccionar Carpeta de Salida", self)
        self.boton_seleccionar_output.clicked.connect(self.seleccionar_output)

        # CheckBox para la opción de redimensionar y los inputs para la resolución deseada (FullHD por defecto)
        self.checkbox_resolucion = QCheckBox("Redimensionar ", self)
        self.checkbox_resolucion.setChecked(True)  # Marcar por defecto
        self.checkbox_resolucion.stateChanged.connect(self.toggle_input_resolucion)
        
        self.resolucion_layout = QHBoxLayout()
        self.input_ancho = QLineEdit(self)
        self.input_ancho.setPlaceholderText("Ancho")
        self.input_ancho.setText("1920")
        self.input_alto = QLineEdit(self)
        self.input_alto.setPlaceholderText("Alto")
        self.input_alto.setText("1080")
        self.resolucion_layout.addWidget(self.input_ancho)
        self.resolucion_layout.addWidget(self.input_alto)
        
        # Checkbox para la opción de atenuar el fondo, pudiendo elegir:
        # -El tamaño del grupo para el cálculo de la mediana (depende del poder de procesamiento disponible)
        # -El factor de atenuación del fondo
        # -El umbral de la diferencia a partir de la cual se considera que hay un pez y no ruido
        self.checkbox_atenuar_fondo = QCheckBox("Atenuar Fondo", self)
        self.checkbox_atenuar_fondo.setChecked(True)
        self.checkbox_atenuar_fondo.stateChanged.connect(self.toggle_atenuar_fondo_options)

        self.atenuar_fondo_layout = QGridLayout()

        self.input_sizeGrupo = QLineEdit(self)
        self.input_sizeGrupo.setPlaceholderText("Tamaño Grupo")
        self.input_sizeGrupo.setText("30")
        self.atenuar_fondo_layout.addWidget(self.input_sizeGrupo, 0, 0)

        self.input_factor_at = QLineEdit(self)
        self.input_factor_at.setPlaceholderText("Factor de Atenuación")
        self.input_factor_at.setText("0.4")
        self.atenuar_fondo_layout.addWidget(self.input_factor_at , 0, 1)

        self.input_umbral_dif = QLineEdit(self)
        self.input_umbral_dif.setPlaceholderText("Umbral de Diferencia")
        self.input_umbral_dif.setText("7")
        self.atenuar_fondo_layout.addWidget(self.input_umbral_dif, 0, 2)

        self.checkbox_apertura = QCheckBox("Apertura", self)
        self.checkbox_apertura.setChecked(True)
        self.atenuar_fondo_layout.addWidget(self.checkbox_apertura, 1, 0)
        self.apertura_kernel_layout = QHBoxLayout()
        self.apertura_kernel_width = QLineEdit(self)
        self.apertura_kernel_width.setPlaceholderText("Ancho kernel")
        self.apertura_kernel_width.setText("3")
        self.apertura_kernel_height = QLineEdit(self)
        self.apertura_kernel_height.setPlaceholderText("Alto kernel")
        self.apertura_kernel_height.setText("3")
        self.apertura_kernel_layout.addWidget(self.apertura_kernel_width)
        self.apertura_kernel_layout.addWidget(self.apertura_kernel_height)
        self.atenuar_fondo_layout.addLayout(self.apertura_kernel_layout, 1, 1, 1, 2)


        self.checkbox_cierre = QCheckBox("Cierre", self)
        self.checkbox_cierre.setChecked(False)
        self.atenuar_fondo_layout.addWidget(self.checkbox_cierre, 2, 0)       
        self.cierre_kernel_layout = QHBoxLayout()
        self.cierre_kernel_width = QLineEdit(self)
        self.cierre_kernel_width.setPlaceholderText("Ancho kernel")
        self.cierre_kernel_width.setText("3")
        self.cierre_kernel_height = QLineEdit(self)
        self.cierre_kernel_height.setPlaceholderText("Alto kernel")
        self.cierre_kernel_height.setText("3")
        self.cierre_kernel_layout.addWidget(self.cierre_kernel_width)
        self.cierre_kernel_layout.addWidget(self.cierre_kernel_height)
        self.atenuar_fondo_layout.addLayout(self.cierre_kernel_layout, 2, 1, 2, 2)


        
        # Botón para iniciar el proceso de preprocesado
        self.boton_iniciar_preprocesado = QPushButton("Iniciar Preprocesado", self)
        self.boton_iniciar_preprocesado.clicked.connect(self.toggle_iniciar_preprocesado)
        
        # Layout de la seleccion de parámetros
        layout = QVBoxLayout()
        layout.addWidget(self.label_video_input)
        layout.addWidget(self.boton_seleccionar_video)
        layout.addWidget(self.label_output)
        layout.addWidget(self.boton_seleccionar_output)
        layout.addWidget(self.checkbox_resolucion)
        layout.addLayout(self.resolucion_layout)
        layout.addWidget(self.checkbox_atenuar_fondo)
        layout.addLayout(self.atenuar_fondo_layout)
        layout.addWidget(self.boton_iniciar_preprocesado)

        self.setLayout(layout)
        self.worker_thread = None
        
        # Etiqueta para mostrar la etapa actual
        self.etapa_actual = QLabel("Esperando para iniciar... ", self)
        self.etapa_actual.setAlignment(Qt.AlignCenter)
        
        # Barra de progreso general (etapas)
        self.barra_progreso_etapas = QProgressBar(self)
        self.barra_progreso_etapas.setValue(0)
        self.barra_progreso_etapas.setRange(0, 100)
        
        # Barra de progreso especifica (progreso dentro de la etapa)
        self.barra_progreso_especifica = QProgressBar(self)
        self.barra_progreso_especifica.setValue(0)
        self.barra_progreso_especifica.setRange(0, 100)
        
        # Layout del feedback del progreso del procesado
        layout.addWidget(self.etapa_actual)
        layout.addWidget(QLabel("Progreso general: ", self))
        layout.addWidget(self.barra_progreso_etapas)
        layout.addWidget(QLabel("Progreso de la etapa actual: ", self))
        layout.addWidget(self.barra_progreso_especifica)

        # Layout de la imagen procesada
        self.label_imagen_procesada = QLabel("Imagen procesada", self)
        self.label_imagen_procesada.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_imagen_procesada)
        
        # Cargar las rutas guardadas en QSettings
        self.cargar_rutas_guardadas()

        # Inicializar las opciones que dependen de los checkboxes
        self.toggle_input_resolucion()
        self.toggle_atenuar_fondo_options()
    
        
    # Función para actualizar la etapa actual en la interfaz
    def actualizar_etapa(self, etapa):
        self.etapa_actual.setText(etapa)
        
       
    # Función para cargar las rutas guardadas en QSettings 
    def cargar_rutas_guardadas(self):
        ultima_ruta_video = self.settings.value("ultima_ruta_video", "")
        if ultima_ruta_video:
            self.label_video_input.setText(f"Archivo seleccionado: {ultima_ruta_video}")
        
        ultima_ruta_output = self.settings.value("ultima_ruta_output", "")
        if ultima_ruta_output:
            self.label_output.setText(f"Carpeta de salida seleccionada: {ultima_ruta_output}")
        
    
    # Función para habilitar o deshabilitar los inputs de resolución según el estado del checkbox
    def toggle_input_resolucion(self):
        if self.checkbox_resolucion.isChecked():
            self.input_ancho.setEnabled(True)
            self.input_alto.setEnabled(True)
        else:
            self.input_ancho.setEnabled(False)
            self.input_alto.setEnabled(False)

    # Función para habilitar o deshabilitar los inputs de atenuar fondo según el estado del checkbox
    def toggle_atenuar_fondo_options(self):
        enabled = self.checkbox_atenuar_fondo.isChecked()
        self.input_sizeGrupo.setEnabled(enabled)
        self.input_factor_at.setEnabled(enabled)
        self.input_umbral_dif.setEnabled(enabled)
        self.checkbox_apertura.setEnabled(enabled)
        self.checkbox_cierre.setEnabled(enabled)
        self.apertura_kernel_width.setEnabled(enabled)
        self.apertura_kernel_height.setEnabled(enabled)
        self.cierre_kernel_width.setEnabled(enabled)
        self.cierre_kernel_height.setEnabled(enabled)
        
    
    # Funcion para seleccionar el archivo de video
    def seleccionar_video(self):
        # Obtener ultima ruta de video seleccionada
        ultima_ruta_video = self.settings.value("ultima_ruta_video", "")
        
        # Abrir el cuadro de diálogo para seleccionar un archivo de video
        video, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", ultima_ruta_video, "Archivos de Video (*.mp4 *.avi *.mov)")
        
        # Verificar si el usuario ha seleccionado un archivo
        if video:
            # Guardar la ruta del video en QSettings
            self.settings.setValue("ultima_ruta_video", video)
            # Mostrar el nombre del archivo seleccionado en el QLabel
            self.label_video_input.setText(f"Archivo seleccionado: {video}")
            
            
    # Funcion para seleccionar la carpeta de salida
    def seleccionar_output(self):
        # Obtener ultima ruta de dorectorio de output seleccionada
        ultima_ruta_output = self.settings.value("ultima_ruta_output", "")
        
        # Abrir el cuadro de diálogo para seleccionar una carpeta de salida
        output = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Salida", ultima_ruta_output)
        
        # Verificar si el usuario ha seleccionado una carpeta de salida
        if output:
            # Guardar la ruta de la carpeta de salida en QSettings
            self.settings.setValue("ultima_ruta_output", output)
            # Mostrar la carpeta de salida seleccionada en el QLabel
            self.label_output.setText(f"Carpeta de salida seleccionada: {output}")
         
         
    # Función para iniciar o cancelar el proceso de preprocesado
    def toggle_iniciar_preprocesado(self):
        #Cancelar el proceso si ya se ha iniciado
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancelar()
            self.boton_iniciar_preprocesado.setText("Iniciar Preprocesado")
        else:
            # Obtener el video seleccionado y la carpeta de salida
            video_path = self.label_video_input.text().split(":")[1].strip()
            output_path = self.label_output.text().split(":")[1].strip()

            if video_path and output_path:
                # Obtener el estado del checkbox de redimensionar
                redimensionar_flag = self.checkbox_resolucion.isChecked()
                width = int(self.input_ancho.text()) if self.input_ancho.text().isdigit() else 1920
                height = int(self.input_alto.text()) if self.input_alto.text().isdigit() else 1080
                # Obtener el estado del checkbox de atenuar el fondo y sus parámetros
                atenuar_fondo_flag = self.checkbox_atenuar_fondo.isChecked()
                sizeGrupo = int(self.input_sizeGrupo.text()) if self.input_sizeGrupo.text().isdigit() else 30
                factor_at = float(self.input_factor_at.text()) if self.input_factor_at.text().isdigit() else 0.4
                umbral_dif = int(self.input_umbral_dif.text()) if self.input_umbral_dif.text().isdigit() else 7
                apertura_flag = self.checkbox_apertura.isChecked()
                cierre_flag = self.checkbox_cierre.isChecked()
                apertura_kernel_width = int(self.apertura_kernel_width.text()) if self.apertura_kernel_width.text().isdigit() else 3
                apertura_kernel_height = int(self.apertura_kernel_height.text()) if self.apertura_kernel_height.text().isdigit() else 3
                apertura_kernel_size = (apertura_kernel_width, apertura_kernel_height)
                cierre_kernel_width = int(self.cierre_kernel_width.text()) if self.cierre_kernel_width.text().isdigit() else 3
                cierre_kernel_height = int(self.cierre_kernel_height.text()) if self.cierre_kernel_height.text().isdigit() else 3
                cierre_kernel_size = (cierre_kernel_width, cierre_kernel_height)
                #Crear y ejecutar el hilo de trabajo
                self.worker_thread = WorkerThread(video_path, output_path, redimensionar_flag, width, height, atenuar_fondo_flag, sizeGrupo, factor_at, umbral_dif, apertura_flag, cierre_flag, apertura_kernel_size, cierre_kernel_size)
                 # Conectar las señales del hilo de trabajo con las funciones de actualización de la interfaz
                self.worker_thread.etapa_actual_signal.connect(self.actualizar_etapa)
                self.worker_thread.progreso_general_signal.connect(self.barra_progreso_etapas.setValue)
                self.worker_thread.progreso_especifico_signal.connect(self.barra_progreso_especifica.setValue)
                # Conectar una señal para recibir la imagen procesada y mostrarla
                self.worker_thread.finished.connect(lambda: self.mostrar_resultado())
                
                self.worker_thread.start()
                
                self.boton_iniciar_preprocesado.setText("Cancelar Preprocesado")
         

            
            
    # Función para mostrar el resultado del preprocesado
    def mostrar_resultado(self):
        self.boton_iniciar_preprocesado.setText("Iniciar Preprocesado")
        #Mostrar la imagen procesada 
        # if imagen_procesada is not None:
        #     # Convertir de BGR a RGB
        #     imagen_rgb = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2RGB)
        #     # Obtener dimensiones y convertir a QImage
        #     height, width, channel = imagen_rgb.shape
        #     bytes_per_line = 3 * width
        #     q_img = QImage(imagen_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        #     # Coonvertir a QPixmap y mostrar en el QLabel
        #     pixmap = QPixmap.fromImage(q_img)
        #     self.label_imagen_procesada.setPixmap(pixmap)
        #     self.label_imagen_procesada.setAlignment(Qt.AlignCenter)


