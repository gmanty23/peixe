from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
# from processing_GUI.ventanas.ventana_preprocesado import VentanaPreprocesado  
from processing_GUI.ventanas.ventana_etiquetado import VentanaEtiquetado
from processing_GUI.ventanas.ventana_visualizacion import VentanaVisualizacion  
from processing_GUI.ventanas.ventana_postprocesado import VentanaPostprocesado
from processing_GUI.ventanas.ventana_analisis import VentanaAnalisis


class VentanaInicio(QWidget):
    def __init__(self):
        super().__init__()
        
        # Configuración inicial de la ventana: título y tamaño
        self.setWindowTitle("Sistema de Automatización de Estudios Etológicos")
        self.setMinimumSize(400, 300)
        
        # Estilo CSS para darle los colores
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f4f8;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel#titulo {
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 12px 20px;
                font-size: 16px;
                min-width: 200px;
                margin: 5px 0;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a5276;
            }
        """)
        
        # Texto con el título y el pez
        self.logo = QLabel("🐟", self)
        self.logo.setAlignment(Qt.AlignCenter)
        self.logo.setFont(QFont('Arial', 60))
        
        self.titulo = QLabel("PEIXE", self)
        self.titulo.setObjectName("titulo")         # Para aplicar el estilo CSS
        self.titulo.setAlignment(Qt.AlignCenter)
        
        # Crear los botones
        # self.boton_preprocesado = QPushButton("Preprocesado", self)
        self.boton_etiquetado = QPushButton("Etiquetado", self)
        self.boton_postprocesado = QPushButton("Postprocesado", self)
        self.boton_visualizacion = QPushButton("Visualización", self)
        self.boton_analisis = QPushButton("Análisis", self)
        
        # Configurar cursor de mano al pasar por encima de todos los botones
        for boton in [self.boton_etiquetado, self.boton_postprocesado]:
            boton.setCursor(Qt.PointingHandCursor)
        
        
        # Conecatar los botones a las funciones
        # self.boton_preprocesado.clicked.connect(self.preprocesado)
        self.boton_etiquetado.clicked.connect(self.etiquetado)
        self.boton_postprocesado.clicked.connect(self.postprocesado)
        self.boton_visualizacion.clicked.connect(self.visualizacion)
        self.boton_analisis.clicked.connect(self.analisis)
        
        # Layout principal
        layout = QVBoxLayout()
        layout.addStretch(1) # Espacio entre el pez y el título
        layout.addWidget(self.logo)
        layout.addWidget(self.titulo)
        layout.addStretch(1) # Espacio entre el título y los botones
        # layout.addWidget(self.boton_preprocesado, alignment=Qt.AlignCenter)
        layout.addWidget(self.boton_etiquetado, alignment=Qt.AlignCenter)
        layout.addWidget(self.boton_postprocesado, alignment=Qt.AlignCenter)
        layout.addWidget(self.boton_visualizacion, alignment=Qt.AlignCenter)
        layout.addWidget(self.boton_analisis, alignment=Qt.AlignCenter)
        layout.addStretch(2) # Espacio entre los botones y el final de la ventana
        
        self.setLayout(layout)
        
    # def preprocesado(self):
    #     self.ventana_preprocesado = VentanaPreprocesado(parent=self)
    #     self.ventana_preprocesado.show()
    #     #self.hide()
        
    def postprocesado(self):
        self.ventana_postprocesado = VentanaPostprocesado(parent=self)
        self.ventana_postprocesado.show()
        #self.hide()
        
    
    def etiquetado(self):
        self.ventana_etiquetado = VentanaEtiquetado(parent=self)
        self.ventana_etiquetado.show()
        #self.hide()
    
    def visualizacion(self):
        self.ventana_visualizacion = VentanaVisualizacion(parent=self)
        self.ventana_visualizacion.show()
        #self.hide()

    def analisis(self):
        self.ventana_analisis = VentanaAnalisis(parent=self)
        self.ventana_analisis.show()
        #self.hide()













































