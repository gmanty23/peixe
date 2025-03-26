# from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
# from PySide6.QtCore import Qt
# from PySide6.QtGui import QFont
# from processing_GUI.ventanas.ventana_preprocesado import VentanaPreprocesado  
# from processing_GUI.ventanas.ventana_etiquetado import VentanaEtiquetado

# class VentanaInicio(QWidget):
#     def __init__(self):
#         super().__init__()
        
#         # Establecer título de la ventana
#         self.setWindowTitle("Herramienta de etiquetado")

#         # Establecer el tamaño de la ventana
#         self.setGeometry(100, 100, 400, 300)  # x, y, ancho, alto
        
#         # Crear el texto con el emoticono del pez
#         self.texto_inicio = QLabel("🐟", self)
#         self.texto_inicio.setAlignment(Qt.AlignCenter)
#         self.texto_inicio.setFont(QFont('Arial', 50))

#         # Crear los botones. 
#         # El self le dice al botón (y a otros widgets) que la ventana es su contenedor. Esto es esencial para el manejo de la disposición de los widgets y la destrucción de los objetos cuando se cierra la ventana.
#         self.boton_preprocesado = QPushButton("Preprocesado", self)
#         self.boton_etiquetado = QPushButton("Etiquetado", self)
#         self.boton_postprocesado= QPushButton("Postprocesado", self)
        
#         #Conectar los botones con las funciones
#         self.boton_preprocesado.clicked.connect(self.preprocesado)
#         self.boton_etiquetado.clicked.connect(self.etiquetado)
#         self.boton_postprocesado.clicked.connect(self.postprocesado)
        
#         # Crear un layout 
#         layout = QVBoxLayout()
        
#         layout.addWidget(self.texto_inicio)
#         layout.addWidget(self.boton_preprocesado)
#         layout.addWidget(self.boton_etiquetado)
#         layout.addWidget(self.boton_postprocesado)
        
#         layout.setSpacing(15)
        
        
#         # Establecer el layout de la ventana
#         self.setLayout(layout)
        
#     def preprocesado(self):
#         print("Preprocesado")
#         self.ventana_preprocesado = VentanaPreprocesado()
#         self.ventana_preprocesado.show()
        
#     def postprocesado(self):
#         print("Postprocesado")

    
#     def etiquetado(self):
#         print("Etiquetado")
#         self.ventana_etiquetado = VentanaEtiquetado()
#         self.ventana_etiquetado.show()
    
    
    
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from processing_GUI.ventanas.ventana_preprocesado import VentanaPreprocesado  
from processing_GUI.ventanas.ventana_etiquetado import VentanaEtiquetado

class VentanaInicio(QWidget):
    def __init__(self):
        super().__init__()
        
        # Configuración inicial de la ventana: título y tamaño
        self.setWindowTitle("Herramienta de Etiquetado de Peces")
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
        self.boton_preprocesado = QPushButton("Preprocesado", self)
        self.boton_etiquetado = QPushButton("Etiquetado", self)
        self.boton_postprocesado = QPushButton("Postprocesado", self)
        
        # Configurar cursor de mano al pasar por encima de todos los botones
        for boton in [self.boton_preprocesado, self.boton_etiquetado, self.boton_postprocesado]:
            boton.setCursor(Qt.PointingHandCursor)
        
        
        # Conecatar los botones a las funciones
        self.boton_preprocesado.clicked.connect(self.preprocesado)
        self.boton_etiquetado.clicked.connect(self.etiquetado)
        self.boton_postprocesado.clicked.connect(self.postprocesado)
        
        # Layout principal
        layout = QVBoxLayout()
        layout.addStretch(1) # Espacio entre el pez y el título
        layout.addWidget(self.logo)
        layout.addWidget(self.titulo)
        layout.addStretch(1) # Espacio entre el título y los botones
        layout.addWidget(self.boton_preprocesado, alignment=Qt.AlignCenter)
        layout.addWidget(self.boton_etiquetado, alignment=Qt.AlignCenter)
        layout.addWidget(self.boton_postprocesado, alignment=Qt.AlignCenter)
        layout.addStretch(2) # Espacio entre los botones y el final de la ventana
        
        self.setLayout(layout)
        
    def preprocesado(self):
        self.ventana_preprocesado = VentanaPreprocesado()
        self.ventana_preprocesado.show()
        self.hide()  
        
    def postprocesado(self):
        print("Postprocesado")
        # A implementar
    
    def etiquetado(self):
        self.ventana_etiquetado = VentanaEtiquetado()
        self.ventana_etiquetado.show()
        self.hide()  # Oculta la ventana actual