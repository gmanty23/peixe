# GUI para agilizar el etiquetado de imagenes

# QApplication: Es necesario para manejar la aplicación.
# QWidget: Es la clase base para cualquier ventana o contenedor en Qt.
# QPushButton: Es la clase para crear botones.
# QVBoxLayout: Es un layout que organiza los widgets de forma vertical (uno encima de otro).
# sys: Necesitamos esto para pasar los argumentos de la línea de comandos a QApplication y también para terminar la aplicación correctamente.
import sys
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class VentanaInicio(QWidget):
    def __init__(self):
        super().__init__()
        
        # Establecer título de la ventana
        self.setWindowTitle("Herramienta de etiquetado")

        # Establecer el tamaño de la ventana
        self.setGeometry(100, 100, 400, 300)  # x, y, ancho, alto
        
        # Crear el texto con el emoticono del pez
        self.texto_inicio = QLabel("🐟", self)
        self.texto_inicio.setAlignment(Qt.AlignCenter)
        self.texto_inicio.setFont(QFont('Arial', 30))

        # Crear los botones. 
        # El self le dice al botón (y a otros widgets) que la ventana es su contenedor. Esto es esencial para el manejo de la disposición de los widgets y la destrucción de los objetos cuando se cierra la ventana.
        self.boton_preprocesado = QPushButton("Preprocesado", self)
        self.boton_etiquetado = QPushButton("Etiquetado", self)
        self.boton_postprocesado= QPushButton("Postprocesado", self)
        
        #Conectar los botones con las funciones
        self.boton_preprocesado.clicked.connect(self.preprocesado)
        self.boton_etiquetado.clicked.connect(self.etiquetado)
        self.boton_postprocesado.clicked.connect(self.postprocesado)
        
        # Crear un layout 
        layout = QVBoxLayout()
        
        layout.addWidget(self.texto_inicio)
        layout.addWidget(self.boton_preprocesado)
        layout.addWidget(self.boton_etiquetado)
        layout.addWidget(self.boton_postprocesado)
        
        layout.setSpacing(15)
        
        
        # Establecer el layout de la ventana
        self.setLayout(layout)
        
    def preprocesado(self):
        print("Preprocesado")
        
    def postprocesado(self):
        print("Postprocesado")
    
    def etiquetado(self):
        print("Etiquetado")
    
    
    
    
    
    
if __name__ == "__main__":
    
    #Crear aplicacion
    app = QApplication(sys.argv)
    
    #Crear ventana
    ventana = VentanaInicio()
    
    #Mostrar ventana
    ventana.show()
    
    #Ejecurta la aplicación
    sys.exit(app.exec())