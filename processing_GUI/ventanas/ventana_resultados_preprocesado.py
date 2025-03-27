from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
import os

class VentanaResultados(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparación de Resultados")
        self.setMinimumSize(800, 400)
        self.setup_ui()
        self.setup_styles()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Imagen original
        self.label_original = QLabel("Original")
        self.label_original.setAlignment(Qt.AlignCenter)
        
        # Imagen procesada
        self.label_procesada = QLabel("Procesada")
        self.label_procesada.setAlignment(Qt.AlignCenter)
        
        # Tamaño fijo para visualización
        for label in [self.label_original, self.label_procesada]:
            label.setFixedSize(640, 360)
            label.setScaledContents(True)
        
        layout.addWidget(self.label_original)
        layout.addWidget(self.label_procesada)

    def setup_styles(self):
        self.setStyleSheet("""
            QLabel {
                background: #f8f9fa;
                font-size: 16px;
                font-weight: bold;
                qproperty-alignment: 'AlignCenter';
            }
        """)

    def cargar_imagenes(self, ruta_original, ruta_procesada):
        """Carga y muestra ambas imágenes"""
        try:
            for label, ruta in [
                (self.label_original, ruta_original),
                (self.label_procesada, ruta_procesada)
            ]:
                if os.path.exists(ruta):
                    pixmap = QPixmap(ruta)
                    if not pixmap.isNull():
                        label.setPixmap(pixmap)
                        label.setText("")
                else:
                    label.setText(f"Imagen no encontrada:\n{os.path.basename(ruta)}")
                    
            self.show()
        except Exception as e:
            print(f"Error cargando comparación: {str(e)}")