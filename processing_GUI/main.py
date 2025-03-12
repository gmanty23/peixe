import sys
from PySide6.QtWidgets import QApplication
from ventanas.ventana_inicio import VentanaInicio

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaInicio()
    ventana.show()
    sys.exit(app.exec())