from PySide6.QtWidgets import QApplication
from processing_GUI.ventanas.ventana_analisis import VentanaAnalisis
import sys

def main():
    app = QApplication(sys.argv)
    ventana = VentanaAnalisis()
    ventana.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
