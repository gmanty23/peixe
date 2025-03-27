if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    from processing_GUI.ventanas.ventana_inicio import VentanaInicio

    app = QApplication(sys.argv)
    ventana = VentanaInicio()
    ventana.show()
    sys.exit(app.exec())

