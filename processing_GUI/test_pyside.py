from PySide6.QtWidgets import QApplication, QWidget, QPushButton

app = QApplication([])  # Inicia la aplicación Qt

window = QWidget()  # Crea una ventana
window.setWindowTitle("Prueba de PySide6")

button = QPushButton("Haz clic aquí", window)  # Crea un botón
button.resize(200, 100)  # Ajusta el tamaño del botón
button.move(50, 50)  # Mueve el botón dentro de la ventana

window.resize(300, 200)  # Ajusta el tamaño de la ventana
window.show()  # Muestra la ventana

app.exec()  # Ejecuta la aplicación
