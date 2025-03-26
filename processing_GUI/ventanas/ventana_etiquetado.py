import subprocess
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                              QLabel, QSpinBox, QPushButton, QFileDialog, 
                              QHBoxLayout, QLineEdit, QMessageBox)

class VentanaEtiquetado(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parámetros Etiquetado")
        self.setMinimumSize(400, 150)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Sección para el número de objetos
        num_layout = QHBoxLayout()
        num_label = QLabel("Número de objetos:")
        self.num_spinbox = QSpinBox()
        self.num_spinbox.setRange(1, 100)
        self.num_spinbox.setValue(5)
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_spinbox)
        self.layout.addLayout(num_layout)
        
        # Sección para el directorio
        dir_layout = QHBoxLayout()
        dir_label = QLabel("Directorio de trabajo:")
        self.dir_lineedit = QLineEdit()
        self.dir_lineedit.setPlaceholderText("Selecciona un directorio...")
        dir_button = QPushButton("Seleccionar...")
        dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.dir_lineedit)
        dir_layout.addWidget(dir_button)
        self.layout.addLayout(dir_layout)
        
        # Botón de ejecución
        self.generate_button = QPushButton("Iniciar Etiquetado")
        self.generate_button.clicked.connect(self.execute_command)
        self.layout.addWidget(self.generate_button)
    
    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "Seleccionar Directorio de Trabajo",
            str(Path.home())
        )
        if dir_path:
            self.dir_lineedit.setText(dir_path)
    
    def execute_command(self):
        num_objects = self.num_spinbox.value()
        workspace = self.dir_lineedit.text()
        
        if not workspace:
            QMessageBox.critical(self, "Error", "Debes seleccionar un directorio de trabajo")
            return
        
        cmd = f'python interactive_demo.py --num_objects "{num_objects}" --workspace "{workspace}"'
        
        try:
            # Primero intentamos con gnome-terminal (común en Ubuntu nativo)
            try:
                subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{cmd}; exec bash'])
            except FileNotFoundError:
                # Si falla, probamos con xterm (disponible en la mayoría de sistemas)
                try:
                    subprocess.Popen(['xterm', '-e', f'{cmd}; exec bash'])
                except FileNotFoundError:
                    # Si todo falla, usamos el terminal por defecto del sistema
                    subprocess.Popen(['bash', '-c', f'{cmd}'])
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"No se pudo ejecutar el comando:\n{str(e)}\n"
                               "Intenta instalar gnome-terminal o xterm:\n"
                               "sudo apt-get install gnome-terminal xterm")