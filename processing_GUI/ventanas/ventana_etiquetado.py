import sys
import subprocess
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                              QLabel, QSpinBox, QPushButton, QFileDialog, 
                              QHBoxLayout, QLineEdit, QMessageBox)

class VentanaEtiquetado(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generador de Comandos")
        self.setMinimumSize(400, 200)
        
        # Widgets principales
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Configuración de widgets
        self.setup_ui()
    
    def setup_ui(self):
        # Sección para el número de objetos
        num_layout = QHBoxLayout()
        num_label = QLabel("Número de objetos:")
        self.num_spinbox = QSpinBox()
        self.num_spinbox.setRange(1, 100)  # Rango de 1 a 100
        self.num_spinbox.setValue(5)       # Valor por defecto
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_spinbox)
        self.layout.addLayout(num_layout)
        
        # Sección para la selección del directorio
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
        
        # Sección para el comando generado
        cmd_layout = QHBoxLayout()
        cmd_label = QLabel("Comando:")
        self.cmd_lineedit = QLineEdit()
        self.cmd_lineedit.setReadOnly(True)
        cmd_layout.addWidget(cmd_label)
        cmd_layout.addWidget(self.cmd_lineedit)
        self.layout.addLayout(cmd_layout)
        
        # Botón para generar y ejecutar el comando
        self.generate_button = QPushButton("Generar y Ejecutar Comando")
        self.generate_button.clicked.connect(self.generate_and_execute)
        self.layout.addWidget(self.generate_button)
        
        # Conectar señales para actualización en tiempo real
        self.num_spinbox.valueChanged.connect(self.update_command)
        self.dir_lineedit.textChanged.connect(self.update_command)
    
    def select_directory(self):
        """Abre un diálogo para seleccionar directorio y actualiza el campo"""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "Seleccionar Directorio de Trabajo",
            str(Path.home())  # Directorio home como punto de partida
        )
        if dir_path:
            self.dir_lineedit.setText(dir_path)
    
    def update_command(self):
        """Actualiza el comando mostrado según los valores actuales"""
        num_objects = self.num_spinbox.value()
        workspace = self.dir_lineedit.text()
        
        if workspace:
            cmd = f'python interactive_demo.py --num_objects "{num_objects}" --workspace "{workspace}"'
            self.cmd_lineedit.setText(cmd)
        else:
            self.cmd_lineedit.setText("")
    
    def generate_and_execute(self):
        """Genera el comando final y lo ejecuta"""
        num_objects = self.num_spinbox.value()
        workspace = self.dir_lineedit.text()
        
        if not workspace:
            QMessageBox.critical(self, "Error", "Debes seleccionar un directorio de trabajo")
            return
        
        cmd = f'python interactive_demo.py --num_objects "{num_objects}" --workspace "{workspace}"'
        
        # Mostrar confirmación
        reply = QMessageBox.question(
            self,
            "Confirmar Ejecución",
            f"¿Estás seguro de ejecutar el siguiente comando?\n\n{cmd}",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Ejecutar el comando en una terminal nueva (dependiendo del SO)
                if sys.platform == "win32":
                    subprocess.Popen(['start', 'cmd', '/k', cmd], shell=True)
                elif sys.platform == "darwin":
                    subprocess.Popen(['open', '-a', 'Terminal', cmd])
                else:  # Linux y otros
                    subprocess.Popen(['x-terminal-emulator', '-e', cmd])
                
                QMessageBox.information(self, "Éxito", "Comando ejecutado correctamente")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo ejecutar el comando:\n{str(e)}")

