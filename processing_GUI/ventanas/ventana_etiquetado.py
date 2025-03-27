import subprocess
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                              QLabel, QSpinBox, QPushButton, QFileDialog, 
                              QHBoxLayout, QLineEdit, QMessageBox, QGroupBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class VentanaEtiquetado(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent  # Referencia a la ventana principal
        self.setWindowTitle("Herramienta de Etiquetado - Cutie")
        self.setMinimumSize(500, 250)
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        # Widget central y layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Grupo: Parámetros de etiquetado
        params_group = QGroupBox("Configuración de Etiquetado")
        params_layout = QVBoxLayout()
        
        # Sección para el número de objetos
        num_layout = QHBoxLayout()
        num_label = QLabel("Número de objetos:")
        self.num_spinbox = QSpinBox()
        self.num_spinbox.setRange(1, 100)
        self.num_spinbox.setValue(5)
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_spinbox)
        params_layout.addLayout(num_layout)
        
        # Sección para el directorio
        dir_layout = QHBoxLayout()
        dir_label = QLabel("Directorio de trabajo:")
        self.dir_lineedit = QLineEdit()
        self.dir_lineedit.setPlaceholderText("Selecciona un directorio...")
        dir_button = QPushButton("Examinar...")
        dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.dir_lineedit)
        dir_layout.addWidget(dir_button)
        params_layout.addLayout(dir_layout)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        self.back_button = QPushButton("Atrás")
        self.back_button.clicked.connect(self.volver_a_inicio)
        self.generate_button = QPushButton("Iniciar Etiquetado")
        self.generate_button.clicked.connect(self.execute_command)
        
        buttons_layout.addWidget(self.back_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.generate_button)
        main_layout.addLayout(buttons_layout)
    
    def setup_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a5276;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
            QSpinBox {
                padding: 5px;
            }
        """)
        
    
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
        
        cmd = f'python interactive_demo.py --num_objects {num_objects} --workspace "{workspace}"'
        
        try:
            # Detección automática del terminal
            terminals = ['gnome-terminal', 'konsole', 'xterm', 'xfce4-terminal']
            terminal_found = False
            
            for terminal in terminals:
                try:
                    if terminal == 'gnome-terminal':
                        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{cmd}; exec bash'])
                    elif terminal == 'konsole':
                        subprocess.Popen(['konsole', '-e', 'bash', '-c', f'{cmd}; exec bash'])
                    else:
                        subprocess.Popen([terminal, '-e', f'{cmd}; exec bash'])
                    terminal_found = True
                    break
                except FileNotFoundError:
                    continue
            
            if not terminal_found:
                # Fallback para sistemas sin terminal gráfica
                QMessageBox.information(
                    self, 
                    "Ejecutando en segundo plano", 
                    f"El etiquetado se está ejecutando en segundo plano.\nComando: {cmd}"
                )
                subprocess.Popen(['bash', '-c', cmd])
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"No se pudo ejecutar el comando:\n{str(e)}\n\n"
                "Asegúrate de tener instalado un terminal como:\n"
                "gnome-terminal, konsole, xterm o xfce4-terminal"
            )
    
    def volver_a_inicio(self):
        self.close()
        if self.parent_window:
            self.parent_window.show()