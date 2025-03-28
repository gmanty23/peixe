import subprocess
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                              QLabel, QSpinBox, QPushButton, QFileDialog, 
                              QHBoxLayout, QLineEdit, QMessageBox, QGroupBox, QCheckBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import os
import subprocess
import shutil

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
        
        # Sección para el reinicio del contador de mascaras guardadas 
        reset_layout = QHBoxLayout()
        reset_label = QLabel("Reiniciar contador de máscaras guardadas:")
        self.reset_checkbox = QCheckBox()
        reset_layout.addWidget(reset_label)
        reset_layout.addWidget(self.reset_checkbox)
        params_layout.addLayout(reset_layout)
        
        
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
        reset_flag = self.reset_checkbox.isChecked()
        mask_guardadas_path = os.path.join(workspace, "masks_guardadas")
        if reset_flag:
            if os.path.exists(mask_guardadas_path):
                shutil.rmtree(mask_guardadas_path)
            file_path = os.path.join(workspace, "mask_counter.txt")
            with open(file_path, "w") as f:
                f.write(str("0"))
                
        
        if not workspace:
            QMessageBox.critical(self, "Error", "Debes seleccionar un directorio de trabajo")
            return
        
        cmd = f'python interactive_demo.py --num_objects "{num_objects}" --workspace "{workspace}"'
        
        # 1. Primero intentamos ejecución directa (sin terminal nuevo)
        if self._try_direct_execution(cmd):
            return
            
        # 2. Si falla, probamos con terminales gráficos
        if self._try_graphical_terminals(cmd):
            return
            
        # 3. Último recurso: ejecución con captura de output
        self._execute_with_output(cmd)

    def _try_direct_execution(self, cmd):
        """Intenta ejecución directa sin terminal nuevo"""
        try:
            # Ejecutar en segundo plano sin bloquear
            subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/bash',
                start_new_session=True
            )
            return True
        except Exception:
            return False

    def _try_graphical_terminals(self, cmd):
        """Intenta con varios terminales gráficos"""
        terminals = [
            ('xterm', ['xterm', '-hold', '-e', cmd]),
            ('konsole', ['konsole', '-e', 'bash', '-c', f'{cmd}; exec bash']),
            ('gnome-terminal', ['gnome-terminal', '--', 'bash', '-c', f'{cmd}; exec bash']),
            ('xfce4-terminal', ['xfce4-terminal', '-x', 'bash', '-c', f'{cmd}; exec bash']),
            ('lxterminal', ['lxterminal', '-e', 'bash', '-c', f'{cmd}; exec bash'])
        ]
        
        for name, command in terminals:
            if self._try_execute(command):
                print(f"Éxito usando terminal: {name}")
                return True
        return False

    def _try_execute(self, command):
        """Intenta ejecutar un comando y devuelve True si tiene éxito"""
        try:
            subprocess.Popen(command)
            return True
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def _execute_with_output(self, cmd):
        """Ejecuta el comando directamente y muestra el resultado"""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable='/bin/bash'
            )
            
            output = f"Comando: {cmd}\n\nSalida:\n{result.stdout}"
            if result.stderr:
                output += f"\n\nErrores:\n{result.stderr}"
                
            QMessageBox.information(self, "Resultado", output)
            
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Error al ejecutar:\n{cmd}\n\n"
                f"Código: {e.returncode}\n\n"
                f"Error:\n{e.stderr}\n\n"
                f"Salida:\n{e.stdout}"
            )
            QMessageBox.critical(self, "Error de Ejecución", error_msg)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Grave",
                f"No se pudo ejecutar el comando:\n{cmd}\n\n"
                f"Error: {str(e)}\n\n"
                "Solución recomendada:\n"
                "1. Ejecuta manualmente en una terminal:\n"
                f"{cmd}\n\n"
                "2. O instala un terminal gráfico como xterm:\n"
                "sudo apt install xterm"
            )
    
    def volver_a_inicio(self):
        self.close()
        if self.parent_window:
            self.parent_window.show()
            
            
            
    