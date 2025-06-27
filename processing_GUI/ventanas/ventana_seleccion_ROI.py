from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PySide6.QtCore import Qt, QRectF, QPointF, QEvent
import numpy as np
import cv2

class VentanaSeleccionROI(QDialog):
    def __init__(self, frame, titulo="Seleccionar ROI"):
        super().__init__()
        self.setWindowTitle(titulo)
        self.setModal(True)
        self.roi = None  # (x, y, w, h)
        self.rect_item = None
        self.inicio_pos = None

        # Convertir frame de OpenCV a QImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Escalar imagen si es muy grande (máximo 900px)
        max_dim = 900
        escala = min(max_dim / w, max_dim / h, 1.0)
        self.escala = escala

        if escala < 1.0:
            new_w = int(w * escala)
            new_h = int(h * escala)
            qimg = qimg.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Crear escena y vista
        self.scene = QGraphicsScene()
        self.pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.pixmap_item)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setMouseTracking(True)

        # Botones
        self.boton_aceptar = QPushButton("Aceptar")
        self.boton_aceptar.clicked.connect(self.acceptar_roi)
        self.boton_cancelar = QPushButton("Cancelar")
        self.boton_cancelar.clicked.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.boton_aceptar)
        layout.addWidget(self.boton_cancelar)
        self.setLayout(layout)

        # Tamaño automático de la ventana
        self.resize(self.pixmap.width() + 50, self.pixmap.height() + 100)

        # Conectar eventos
        self.view.viewport().installEventFilter(self)

    def eventFilter(self, source, event):
        if source == self.view.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.inicio_pos = self.view.mapToScene(event.pos())
                if self.rect_item:
                    self.scene.removeItem(self.rect_item)
                    self.rect_item = None
            elif event.type() == QEvent.MouseMove and self.inicio_pos:
                current_pos = self.view.mapToScene(event.pos())
                rect = QRectF(self.inicio_pos, current_pos).normalized()

                if self.rect_item:
                    self.scene.removeItem(self.rect_item)
                self.rect_item = QGraphicsRectItem(rect)
                pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
                self.rect_item.setPen(pen)
                self.scene.addItem(self.rect_item)
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.fin_pos = self.view.mapToScene(event.pos())
                self.inicio_pos = None  # ✅ Esto corta el arrastre
            return False
        return super().eventFilter(source, event)

    def acceptar_roi(self):
        if self.rect_item:
            rect = self.rect_item.rect()
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

            # Ajustar a la escala original si se redimensionó
            if self.escala < 1.0:
                x = int(x / self.escala)
                y = int(y / self.escala)
                w = int(w / self.escala)
                h = int(h / self.escala)
            else:
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

            self.roi = (x, y, w, h)
        else:
            self.roi = None

        self.accept()
