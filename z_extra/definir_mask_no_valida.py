import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPolygonF
from PySide6.QtCore import Qt, QPointF, QEvent

class VentanaSeleccionMascara(QDialog):
    def __init__(self, frame, output_mask_path="zona_no_valida.png"):
        super().__init__()
        self.setWindowTitle("Seleccionar zona no válida")
        self.setModal(True)
        self.mask_path = output_mask_path
        self.points = []
        self.polygon_item = None

        # Convertir frame a QImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Crear escena y vista
        self.scene = QGraphicsScene()
        self.pixmap = QPixmap.fromImage(qimg)
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.pixmap_item)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        # Botones
        self.boton_guardar = QPushButton("Guardar máscara")
        self.boton_guardar.clicked.connect(self.guardar_mascara)
        self.boton_cancelar = QPushButton("Cancelar")
        self.boton_cancelar.clicked.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.boton_guardar)
        layout.addWidget(self.boton_cancelar)
        self.setLayout(layout)

        self.frame_size = (w, h)

    def eventFilter(self, source, event):
        if source == self.view.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                pos = self.view.mapToScene(event.pos())
                self.points.append(pos)
                self.actualizar_poligono()
            elif event.type() == QEvent.MouseMove:
                pass  # Opcional: podrías añadir feedback visual dinámico si quieres
            elif event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                self.guardar_mascara()
        return super().eventFilter(source, event)

    def actualizar_poligono(self):
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
        if len(self.points) >= 2:
            polygon = QPolygonF(self.points)
            self.polygon_item = QGraphicsPolygonItem(polygon)
            pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
            self.polygon_item.setPen(pen)
            self.polygon_item.setBrush(QColor(255, 0, 0, 50))
            self.scene.addItem(self.polygon_item)

    def guardar_mascara(self):
        mask = np.zeros((self.frame_size[1], self.frame_size[0]), dtype=np.uint8)
        if len(self.points) >= 3:
            pts = np.array([[p.x(), p.y()] for p in self.points], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            cv2.imwrite(self.mask_path, mask)
            print(f"✅ Máscara guardada en: {self.mask_path}")
            self.accept()
        else:
            print("⚠️ Necesitas al menos 3 puntos para formar un polígono.")

def main(video_path, frame_num, bbox, output_size=(1920, 1080)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el vídeo {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: No se pudo leer el frame {frame_num}")
        return

    # Recorte
    x, y, w, h = bbox
    frame_cropped = frame[y:y+h, x:x+w]

    # Resize
    frame_resized = cv2.resize(frame_cropped, output_size)

    # Lanzar ventana de selección
    app = QApplication(sys.argv)
    ventana = VentanaSeleccionMascara(frame_resized)
    ventana.exec()

if __name__ == "__main__":
    # Configura tus parámetros
    video_path = "/home/gmanty/code/AnemoNAS/06-12-23/0921/USCL2-091606-092106.mp4" 
    frame_num = 5000
    bbox = (550, 960, 2225, 1186)
    main(video_path, frame_num, bbox)
