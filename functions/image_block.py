import os
from PIL import Image
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsItem
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPen, QBrush, QColor
)
from PyQt6.QtCore import Qt, QLocale

from . import coordinate_transforms as ct

QLocale.setDefault(QLocale.c())


def load_image_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    
    # Fix EXIF rotation that cv2.imread ignores
    try:
        pil_img = Image.open(path)
        exif = pil_img._getexif()
        if exif:
            orientation = exif.get(274)  # 274 is the EXIF orientation tag
            if orientation == 3:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif orientation == 6:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        pass  # No EXIF data or not a JPEG, continue without rotation fix

    # normalise to RGBA
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    return img


# holds image path, display pixmap, and 3x3 homography
class ImageEntry:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        # H maps display-pixel coords -> canvas coords
        self.H = np.eye(3, dtype=np.float64)
        self.pixmap = None
        self.thumb = None
        self.scene_item = None
        self._load()

    def _load(self):
        img = load_image_rgba(self.path)

        # cap display size to avoid huge scene items
        h, w = img.shape[:2]
        max_dim = 2000
        if max(h, w) > max_dim:
            sc = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

        h, w = img.shape[:2]
        img = np.ascontiguousarray(img)
        # tobytes() keeps pixel data alive beyond numpy array scope
        qimg = QImage(img.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.thumb = self.pixmap.scaled(
            52, 52,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )


# canvas item - drag is implemented manually so H stays as source of truth (no Qt setPos drift)
class ImageItem(QGraphicsPixmapItem):
    def __init__(self, entry, on_select, on_moved):
        super().__init__(entry.pixmap)
        self.entry = entry
        self.on_select = on_select
        self.on_moved = on_moved
        self._drag_scene_start = None
        self._H_at_drag_start = None

        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setPos(0, 0)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_scene_start = event.scenePos()
            self._H_at_drag_start = self.entry.H.copy()
            self.on_select(self.entry)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_scene_start is not None:
            # scene delta
            delta_scene = event.scenePos() - self._drag_scene_start

            dx = delta_scene.x()
            dy = delta_scene.y()

            H = self._H_at_drag_start.copy()
            H[0, 2] += dx
            H[1, 2] += dy
            self.entry.H = H
            self.setTransform(ct.h_to_qt(self.entry.H))
            self.on_moved(self.entry)

        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_scene_start = None
        self._H_at_drag_start = None
        event.accept()

class CameraFrame(QGraphicsRectItem):
    def __init__(self, w, h):
        super().__init__(0, 0, w, h)
        self.setPen(QPen(QColor("#e94560"), 2, Qt.PenStyle.DashLine))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setZValue(9999)