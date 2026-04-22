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
from PyQt6.QtCore import Qt, QLocale, QRectF, QPointF

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


# It's a handle for corner pin mode
class CornerHandle(QGraphicsItem):
    SIZE = 8  # visual size in pixels

    def __init__(self, parent, idx):
        super().__init__()
        self.parent = parent
        self.idx = idx

        self.setZValue(10000)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)

        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)

        # cosmetic pen (constant width)
        self.pen = QPen(QColor("#ffcc00"))
        self.pen.setWidth(2)
        self.pen.setCosmetic(True)

        self.pen_hover = QPen(QColor("#ffffff"))
        self.pen_hover.setWidth(2)
        self.pen_hover.setCosmetic(True)

        self._hover = False

    def boundingRect(self):
        s = self.SIZE
        return QRectF(-s, -s, s * 2, s * 2)

    def paint(self, painter, option, widget):
        painter.setPen(self.pen_hover if self._hover else self.pen)

        s = self.SIZE

        # draw crosshair (+)
        painter.drawLine(-s, 0, s, 0)
        painter.drawLine(0, -s, 0, s)

        # optional: small center dot
        painter.drawPoint(0, 0)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.parent.handle_moved(self.idx, value)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()


# canvas item - drag is implemented manually so H stays as source of truth (no Qt setPos drift)
class ImageItem(QGraphicsPixmapItem):
    def __init__(self, entry, on_select, on_moved):
        super().__init__(entry.pixmap)
        self.entry = entry
        self.on_select = on_select
        self.on_moved = on_moved

        self._drag_scene_start = None
        self._H_at_drag_start = None

        # Corner pin
        self._updating_handles = False
        self._corner_pin_mode = False
        self._handles = []
        self._corner_pts = None  # 4x2 array in scene space

        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setPos(0, 0)

    # Normal drag (disabled in corner pin)
    def mousePressEvent(self, event):
        if self._corner_pin_mode:
            event.ignore()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_scene_start = event.scenePos()
            self._H_at_drag_start = self.entry.H.copy()
            self.on_select(self.entry)

        event.accept()

    def mouseMoveEvent(self, event):
        if self._corner_pin_mode:
            event.ignore()
            return

        if self._drag_scene_start is not None:
            delta = event.scenePos() - self._drag_scene_start
            H = self._H_at_drag_start.copy()
            H[0, 2] += delta.x()
            H[1, 2] += delta.y()
            self.entry.H = H

            self.setTransform(ct.h_to_qt(self.entry.H))
            self.on_moved(self.entry)

        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_scene_start = None
        self._H_at_drag_start = None
        event.accept()

    # Corner pin toggle
    def toggle_corner_pin(self):
        if self._corner_pin_mode:
            self.disable_corner_pin()
        else:
            self.enable_corner_pin()

    def enable_corner_pin(self):
        self._corner_pin_mode = True
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

        # initial corners from current H
        w = self.pixmap().width()
        h = self.pixmap().height()

        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float64)

        self._corner_pts = ct.apply_H(self.entry.H, corners)

        # create handles
        for i in range(4):
            hnd = CornerHandle(self, i)
            hnd.setPos(self._corner_pts[i][0], self._corner_pts[i][1])
            self.scene().addItem(hnd)
            self._handles.append(hnd)

    def disable_corner_pin(self):
        self._corner_pin_mode = False
        self.setAcceptedMouseButtons(Qt.MouseButton.AllButtons)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

        for h in self._handles:
            self.scene().removeItem(h)
        self._handles.clear()
        self._corner_pts = None

    # Update H on handle move
    def handle_moved(self, idx, pos):
        if self._corner_pts is None or self._updating_handles:
            return

        self._corner_pts[idx] = [pos.x(), pos.y()]

        w = self.pixmap().width()
        h = self.pixmap().height()

        src = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)

        dst = self._corner_pts.astype(np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        self.entry.H = H

        self.setTransform(ct.h_to_qt(self.entry.H))
        self.on_moved(self.entry)

    # Sync handles from H
    def sync_handles(self):
        if not self._corner_pin_mode or self._corner_pts is None:
            return

        w = self.pixmap().width()
        h = self.pixmap().height()

        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float64)

        pts = ct.apply_H(self.entry.H, corners)

        self._updating_handles = True

        self._corner_pts = pts
        for i, hnd in enumerate(self._handles):
            hnd.setPos(pts[i][0], pts[i][1])

        self._updating_handles = False