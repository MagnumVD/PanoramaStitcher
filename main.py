"""
Panorama Stitcher


Dependencies: PyQt6, opencv-python, numpy
Install: pip install PyQt6 opencv-python numpy

Layout:
  Left   - image list with thumbnails
  Center - canvas (crop preview, draggable images), auto-align + stitch buttons
  Right  - transform controls (spinboxes decompose H for manual editing), layer order
"""

import sys
import os
import math
from PIL import Image
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QLabel, QSpinBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsItem, QStatusBar
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPen, QBrush, QColor, QPainter, QIcon
)
from PyQt6.QtCore import QEvent, Qt, QSize, QLocale

import functions.image_block as ib
import functions.coordinate_transforms as ct
import functions.auto_align as aa

QLocale.setDefault(QLocale.c())


# canvas view: pinch/pan gestures, ctrl+scroll zoom, keyboard modal tools
class CanvasView(QGraphicsView):
    def __init__(self, scene, window):
        super().__init__(scene)
        self.main_window = window
        self.tools = ToolController(self, window)

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )

        self.scale(0.5, 0.5)

        self.setBackgroundBrush(QBrush(QColor("#111118")))
        self.setMinimumSize(600, 600)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.grabGesture(Qt.GestureType.PanGesture)
        self.grabGesture(Qt.GestureType.PinchGesture)

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self._gesture_event(event)
        return super().event(event)

    def _gesture_event(self, event):
        if g := event.gesture(Qt.GestureType.PanGesture):
            d = g.delta()
            self.translate(d.x(), d.y())
        if g := event.gesture(Qt.GestureType.PinchGesture):
            self.scale(g.scaleFactor(), g.scaleFactor())
        return True

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.0015 ** event.angleDelta().y()
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if self.tools.active_tool:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.tools.finish(commit=True)
                return
            if key == Qt.Key.Key_Escape:
                self.tools.finish(commit=False)
                return
        if key == Qt.Key.Key_G:
            self.tools.start("move")
        elif key == Qt.Key.Key_R:
            self.tools.start("rotate")
        elif key == Qt.Key.Key_S:
            self.tools.start("scale")
        else:
            super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        if self.tools.update(event):
            return
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if self.tools.active_tool:
            if event.button() == Qt.MouseButton.LeftButton:
                self.tools.finish(commit=True)
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.tools.finish(commit=False)
                return

        item = self.itemAt(event.pos())

        # clicked empty space → select camera
        if item is None:
            self.main_window._select_camera()
            self.main_window.selected_entry = self.main_window.camera
            return

        super().mousePressEvent(event)


# modal keyboard tools (G/R/S): compose directly onto H so perspective is preserved
class ToolController:
    def __init__(self, canvas, main_window):
        self.canvas = canvas
        self.main = main_window
        self.active_tool = None
        self.entry = None
        self._start_mouse = None
        self._start_H = None

    def start(self, tool):
        if not self.main.selected_entry:
            return
        if self.active_tool:
            self.finish(commit=False)
        self.active_tool = tool
        self.entry = self.main.selected_entry
        self._start_mouse = self.canvas.mapToScene(
            self.canvas.mapFromGlobal(self.canvas.cursor().pos())
        )
        self._start_H = self.entry.H.copy()

    def update(self, mouse_event):
        if not self.active_tool or not self.entry:
            return False

        cur = self.canvas.mapToScene(mouse_event.pos())
        delta = cur - self._start_mouse
        H0 = self._start_H

        if self.active_tool == "move":
            H = H0.copy()
            H[0, 2] += delta.x()
            H[1, 2] += delta.y()
            self.entry.H = H

        elif self.active_tool == "rotate":
            # rotate in image-local space around image center, then apply existing H
            angle_rad = math.radians(delta.x() * 0.3)
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            cx = self.entry.pixmap.width() / 2
            cy = self.entry.pixmap.height() / 2
            R = np.array([
                [c, -s, cx * (1 - c) + cy * s],
                [s,  c, cy * (1 - c) - cx * s],
                [0,  0, 1]
            ], dtype=np.float64)
            self.entry.H = H0 @ R

        elif self.active_tool == "scale":
            factor = max(0.001, 1 + delta.x() * 0.002)
            cx = self.entry.pixmap.width() / 2
            cy = self.entry.pixmap.height() / 2
            S = np.array([
                [factor, 0, cx * (1 - factor)],
                [0, factor, cy * (1 - factor)],
                [0, 0, 1]
            ], dtype=np.float64)
            self.entry.H = H0 @ S

        self.main._apply_all_transforms()
        self.main._sync_spinboxes_from_H(self.entry.H)
        return True

    def finish(self, commit=True):
        if not self.entry:
            return
        if not commit:
            self.entry.H = self._start_H
            self.main._apply_all_transforms()
            self.main._sync_spinboxes_from_H(self.entry.H)
        self.active_tool = None
        self.entry = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panorama Stitcher")
        self.setMinimumSize(1100, 720)
        self.images = []
        self._updating_ui = False
        self._build_ui()
        self.selected_entry = self.camera
        self._apply_styles()
    
    def _create_camera_entry(self):
        # Create a proper ImageEntry for the camera/canvas
        self.camera = ib.ImageEntry.__new__(ib.ImageEntry)
        self.camera.path = None
        self.camera.name = "Canvas"
        self.camera.pixmap = QPixmap(800, 600)
        self.camera.pixmap.fill(QColor("#2a2a45"))
        
        # Draw grid pattern for visibility
        painter = QPainter(self.camera.pixmap)
        painter.setPen(QPen(QColor("#555870"), 1, Qt.PenStyle.DotLine))
        for i in range(0, 800, 50):
            painter.drawLine(i, 0, i, 600)
        for j in range(0, 600, 50):
            painter.drawLine(0, j, 800, j)
        painter.end()
        
        self.camera.thumb = self.camera.pixmap.scaled(
            52, 52,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera.H = np.eye(3, dtype=np.float64)
        self.camera.scene_item = None  # Will be set after scene is created
        self.camera.is_camera = True

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._build_left(), 0)
        layout.addWidget(self._build_center(), 1)
        layout.addWidget(self._build_right(), 0)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def _build_left(self):
        panel = QWidget()
        panel.setFixedWidth(210)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        vbox.addWidget(self._section_label("Images"))

        self.img_list = QListWidget()
        self.img_list.setIconSize(QSize(52, 52))
        self.img_list.setSpacing(2)
        self.img_list.currentRowChanged.connect(self._on_list_row_changed)
        vbox.addWidget(self.img_list, 1)

        btn_add = QPushButton("＋  Add Images")
        btn_add.clicked.connect(self._add_images)
        btn_remove = QPushButton("✕  Remove Selected")
        btn_remove.clicked.connect(self._remove_selected)
        vbox.addWidget(btn_add)
        vbox.addWidget(btn_remove)
        return panel

    def _build_center(self):
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(8)

        self.scene = QGraphicsScene()
        self._create_camera_entry()
        
        # Create camera scene item
        self.camera.scene_item = ib.ImageItem(self.camera, self._on_canvas_select, self._on_item_moved)
        self.camera.scene_item.setZValue(-9999)
        self.scene.addItem(self.camera.scene_item)
        
        self._update_canvas_rect()

        self.canvas = CanvasView(self.scene, self)
        vbox.addWidget(self.canvas, 1)

        btns = QHBoxLayout()
        align_btn = QPushButton("Auto Align")
        align_btn.setObjectName("alignBtn")
        align_btn.clicked.connect(self._auto_align)

        stitch_btn = QPushButton("Stitch and Export")
        stitch_btn.setObjectName("stitchBtn")
        stitch_btn.clicked.connect(self._stitch_and_export)

        btns.addWidget(align_btn)
        btns.addWidget(stitch_btn)
        vbox.addLayout(btns)
        return container

    def _update_canvas_rect(self):
        w, h = self.camera.pixmap.width(), self.camera.pixmap.height()
        self.scene.setSceneRect(-w * 4, -h * 4, w * 9, h * 9)

        if hasattr(self, "crop_border"):
            self.scene.removeItem(self.crop_border)
        self.crop_border = QGraphicsRectItem(0, 0, w, h)
        self.crop_border.setPen(QPen(QColor("#e94560"), 2, Qt.PenStyle.DashLine))
        self.crop_border.setBrush(QBrush(QColor(0, 0, 0, 0)))
        self.crop_border.setZValue(9999)
        self.crop_border.setTransform(ct.h_to_qt(self.camera.H))
        self.scene.addItem(self.crop_border)

    def _build_right(self):
        panel = QWidget()
        panel.setFixedWidth(210)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        vbox.addWidget(self._section_label("Canvas"))

        self.spin_canvas_w = self._make_spin(64, 10000, 10, value=self.camera.pixmap.width())
        self.spin_canvas_h = self._make_spin(64, 10000, 10, value=self.camera.pixmap.height())
        self.spin_canvas_w.valueChanged.connect(self._on_canvas_size_changed)
        self.spin_canvas_h.valueChanged.connect(self._on_canvas_size_changed)

        form_canvas = QFormLayout()
        form_canvas.addRow("Width", self.spin_canvas_w)
        form_canvas.addRow("Height", self.spin_canvas_h)
        vbox.addLayout(form_canvas)
        
        # Add canvas background image button
        btn_load_canvas = QPushButton("Load Canvas Background")
        btn_load_canvas.clicked.connect(self._load_canvas_image)
        vbox.addWidget(btn_load_canvas)

        # Bake canvas transform button
        btn_bake_camera = QPushButton("Bake Canvas Transform")
        btn_bake_camera.clicked.connect(self._bake_canvas_transform)
        vbox.addWidget(btn_bake_camera)

        vbox.addWidget(self._section_label("Transform"))

        hint = QLabel("(G) move  (R) rotate  (S) scale\nEnter=confirm  Esc=cancel")
        hint.setObjectName("hintLabel")
        vbox.addWidget(hint)

        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setSpacing(8)
        form.setContentsMargins(0, 4, 0, 4)

        self.spin_x     = self._make_double_spin(-9999, 9999, 1.0)
        self.spin_y     = self._make_double_spin(-9999, 9999, 1.0)
        self.spin_scale = self._make_double_spin(0.01, 20.0, 0.05, value=1.0)
        self.spin_rot   = self._make_double_spin(-360, 360, 1.0)

        for spin in (self.spin_x, self.spin_y, self.spin_scale, self.spin_rot):
            spin.valueChanged.connect(self._on_spinbox_changed)

        form.addRow("X", self.spin_x)
        form.addRow("Y", self.spin_y)
        form.addRow("Scale", self.spin_scale)
        form.addRow("Rotation°", self.spin_rot)
        vbox.addWidget(form_host)

        btn_reset = QPushButton("Reset Transform")
        btn_reset.clicked.connect(self._reset_transform)
        vbox.addWidget(btn_reset)

        vbox.addStretch()

        vbox.addWidget(self._section_label("Layer Order"))
        btn_up = QPushButton("↑  Move Forward")
        btn_up.clicked.connect(self._layer_up)
        btn_dn = QPushButton("↓  Move Backward")
        btn_dn.clicked.connect(self._layer_down)
        vbox.addWidget(btn_up)
        vbox.addWidget(btn_dn)

        return panel

    def _section_label(self, text):
        lbl = QLabel(text.upper())
        lbl.setObjectName("sectionLabel")
        return lbl

    def _make_spin(self, lo, hi, step, value=0):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    def _make_double_spin(self, lo, hi, step, value=0.0):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        for path in paths:
            entry = ib.ImageEntry(path)
            item = ib.ImageItem(entry, self._on_canvas_select, self._on_item_moved)
            entry.scene_item = item
            self.scene.addItem(item)
            self.images.append(entry)
            self.img_list.addItem(QListWidgetItem(QIcon(entry.thumb), entry.name))
            self._select(entry)
            self._sync_z_order()
            self._reset_transform()

    def _remove_selected(self):
        row = self.img_list.currentRow()
        if row < 0 or row >= len(self.images):
            return
        entry = self.images.pop(row)
        self.scene.removeItem(entry.scene_item)
        self.img_list.takeItem(row)
        self.selected_entry = None
        self._clear_spinboxes()

    def _on_canvas_size_changed(self):
        # Create new pixmap with grid
        new_pixmap = QPixmap(
            int(self.spin_canvas_w.value()),
            int(self.spin_canvas_h.value())
        )
        new_pixmap.fill(QColor("#2a2a45"))
        
        painter = QPainter(new_pixmap)
        painter.setPen(QPen(QColor("#555870"), 1, Qt.PenStyle.DotLine))
        w, h = new_pixmap.width(), new_pixmap.height()
        for i in range(0, w, 50):
            painter.drawLine(i, 0, i, h)
        for j in range(0, h, 50):
            painter.drawLine(0, j, w, j)
        painter.end()
        
        self.camera.pixmap = new_pixmap
        self.camera.scene_item.setPixmap(new_pixmap)
        self._update_canvas_rect()

    def _load_canvas_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Canvas Background Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if path:
            try:
                img = ib.load_image_rgba(path)
                if img is None:
                    self.status.showMessage("Failed to load image", 3000)
                    return

                # Resize if too large
                h, w = img.shape[:2]
                max_dim = 2000
                if max(h, w) > max_dim:
                    sc = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

                h, w = img.shape[:2]
                qimg = QImage(img.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
                self.camera.pixmap = QPixmap.fromImage(qimg)
                self.camera.scene_item.setPixmap(self.camera.pixmap)

                self.spin_canvas_w.setValue(w)
                self.spin_canvas_h.setValue(h)
                self._update_canvas_rect()

                self.status.showMessage(f"Loaded canvas background: {os.path.basename(path)}", 3000)

            except Exception as e:
                self.status.showMessage(f"Failed to load canvas image: {str(e)}", 4000)

    def _on_list_row_changed(self, row):
        if 0 <= row < len(self.images):
            self._select(self.images[row])

    def _on_canvas_select(self, entry):
        if entry is self.camera:
            self._select_camera()
        else:
            self._select(entry, list_row=self.images.index(entry))

    def _on_item_moved(self, entry):
        if entry is self.camera:
            # Update crop border position when camera is moved
            if hasattr(self, 'crop_border'):
                self.crop_border.setTransform(ct.h_to_qt(entry.H))
            self._apply_all_transforms()
        self._sync_spinboxes_from_H(entry.H)

    def _select(self, entry, list_row=None):
        self.selected_entry = entry

        for img in self.images:
            img.scene_item.setOpacity(0.45 if img is not entry else 1.0)

        self.camera.scene_item.setOpacity(0.45 if entry is not self.camera else 1.0)

        if list_row is not None:
            self.img_list.blockSignals(True)
            self.img_list.setCurrentRow(list_row)
            self.img_list.blockSignals(False)

        self._sync_spinboxes_from_H(entry.H)
        self.canvas.setFocus()

    def _select_camera(self):
        self.selected_entry = self.camera
        self.img_list.clearSelection()

        # Dim all images, brighten camera
        for img in self.images:
            img.scene_item.setOpacity(1.0)
        self.camera.scene_item.setOpacity(1.0)

        # Update spinboxes
        self._sync_spinboxes_from_H(self.camera.H)

    def _sync_spinboxes_from_H(self, H):
        tx, ty, scale, rot = ct.decompose_H(H)
        self._updating_ui = True
        self.spin_x.setValue(tx)
        self.spin_y.setValue(ty)
        self.spin_scale.setValue(scale)
        self.spin_rot.setValue(rot)
        self._updating_ui = False

    def _on_spinbox_changed(self):
        if self._updating_ui:
            return
        # rebuilds as similarity H - clears any perspective from auto-align
        H = ct.build_similarity_H(
            self.spin_x.value(),
            self.spin_y.value(),
            self.spin_scale.value(),
            self.spin_rot.value()
        )
        self.selected_entry.H = H
        if self.selected_entry is self.camera:
            self._apply_all_transforms()
        else:
            self._apply_H_to_item(self.selected_entry)

    def _apply_H_to_item(self, entry):
        entry.scene_item.setTransform(ct.h_to_qt(entry.H))
        # Also update crop border
        if entry is self.camera and hasattr(self, 'crop_border'):
            self.crop_border.setTransform(ct.h_to_qt(entry.H))


    def _apply_all_transforms(self):
        # Apply camera transform to itself
        self.camera.scene_item.setTransform(ct.h_to_qt(self.camera.H))
        # Update crop border
        if hasattr(self, 'crop_border'):
            self.crop_border.setTransform(ct.h_to_qt(self.camera.H))
        # Apply combined transform to other images
        for e in self.images:
            e.scene_item.setTransform(ct.h_to_qt(e.H))

    def _reset_transform(self):
        if not self.selected_entry:
            return
        e = self.selected_entry

        if e is self.camera:
            e.H = np.eye(3, dtype=np.float64)
            self._apply_all_transforms()
            self._sync_spinboxes_from_H(e.H)
            return
        
        cx = (self.camera.pixmap.width() - e.pixmap.width()) / 2
        cy = (self.camera.pixmap.height() - e.pixmap.height()) / 2

        e.H = ct.build_similarity_H(cx, cy, 1.0, 0.0)

        self._apply_H_to_item(e)
        self._sync_spinboxes_from_H(e.H)
    
    def _bake_canvas_transform(self):
        if not self.images:
            return

        # compose each image's H with the inverse of the camera H
        view = np.linalg.inv(self.camera.H)
        for entry in self.images:
            entry.H = view @ entry.H
            self._apply_H_to_item(entry)

        # reset camera to identity
        self.camera.H = np.eye(3, dtype=np.float64)
        self._apply_all_transforms()

        if self.selected_entry:
            self._sync_spinboxes_from_H(self.selected_entry.H)

        self.status.showMessage("Canvas transform baked into all images.", 3000)

    def _clear_spinboxes(self):
        self._updating_ui = True
        self.spin_x.setValue(0)
        self.spin_y.setValue(0)
        self.spin_scale.setValue(1.0)
        self.spin_rot.setValue(0)
        self._updating_ui = False

    def _layer_up(self):
        row = self.img_list.currentRow()
        if row <= 0:
            return
        self.images[row], self.images[row - 1] = self.images[row - 1], self.images[row]
        item = self.img_list.takeItem(row)
        self.img_list.insertItem(row - 1, item)
        self.img_list.setCurrentRow(row - 1)
        self._sync_z_order()

    def _layer_down(self):
        row = self.img_list.currentRow()
        if row < 0 or row >= len(self.images) - 1:
            return
        self.images[row], self.images[row + 1] = self.images[row + 1], self.images[row]
        item = self.img_list.takeItem(row)
        self.img_list.insertItem(row + 1, item)
        self.img_list.setCurrentRow(row + 1)
        self._sync_z_order()

    def _sync_z_order(self):
        enum = enumerate(self.images[::-1])  # reverse so top of list = top layer
        for i, entry in enum:
            entry.scene_item.setZValue(i)

    # load grayscale at display pixmap resolution so keypoint coords match H-space
    def _load_gray_display(self, entry):
        img = ib.load_image_rgba(entry.path)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        return cv2.resize(
            img, (entry.pixmap.width(), entry.pixmap.height()),
            interpolation=cv2.INTER_AREA
        )

    def _auto_align(self):
        aa.auto_align(self)

    def _stitch_and_export(self):
        if not self.images:
            return

        self.status.showMessage("Stitching at full resolution...")
        QApplication.processEvents()

        canvas = np.zeros((self.camera.pixmap.height(), self.camera.pixmap.width(), 4), dtype=np.uint8)

        for entry in self.images[::-1]:  # reverse so top of list = top layer
            img = ib.load_image_rgba(entry.path)

            orig_h, orig_w = img.shape[:2]
            disp_w = entry.pixmap.width()
            disp_h = entry.pixmap.height()

            # entry.H is in display coords; compose with S that maps orig -> display
            # so H_full = entry.H @ S maps original pixel coords -> canvas coords
            S = np.diag([disp_w / orig_w, disp_h / orig_h, 1.0])
            view = np.linalg.inv(self.camera.H)
            H_full = view @ entry.H @ S

            warped = cv2.warpPerspective(
                img, H_full,
                (self.camera.pixmap.width(), self.camera.pixmap.height()),
                flags=cv2.INTER_LANCZOS4
            )

            # "over" alpha composite
            src_a = warped[:, :, 3:4].astype(np.float32) / 255.0
            dst_a = canvas[:, :, 3:4].astype(np.float32) / 255.0
            out_a = src_a + dst_a * (1.0 - src_a)

            for c in range(3):
                num = (warped[:, :, c] * src_a[:, :, 0]
                       + canvas[:, :, c] * dst_a[:, :, 0] * (1.0 - src_a[:, :, 0]))
                canvas[:, :, c] = np.where(
                    out_a[:, :, 0] > 0, num / out_a[:, :, 0], 0
                ).astype(np.uint8)
            canvas[:, :, 3] = (out_a[:, :, 0] * 255).astype(np.uint8)

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Panorama", "panorama.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if save_path:
            out_bgra = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGRA)
            if save_path.lower().endswith((".jpg", ".jpeg")):
                out_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(save_path, out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(save_path, out_bgra)
            self.status.showMessage(f"Saved: {save_path}", 6000)

    def _apply_styles(self):
        self.setStyleSheet("""
            * {
                font-family: 'Segoe UI', Helvetica, sans-serif;
                font-size: 13px;
                color: #dde0e8;
            }
            QMainWindow, QWidget {
                background: #10101c;
            }
            QLabel#sectionLabel {
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 2px;
                color: #555870;
                padding: 2px 0 4px 0;
                border-bottom: 1px solid #23233a;
            }
            QLabel#hintLabel {
                font-size: 11px;
                color: #3d3d5c;
                padding: 2px 0;
            }
            QListWidget {
                background: #14141f;
                border: 1px solid #21213a;
                border-radius: 5px;
                padding: 2px;
            }
            QListWidget::item {
                padding: 4px 6px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background: #e94560;
                color: #fff;
            }
            QListWidget::item:hover:!selected {
                background: #1e1e30;
            }
            QPushButton {
                background: #1c1c2e;
                border: 1px solid #2a2a45;
                border-radius: 5px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background: #22223a;
                border-color: #e94560;
                color: #fff;
            }
            QPushButton:pressed {
                background: #e94560;
            }
            QPushButton#stitchBtn {
                background: #e94560;
                border: none;
                color: #fff;
                font-weight: 700;
                font-size: 14px;
                padding: 11px;
                border-radius: 6px;
                letter-spacing: 1px;
            }
            QPushButton#stitchBtn:hover {
                background: #ff5f77;
            }
            QPushButton#alignBtn {
                background: #1a3a4a;
                border: 1px solid #2a5a6a;
                color: #7dd3e8;
                font-weight: 600;
                font-size: 14px;
                padding: 11px;
                border-radius: 6px;
            }
            QPushButton#alignBtn:hover {
                background: #1e4a5e;
                border-color: #7dd3e8;
                color: #fff;
            }
            QDoubleSpinBox, QSpinBox {
                background: #14141f;
                border: 1px solid #21213a;
                border-radius: 4px;
                padding: 4px 6px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #e94560;
            }
            QStatusBar {
                background: #0c0c16;
                color: #555870;
                font-size: 11px;
                padding: 2px 8px;
            }
            QScrollBar {
                background: #14141f;
                width: 8px;
                height: 8px;
            }
            QScrollBar::handle {
                background: #2a2a45;
                border-radius: 4px;
            }
        """)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()