import time as _time
import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QTimer, Qt, Signal
import moderngl

from render.renderer import Renderer
from config.defaults import NUM_BARS
from core.layer import LayerManager, IMAGE_MODE


_HANDLE_HIT_PX = 10
_MIN_IMAGE_SIZE_PX = 12


def _image_rect_pixels(layer, width: int, height: int) -> tuple[float, float, float, float]:
    """Return image bounds in widget pixels as left, top, right, bottom."""
    return _image_rect_from_values(
        layer.x, layer.y, layer.image_width, layer.image_height, width, height)


def _image_rect_from_values(x: float, y: float, image_width: float,
                            image_height: float, width: int, height: int
                            ) -> tuple[float, float, float, float]:
    cx = (x + 1.0) * 0.5 * width
    cy = (1.0 - y) * 0.5 * height
    half_w = image_width * width * 0.5
    half_h = image_height * height * 0.5
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _image_resize_handle(layer, px: float, py: float,
                         width: int, height: int) -> str | None:
    left, top, right, bottom = _image_rect_pixels(layer, width, height)
    near_left = abs(px - left) <= _HANDLE_HIT_PX
    near_right = abs(px - right) <= _HANDLE_HIT_PX
    near_top = abs(py - top) <= _HANDLE_HIT_PX
    near_bottom = abs(py - bottom) <= _HANDLE_HIT_PX
    within_x = left - _HANDLE_HIT_PX <= px <= right + _HANDLE_HIT_PX
    within_y = top - _HANDLE_HIT_PX <= py <= bottom + _HANDLE_HIT_PX
    horizontal = "w" if near_left else "e" if near_right else ""
    vertical = "n" if near_top else "s" if near_bottom else ""
    if horizontal and vertical:
        return vertical + horizontal
    if horizontal and within_y:
        return horizontal
    if vertical and within_x:
        return vertical
    return None


def _apply_image_resize(layer, handle: str, origin: tuple, dx: float, dy: float,
                        width: int, height: int) -> None:
    """Resize one or two edges while keeping every un-dragged edge anchored."""
    _, _, old_x, old_y, old_width, old_height = origin
    left, top, right, bottom = _image_rect_from_values(
        old_x, old_y, old_width, old_height, width, height)
    if "w" in handle:
        left = min(left + dx, right - _MIN_IMAGE_SIZE_PX)
    if "e" in handle:
        right = max(right + dx, left + _MIN_IMAGE_SIZE_PX)
    if "n" in handle:
        top = min(top + dy, bottom - _MIN_IMAGE_SIZE_PX)
    if "s" in handle:
        bottom = max(bottom + dy, top + _MIN_IMAGE_SIZE_PX)
    layer.x = ((left + right) * 0.5 / max(width, 1)) * 2.0 - 1.0
    layer.y = 1.0 - ((top + bottom) * 0.5 / max(height, 1)) * 2.0
    layer.image_width = (right - left) / max(width, 1)
    layer.image_height = (bottom - top) / max(height, 1)


def _physical_size(width: int, height: int, pixel_ratio: float) -> tuple[int, int]:
    """Convert Qt logical dimensions to the actual OpenGL pixel dimensions."""
    ratio = max(float(pixel_ratio), 1.0)
    return max(round(width * ratio), 1), max(round(height * ratio), 1)


class AspectRatioContainer(QWidget):
    """Centers a child canvas at a permanent 16:9 aspect ratio."""

    def __init__(self, child: QWidget, parent=None):
        super().__init__(parent)
        self.child = child
        child.setParent(self)

    def resizeEvent(self, event):
        width, height = self.width(), self.height()
        target_height = round(width * 9 / 16)
        if target_height <= height:
            child_width, child_height = width, target_height
        else:
            child_height = height
            child_width = round(height * 16 / 9)
        self.child.setGeometry((width - child_width) // 2,
                               (height - child_height) // 2,
                               child_width, child_height)
        super().resizeEvent(event)


class PreviewWidget(QOpenGLWidget):
    """OpenGL preview driven by a LayerManager composition.

    The MainWindow owns the LayerManager and sets it via set_layer_manager().
    Dragging moves the currently selected layer.
    """

    layer_moved = Signal()   # emitted after a drag so the UI can refresh

    def __init__(self, parent=None):
        super().__init__(parent)
        self._renderer: Renderer | None = None
        self._lm: LayerManager | None = None
        self._bars = np.zeros(NUM_BARS, dtype=np.float32)
        self._pulse = 0.0
        self._bg_image_path: str | None = None
        self._center_image_path: str | None = None
        self._start_time = _time.perf_counter()

        self._dragging = False
        self._drag_origin = None        # (mouse_x, mouse_y, layer_x, layer_y)
        self._resize_handle: str | None = None
        self._show_selection = True
        self._pending_release: list[int] = []   # layer ids to free on next paint

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(1000 // 60)
        self.setMouseTracking(True)

    # ── Public API ───────────────────────────────────────────────
    def set_layer_manager(self, lm: LayerManager):
        self._lm = lm

    def update_audio_data(self, bars: np.ndarray, pulse: float):
        self._bars = bars
        self._pulse = pulse
        self.update()

    # ── GL lifecycle ─────────────────────────────────────────────
    def initializeGL(self):
        self._ctx = moderngl.create_context()
        w, h = _physical_size(self.width(), self.height(), self.devicePixelRatioF())
        self._renderer = Renderer(w, h, ctx=self._ctx)
        if self._bg_image_path:
            self._renderer.load_background(self._bg_image_path)
        if self._center_image_path:
            self._renderer.load_center_image(self._center_image_path)

    def resizeGL(self, w: int, h: int):
        if not hasattr(self, "_ctx") or self._ctx is None:
            return
        if self._renderer:
            physical_w, physical_h = _physical_size(w, h, self.devicePixelRatioF())
            self._renderer.resize(physical_w, physical_h)

    def paintGL(self):
        if not self._renderer or not self._lm:
            return
        # Free states for removed layers now that the GL context is current
        while self._pending_release:
            self._renderer.release_layer_state(self._pending_release.pop())
        t = _time.perf_counter() - self._start_time
        self._renderer.render_composition(self._lm, self._bars, self._pulse, time=t)

        sel = self._lm.selected()
        if sel is not None and self._show_selection:
            self._renderer.draw_selection_outline(sel)

        qt_fbo = self._ctx.detect_framebuffer(self.defaultFramebufferObject())
        self._ctx.copy_framebuffer(dst=qt_fbo, src=self._renderer.fbo)

    # ── Image loading (global bg + center) ───────────────────────
    def load_background(self, path: str):
        self._bg_image_path = path
        if self._renderer:
            self._renderer.load_background(path)
        self.update()

    def clear_background(self):
        self._bg_image_path = None
        if self._renderer:
            self._renderer.clear_background()
        self.update()

    def load_center_image(self, path: str):
        self._center_image_path = path
        if self._renderer:
            self._renderer.load_center_image(path)
        self.update()

    def clear_center_image(self):
        self._center_image_path = None
        if self._renderer:
            self._renderer.clear_center_image()
        self.update()

    def release_layer_state(self, layer_id: int):
        self._pending_release.append(layer_id)

    # ── Mouse drag → move selected layer ─────────────────────────
    def mousePressEvent(self, event):
        if not self._lm or self._lm.selected() is None:
            return
        if event.button() == Qt.LeftButton:
            layer = self._lm.selected()
            self._dragging = True
            self._drag_origin = (event.position().x(), event.position().y(),
                                 layer.x, layer.y,
                                 getattr(layer, "image_width", 0.0),
                                 getattr(layer, "image_height", 0.0))
            self._resize_handle = (
                _image_resize_handle(layer, event.position().x(), event.position().y(),
                                     self.width(), self.height())
                if layer.mode == IMAGE_MODE else None)
            if layer.mode == IMAGE_MODE and not self._resize_handle:
                left, top, right, bottom = _image_rect_pixels(
                    layer, self.width(), self.height())
                px, py = event.position().x(), event.position().y()
                if not (left <= px <= right and top <= py <= bottom):
                    self._dragging = False
                    self._drag_origin = None

    def mouseMoveEvent(self, event):
        if not self._lm:
            return
        layer = self._lm.selected()
        if layer is None:
            return
        if not self._dragging:
            self._update_resize_cursor(layer, event.position().x(), event.position().y())
            return
        mx0, my0, lx0, ly0, _, _ = self._drag_origin
        dx = event.position().x() - mx0
        dy = event.position().y() - my0
        if layer.mode == IMAGE_MODE and self._resize_handle:
            _apply_image_resize(layer, self._resize_handle, self._drag_origin,
                                dx, dy, self.width(), self.height())
        else:
            # Pixel delta → NDC (y inverted)
            layer.x = lx0 + 2.0 * dx / max(self.width(), 1)
            layer.y = ly0 - 2.0 * dy / max(self.height(), 1)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self._resize_handle = None
            self.layer_moved.emit()

    def _update_resize_cursor(self, layer, px: float, py: float) -> None:
        if layer.mode != IMAGE_MODE:
            self.unsetCursor()
            return
        handle = _image_resize_handle(layer, px, py, self.width(), self.height())
        cursors = {
            "w": Qt.SizeHorCursor, "e": Qt.SizeHorCursor,
            "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
            "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
            "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
        }
        if handle:
            self.setCursor(cursors[handle])
        else:
            left, top, right, bottom = _image_rect_pixels(
                layer, self.width(), self.height())
            self.setCursor(Qt.SizeAllCursor if left <= px <= right and top <= py <= bottom
                           else Qt.ArrowCursor)
