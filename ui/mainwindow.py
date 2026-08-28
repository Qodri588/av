import math
import copy
import time
import numpy as np
import sounddevice as sd
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
    QProgressBar, QGroupBox, QFormLayout, QSizePolicy, QSpinBox, QCheckBox,
    QColorDialog, QListWidget, QListWidgetItem, QAbstractItemView, QScrollArea,
    QTextEdit,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor

from core.audio import AudioFile
from core.fft import FFTProcessor
from core.template import save_template, load_template
from core.layer import (
    Layer, LayerManager, MAX_VISUAL_LAYERS, MAX_TEXT_LAYERS, TEXT_MODE, IMAGE_MODE,
    PARTICLE_MODE,
    BLEND_ADDITIVE, BLEND_NORMAL,
)
from ui.preview import PreviewWidget, AspectRatioContainer
from config.defaults import (
    NUM_BARS, MAX_BAR_HEIGHT, SMOOTHING_DECAY, PULSE_INTENSITY,
    SENSITIVITY, FPS, RESOLUTIONS, FFT_SIZE, PALETTES,
    VIZ_TYPES_VISIBLE, DEFAULT_VIZ_TYPE, VISUAL_PRESETS,
    FREQ_BASS_SPLIT, FREQ_BASS_SPLIT_HZ, CQT_BINS_PER_OCTAVE,
    HALO_SINE_R_BASE, HALO_SINE_AMPLITUDE, HALO_SINE_N_POINTS,
    HALO_SINE_GLOW_LAYERS, HALO_SINE_SMOOTHING_DECAY, HALO_SINE_FILL_OPACITY,
    HALO_SINE_SPLINE_GAP, HALO_SINE_PIXEL_SIZE,
    TUNNEL_SIDES, TUNNEL_RINGS, TUNNEL_SPEED, TUNNEL_KICK_ZOOM, TUNNEL_CHROMA,
    TUNNEL_KICK_SENSITIVITY, TUNNEL_BASS_SPEED, TUNNEL_KICK_MODE,
    TUNNEL_KICK_FREQ_LO, TUNNEL_KICK_FREQ_HI,
    TUNNEL_KICK_THRESHOLD, TUNNEL_KICK_COOLDOWN,
    NUKE_KICK_THRESHOLD, NUKE_SPEED, NUKE_LIFE, NUKE_WIDTH, NUKE_FLASH, NUKE_BG,
    VOID_SPEED, VOID_PULL, VOID_RAYS, VOID_ARMS,
    BG_PULSE_INTENSITY, FLASH_INTENSITY, CIRCLE_RADIUS_RATIO, BAR_WIDTH,
    BAR_SHAPES,
)

FFT_ANALYSIS_BARS = 256   # global analysis resolution; layers resample from this

_RADIAL_MODES     = {0, 4, 5, 6, 7}
_ROTATION_MODES   = {0, 4, 5, 6, 7}
_BAR_MODES        = {0, 1, 2, 4, 5, 7}
_BAR_SHAPE_MODES  = {1, 2}
_HEIGHT_MODES     = {0, 1, 2, 4, 5, 7}
_HALO_MODES       = {4, 5, 6, 7, 8}
_HALO_SINE_MODES  = {6, 9}
_TUNNEL_MODES     = {8}
_MIRROR_MODES     = {0, 4}
_NUKE_MODES       = {10}
_VOID_MODES       = {11}

_AMP_LABELS  = ["Silence", "Low", "Medium", "High", "Saturation"]
_FREQ_LABELS = ["Sub-bass", "Bass", "Midrange", "Presence", "Treble"]
_PERSO_AMP   = "Custom Amplitude"
_PERSO_FREQ  = "Custom Frequency"

_TEXT_FONTS = {
    "Arial": "arial.ttf",
    "Times New Roman": "times.ttf",
    "Courier New": "cour.ttf",
    "Verdana": "verdana.ttf",
    "Georgia": "georgia.ttf",
    "Impact": "impact.ttf",
    "Comic Sans MS": "comic.ttf",
}


class CollapsibleGroupBox(QGroupBox):
    """Native, keyboard-accessible section that occupies one row when collapsed."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.toggled.connect(self._set_expanded)
        self.setChecked(expanded)
        self._set_expanded(expanded)

    def _set_expanded(self, expanded: bool):
        self.setMaximumHeight(16777215 if expanded else 30)
        self.setSizePolicy(QSizePolicy.Preferred,
                           QSizePolicy.Preferred if expanded else QSizePolicy.Fixed)


class ExportWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(str)
    error    = Signal(str)

    def __init__(self, audio, layer_manager, output_dir, resolution, fps,
                 smoothing_decay, bass_split, bass_split_hz,
                 use_cqt, bins_per_octave,
                 bg_image_path, center_image_path):
        super().__init__()
        self._audio = audio
        self._lm = layer_manager
        self._kwargs = dict(
            output_dir=output_dir, resolution=resolution, fps=fps,
            smoothing_decay=smoothing_decay,
            bass_split=bass_split, bass_split_hz=bass_split_hz,
            use_cqt=use_cqt, bins_per_octave=bins_per_octave,
            bg_image_path=bg_image_path, center_image_path=center_image_path,
        )

    def run(self):
        try:
            from export.ffmpeg_export import FFmpegExporter
            exporter = FFmpegExporter(
                audio=self._audio,
                layer_manager=self._lm,
                progress_cb=lambda cur, tot: self.progress.emit(cur, tot),
                **self._kwargs,
            )
            out = exporter.export()
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e))


_THREAD_STOP_TIMEOUT_MS = 2000


class AudioPlaybackThread(QThread):
    frame_ready      = Signal(object, float)
    position_changed = Signal(int, int)

    def __init__(self, audio: AudioFile, fft: FFTProcessor, fps: int,
                 start_frame: int = 0, volume: float = 1.0):
        super().__init__()
        self._audio       = audio
        self._fft         = fft
        self._fps         = fps
        self._running     = False
        self._start_frame = start_frame
        self._volume      = volume

    def update_fft(self, fft: FFTProcessor):
        self._fft = fft

    def run(self):
        self._running = True
        hop   = max(1, int(self._audio.sr / self._fps))
        total = int(len(self._audio.mono) / hop)

        start_sample = self._start_frame * hop
        audio_out    = self._audio.mono[start_sample:] * self._volume
        sd.play(audio_out, samplerate=self._audio.sr)

        t0 = time.perf_counter()
        i  = self._start_frame
        while self._running and i < total:
            samples = self._audio.get_frame_samples(i, hop, FFT_SIZE)
            bars, pulse = self._fft.process(samples)
            self.frame_ready.emit(bars, pulse)
            self.position_changed.emit(i, total)

            # The next frame is whichever one the elapsed time points at, not i+1: a slow
            # render drops frames instead of pushing the whole preview behind the audio.
            # int(1000/fps) also truncates (16 ms for 60 fps, not 16.667), which on its own
            # walked the preview 3% off the audio clock over a track.
            # Target the next boundary strictly ahead of now (+1): when rendering runs
            # late this still yields for the rest of the frame instead of spinning with
            # no sleep at all, which would starve the GUI thread of the GIL.
            elapsed = time.perf_counter() - t0
            nxt     = max(i + 1, self._start_frame + int(elapsed * self._fps) + 1)
            delay   = (nxt - self._start_frame) / self._fps - elapsed
            if delay > 0:
                self.msleep(max(1, int(delay * 1000)))
            i = nxt
        sd.stop()

    def stop(self):
        self._running = False
        sd.stop()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer")
        self.resize(1280, 760)

        self._audio: AudioFile | None = None
        self._fft:   FFTProcessor | None = None
        self._playback_thread: AudioPlaybackThread | None = None
        self._export_thread:   QThread | None = None
        self._bg_image_path:     str | None = None
        self._bg_text_color = (1.0, 1.0, 1.0)
        self._bg_text_stroke_color = (0.0, 0.0, 0.0)
        self._bg_text_shadow_color = (0.0, 0.0, 0.0)
        self._bg_text_font = "arial.ttf"
        self._center_image_path: str | None = None
        self._template_path: str | None = None
        self._volume: float    = 1.0
        self._seek_frame: int  = 0
        self._seeking: bool    = False
        self._loading: bool    = False    # guards UI→layer writes during load

        self._lm = LayerManager()
        self._lm.add_layer(mode=DEFAULT_VIZ_TYPE)

        # Editing buffers for the custom palettes of the selected layer
        _default = [list(c) for c in list(PALETTES.values())[0]]
        self._custom_palette_amp  = [list(c) for c in _default]
        self._custom_palette_freq = [list(c) for c in _default]

        self._build_ui()
        self._preview.set_layer_manager(self._lm)
        self._refresh_layer_list()
        self._load_layer_into_ui(self._lm.selected())

    # ── UI builders ──────────────────────────────────────────────
    def _make_slider_row(self, layout: QFormLayout, label: str,
                         lo: int, hi: int, val: int, callback=None) -> tuple:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        sl = QSlider(Qt.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(val)
        sl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(val)
        sb.setButtonSymbols(QSpinBox.NoButtons)
        sb.setMinimumWidth(90)
        sb.setFixedWidth(90)

        sl.valueChanged.connect(sb.setValue)
        sb.valueChanged.connect(sl.setValue)
        if callback:
            sl.valueChanged.connect(callback)

        h.addWidget(sl)
        h.addWidget(sb)

        lbl = QLabel(label)
        layout.addRow(lbl, container)
        return sl, lbl, container

    def _set_row_visible(self, row: tuple, visible: bool):
        row[1].setVisible(visible)
        row[2].setVisible(visible)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        root.addWidget(scroll)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setAlignment(Qt.AlignTop)
        scroll.setWidget(panel)

        # ── Audio ──
        ag = QGroupBox("Audio")
        al = QVBoxLayout(ag)
        self._file_label = QLabel("No file selected")
        self._file_label.setWordWrap(True)
        btn_open = QPushButton("Open audio file…")
        btn_open.clicked.connect(self._open_file)
        al.addWidget(self._file_label)
        al.addWidget(btn_open)
        pl.addWidget(ag)

        tg = QGroupBox("Template")
        template_layout = QHBoxLayout(tg)
        btn_save_template = QPushButton("Save…")
        btn_save_template.clicked.connect(self._save_template)
        btn_load_template = QPushButton("Load…")
        btn_load_template.clicked.connect(self._load_template)
        template_layout.addWidget(btn_save_template)
        template_layout.addWidget(btn_load_template)
        pl.addWidget(tg)

        # ── Layers ──
        lg = QGroupBox("Layers")
        lgl = QVBoxLayout(lg)
        self._layer_list = QListWidget()
        self._layer_list.setDragDropMode(QAbstractItemView.InternalMove)
        self._layer_list.setMaximumHeight(110)
        self._layer_list.currentRowChanged.connect(self._on_layer_selected)
        self._layer_list.model().rowsMoved.connect(self._on_layers_reordered)
        lgl.addWidget(self._layer_list)

        lbtns = QHBoxLayout()
        btn_add = QPushButton("＋")
        btn_add.clicked.connect(self._add_layer)
        btn_add_text = QPushButton("T+")
        btn_add_text.setToolTip("Add text layer")
        btn_add_text.clicked.connect(self._add_text_layer)
        btn_add_image = QPushButton("I+")
        btn_add_image.setToolTip("Add PNG, SVG, or image overlay")
        btn_add_image.clicked.connect(self._add_image_layer)
        btn_add_particles = QPushButton("P+")
        btn_add_particles.setToolTip("Add particle layer")
        btn_add_particles.clicked.connect(self._add_particle_layer)
        btn_del = QPushButton("🗑")
        btn_del.clicked.connect(self._remove_selected_layer)
        self._btn_layer_vis = QPushButton("👁")
        self._btn_layer_vis.clicked.connect(self._toggle_selected_visible)
        self._btn_layer_solo = QPushButton("S")
        self._btn_layer_solo.clicked.connect(self._toggle_selected_solo)
        for b in (btn_add, btn_add_text, btn_add_image, btn_add_particles, btn_del,
                  self._btn_layer_vis, self._btn_layer_solo):
            b.setFixedWidth(44)
            lbtns.addWidget(b)
        lbtns.addStretch()
        lgl.addLayout(lbtns)

    # ── Background image (BG layer) ──
        self._bg_group = CollapsibleGroupBox("Background")
        bg_l = QVBoxLayout(self._bg_group)
        self._bg_label = QLabel("No image selected")
        self._bg_label.setWordWrap(True)
        bg_row = QHBoxLayout()
        btn_bg = QPushButton("Choose…")
        btn_bg.clicked.connect(self._open_bg_image)
        self._btn_bg_clear = QPushButton("Clear")
        self._btn_bg_clear.setEnabled(False)
        self._btn_bg_clear.clicked.connect(self._clear_bg_image)
        bg_row.addWidget(btn_bg)
        bg_row.addWidget(self._btn_bg_clear)
        bg_l.addWidget(self._bg_label)
        bg_l.addLayout(bg_row)
        bg_form = QFormLayout()
        self._row_bg_opacity = self._make_slider_row(
            bg_form, "Opacity (×0.01)", 0, 100, 100, self._on_bg_changed)
        self._chk_bg_zoom = QCheckBox("Bass-reactive zoom")
        self._chk_bg_zoom.stateChanged.connect(self._on_bg_changed)
        bg_form.addRow("", self._chk_bg_zoom)
        self._row_bg_zoom_int = self._make_slider_row(
            bg_form, "Intensity (×0.01)", 1, 100, int(BG_PULSE_INTENSITY * 100),
            self._on_bg_changed)
        self._combo_bg_fit = QComboBox()
        self._combo_bg_fit.addItems(["Fit", "Cover", "Crop"])
        self._combo_bg_fit.setCurrentIndex(1)
        self._combo_bg_fit.currentIndexChanged.connect(self._on_bg_changed)
        bg_form.addRow("Media fit", self._combo_bg_fit)
        self._row_bg_crop_zoom = self._make_slider_row(
            bg_form, "Crop zoom (%)", 100, 300, 100, self._on_bg_changed)
        self._row_bg_crop_x = self._make_slider_row(
            bg_form, "Crop X", -100, 100, 0, self._on_bg_changed)
        self._row_bg_crop_y = self._make_slider_row(
            bg_form, "Crop Y", -100, 100, 0, self._on_bg_changed)
        for row in (self._row_bg_crop_zoom, self._row_bg_crop_x, self._row_bg_crop_y):
            self._set_row_visible(row, False)
        bg_l.addLayout(bg_form)

        self._text_group = CollapsibleGroupBox("Text Overlay")
        text_form = QFormLayout(self._text_group)
        self._bg_text = QTextEdit()
        self._bg_text.setPlaceholderText("Optional background text")
        self._bg_text.setMaximumHeight(80)
        self._bg_text.textChanged.connect(self._on_text_changed)
        text_form.addRow("Text", self._bg_text)
        self._row_bg_text_size = self._make_slider_row(
            text_form, "Text size", 12, 300, 72, self._on_text_changed)
        self._row_bg_text_x = self._make_slider_row(
            text_form, "Text X (%)", 0, 100, 50, self._on_text_changed)
        self._row_bg_text_y = self._make_slider_row(
            text_form, "Text Y (%)", 0, 100, 50, self._on_text_changed)
        self._row_bg_text_opacity = self._make_slider_row(
            text_form, "Text opacity (%)", 0, 100, 100, self._on_text_changed)
        self._combo_bg_text_font = QComboBox()
        self._combo_bg_text_font.addItems(list(_TEXT_FONTS) + ["Custom font file"])
        self._combo_bg_text_font.currentTextChanged.connect(self._on_bg_text_font_changed)
        text_form.addRow("Font", self._combo_bg_text_font)
        self._btn_bg_text_font = QPushButton("Choose font file…")
        self._btn_bg_text_font.clicked.connect(self._pick_bg_text_font)
        self._btn_bg_text_font.setVisible(False)
        text_form.addRow("Font file", self._btn_bg_text_font)
        font_style_row = QWidget()
        font_style_layout = QHBoxLayout(font_style_row)
        font_style_layout.setContentsMargins(0, 0, 0, 0)
        self._chk_bg_text_bold = QCheckBox("Bold")
        self._chk_bg_text_italic = QCheckBox("Italic")
        self._chk_bg_text_bold.stateChanged.connect(self._on_text_changed)
        self._chk_bg_text_italic.stateChanged.connect(self._on_text_changed)
        font_style_layout.addWidget(self._chk_bg_text_bold)
        font_style_layout.addWidget(self._chk_bg_text_italic)
        text_form.addRow("Style", font_style_row)
        self._combo_bg_text_align = QComboBox()
        self._combo_bg_text_align.addItems(["Left", "Center", "Right"])
        self._combo_bg_text_align.setCurrentIndex(1)
        self._combo_bg_text_align.currentIndexChanged.connect(self._on_text_changed)
        text_form.addRow("Alignment", self._combo_bg_text_align)
        self._row_bg_text_letter_spacing = self._make_slider_row(
            text_form, "Letter spacing", -5, 40, 0, self._on_text_changed)
        self._row_bg_text_line_spacing = self._make_slider_row(
            text_form, "Line spacing", 0, 80, 8, self._on_text_changed)
        self._row_bg_text_stroke = self._make_slider_row(
            text_form, "Text outline", 0, 20, 0, self._on_text_changed)
        self._row_bg_text_rotation = self._make_slider_row(
            text_form, "Text rotation (°)", -180, 180, 0, self._on_text_changed)
        self._chk_bg_text_shadow = QCheckBox("Text shadow")
        self._chk_bg_text_shadow.stateChanged.connect(self._on_text_changed)
        text_form.addRow("", self._chk_bg_text_shadow)
        self._row_bg_text_shadow_x = self._make_slider_row(
            text_form, "Shadow X", -50, 50, 4, self._on_text_changed)
        self._row_bg_text_shadow_y = self._make_slider_row(
            text_form, "Shadow Y", -50, 50, 4, self._on_text_changed)
        self._btn_bg_text_color = QPushButton("Text color…")
        self._btn_bg_text_color.clicked.connect(self._pick_bg_text_color)
        text_form.addRow("Fill color", self._btn_bg_text_color)
        self._btn_bg_text_stroke_color = QPushButton("Outline color…")
        self._btn_bg_text_stroke_color.clicked.connect(self._pick_bg_text_stroke_color)
        text_form.addRow("Outline color", self._btn_bg_text_stroke_color)
        self._btn_bg_text_shadow_color = QPushButton("Shadow color…")
        self._btn_bg_text_shadow_color.clicked.connect(self._pick_bg_text_shadow_color)
        text_form.addRow("Shadow color", self._btn_bg_text_shadow_color)
        pl.addWidget(self._bg_group)
        pl.addWidget(self._text_group)
        self._text_group.setVisible(False)

    # ── Center image (Halo) ──
        self._center_group = CollapsibleGroupBox("Center Image (Halo)")
        cg_l = QVBoxLayout(self._center_group)
        self._center_label = QLabel("No image selected")
        self._center_label.setWordWrap(True)
        cg_row = QHBoxLayout()
        btn_ctr = QPushButton("Choose…")
        btn_ctr.clicked.connect(self._open_center_image)
        self._btn_ctr_clear = QPushButton("Clear")
        self._btn_ctr_clear.setEnabled(False)
        self._btn_ctr_clear.clicked.connect(self._clear_center_image)
        cg_row.addWidget(btn_ctr)
        cg_row.addWidget(self._btn_ctr_clear)
        cg_l.addWidget(self._center_label)
        cg_l.addLayout(cg_row)
        pl.addWidget(self._center_group)

        # ── Image overlay ──
        self._image_group = CollapsibleGroupBox("Image Overlay")
        image_form = QFormLayout(self._image_group)
        self._image_label = QLabel("")
        self._image_label.setWordWrap(True)
        image_form.addRow("Asset", self._image_label)
        self._row_image_x = self._make_slider_row(
            image_form, "X (%)", -100, 100, 0, self._on_image_changed)
        self._row_image_y = self._make_slider_row(
            image_form, "Y (%)", -100, 100, 0, self._on_image_changed)
        self._row_image_width = self._make_slider_row(
            image_form, "Width (%)", 1, 200, 35, self._on_image_changed)
        self._row_image_height = self._make_slider_row(
            image_form, "Height (%)", 1, 200, 35, self._on_image_changed)
        self._row_image_opacity = self._make_slider_row(
            image_form, "Opacity (%)", 0, 100, 100, self._on_image_changed)
        self._chk_image_beat = QCheckBox("Beat-reactive scale")
        self._chk_image_beat.stateChanged.connect(self._on_image_changed)
        image_form.addRow("", self._chk_image_beat)
        self._row_image_beat = self._make_slider_row(
            image_form, "Beat intensity (%)", 0, 200, 20, self._on_image_changed)
        pl.addWidget(self._image_group)
        self._image_group.setVisible(False)

        # Particle overlay, adapted from D:\Project\visual StudioCanvas.
        self._particle_group = CollapsibleGroupBox("Particles")
        particle_form = QFormLayout(self._particle_group)
        self._row_particle_count = self._make_slider_row(
            particle_form, "Particle count", 10, 200, 50, self._on_particle_changed)
        self._row_particle_speed = self._make_slider_row(
            particle_form, "Speed (×0.01)", 5, 400, 100, self._on_particle_changed)
        self._row_particle_size = self._make_slider_row(
            particle_form, "Size (×0.1)", 1, 60, 30, self._on_particle_changed)
        self._row_particle_opacity = self._make_slider_row(
            particle_form, "Opacity (%)", 0, 100, 90, self._on_particle_changed)
        self._row_particle_glossy = self._make_slider_row(
            particle_form, "Glossy (%)", 0, 100, 0, self._on_particle_changed)
        self._combo_particle_direction = QComboBox()
        self._combo_particle_direction.addItems([
            "random", "up", "down", "left", "right", "up-left", "up-right",
            "down-left", "down-right", "center-out", "edges-in",
        ])
        self._combo_particle_direction.currentIndexChanged.connect(self._on_particle_changed)
        particle_form.addRow("Direction", self._combo_particle_direction)
        self._combo_particle_origin = QComboBox()
        self._combo_particle_origin.addItems([
            "random", "top", "bottom", "left", "right", "center", "full",
        ])
        self._combo_particle_origin.setCurrentText("full")
        self._combo_particle_origin.currentIndexChanged.connect(self._on_particle_changed)
        particle_form.addRow("Origin", self._combo_particle_origin)
        self._chk_particle_random_size = QCheckBox("Random size")
        self._chk_particle_random_size.setChecked(False)
        self._chk_particle_random_size.stateChanged.connect(self._on_particle_changed)
        particle_form.addRow("", self._chk_particle_random_size)
        self._combo_particle_source = QComboBox()
        self._combo_particle_source.addItem("none")
        self._combo_particle_source.currentIndexChanged.connect(self._on_particle_changed)
        particle_form.addRow("Reactive source", self._combo_particle_source)
        self._combo_particle_color_mode = QComboBox()
        self._combo_particle_color_mode.addItems(["solid", "palette", "rainbow"])
        self._combo_particle_color_mode.setCurrentText("palette")
        self._combo_particle_color_mode.currentIndexChanged.connect(self._on_particle_changed)
        particle_form.addRow("Color mode", self._combo_particle_color_mode)
        self._particle_colors = [
            [0.545, 0.361, 0.965], [0.133, 0.827, 0.933], [0.957, 0.447, 0.714],
        ]
        self._particle_color_buttons = []
        for index in range(3):
            button = QPushButton(f"Color {index + 1}")
            button.clicked.connect(
                lambda checked=False, color_index=index: self._pick_particle_color(color_index))
            particle_form.addRow(f"Color {index + 1}", button)
            self._particle_color_buttons.append(button)
        self._refresh_particle_color_buttons()
        pl.addWidget(self._particle_group)
        self._particle_group.setVisible(False)

    # ── Visual ──
        self._visual_group = QGroupBox("Visual")
        vl = QFormLayout(self._visual_group)
        self._combo_viz = QComboBox()
        for name in VIZ_TYPES_VISIBLE:
            self._combo_viz.addItem(name)
        self._combo_viz.currentIndexChanged.connect(self._on_viz_changed)
        vl.addRow("Type", self._combo_viz)
        self._row_scale = self._make_slider_row(
            vl, "Size (×0.01)", 10, 300, 100, self._on_params_changed)
        self._row_opacity = self._make_slider_row(
            vl, "Opacity (×0.01)", 0, 100, 100, self._on_params_changed)
        self._combo_blend = QComboBox()
        self._combo_blend.addItems(["Additive", "Normal"])
        self._combo_blend.currentIndexChanged.connect(self._on_params_changed)
        vl.addRow("Blend", self._combo_blend)
        self._combo_palette = QComboBox()
        for name in PALETTES:
            self._combo_palette.addItem(name)
        self._combo_palette.addItem(_PERSO_AMP)
        self._combo_palette.addItem(_PERSO_FREQ)
        self._combo_palette.currentIndexChanged.connect(self._on_palette_changed)
        vl.addRow("Palette", self._combo_palette)
        pl.addWidget(self._visual_group)

    # ── Custom amplitude palette ──
        self._custom_amp_group = CollapsibleGroupBox("Amplitude Palette")
        amp_l = QVBoxLayout(self._custom_amp_group)
        amp_l.setSpacing(3)
        self._amp_buttons: list[QPushButton] = []
        for i, lbl in enumerate(_AMP_LABELS):
            btn = QPushButton(lbl)
            btn.clicked.connect(lambda checked, idx=i: self._pick_amp_color(idx))
            amp_l.addWidget(btn)
            self._amp_buttons.append(btn)
        pl.addWidget(self._custom_amp_group)
        self._custom_amp_group.setVisible(False)
        self._refresh_pal_buttons(self._amp_buttons, self._custom_palette_amp)

    # ── Custom frequency palette ──
        self._custom_freq_group = CollapsibleGroupBox("Frequency Palette")
        freq_l = QVBoxLayout(self._custom_freq_group)
        freq_l.setSpacing(3)
        self._freq_buttons: list[QPushButton] = []
        for i, lbl in enumerate(_FREQ_LABELS):
            btn = QPushButton(lbl)
            btn.clicked.connect(lambda checked, idx=i: self._pick_freq_color(idx))
            freq_l.addWidget(btn)
            self._freq_buttons.append(btn)
        pl.addWidget(self._custom_freq_group)
        self._custom_freq_group.setVisible(False)
        self._refresh_pal_buttons(self._freq_buttons, self._custom_palette_freq)

    # ── Visual parameters ──
        self._params_group = CollapsibleGroupBox("Advanced Visual Parameters")
        pf = QFormLayout(self._params_group)
        self._row_bars = self._make_slider_row(
            pf, "Bars", 32, 256, NUM_BARS, self._on_params_changed)
        self._row_height = self._make_slider_row(
            pf, "Visual limit (×0.01)", 10, 250, int(MAX_BAR_HEIGHT * 100),
            self._on_params_changed)
        self._row_radius = self._make_slider_row(
            pf, "Inner boundary (×0.01)", 5, 70, int(CIRCLE_RADIUS_RATIO * 100),
            self._on_params_changed)
        self._row_bar_width = self._make_slider_row(
            pf, "Bar thickness (%)", 10, 100, int(BAR_WIDTH * 100),
            self._on_params_changed)
        self._combo_bar_shape = QComboBox()
        self._combo_bar_shape.addItems(BAR_SHAPES)
        self._combo_bar_shape.currentIndexChanged.connect(self._on_params_changed)
        pf.addRow("Bar ends", self._combo_bar_shape)
        self._row_smooth = self._make_slider_row(
            pf, "Smoothing (×0.01)", 50, 99, int(SMOOTHING_DECAY * 100),
            self._on_global_smoothing_changed)
        self._row_pulse = self._make_slider_row(
            pf, "Pulse (×0.01)", 0, 200, int(PULSE_INTENSITY * 100),
            self._on_params_changed)
        self._row_sensitivity = self._make_slider_row(
            pf, "Sensitivity (×0.01)", 25, 300, int(SENSITIVITY * 100),
            self._on_params_changed)
        self._row_rotation = self._make_slider_row(
            pf, "Rotation (°)", 0, 360, 0, self._on_params_changed)
        self._chk_mirror = QCheckBox("Mirror (½ circle = full spectrum)")
        self._chk_mirror.stateChanged.connect(self._on_params_changed)
        pf.addRow("", self._chk_mirror)
        pl.addWidget(self._params_group)

    # ── Frequency distribution (global) ──
        self._freq_group = CollapsibleGroupBox("Frequency Distribution")
        ff = QFormLayout(self._freq_group)
        self._cqt_check = QCheckBox("Mode CQT")
        self._cqt_check.stateChanged.connect(self._on_freq_mode_changed)
        ff.addRow("", self._cqt_check)
        self._row_bass_split = self._make_slider_row(
            ff, "Bass (% of bars)", 20, 80, int(FREQ_BASS_SPLIT * 100),
            self._on_freq_changed)
        self._row_bass_hz = self._make_slider_row(
            ff, "Bass crossover (Hz)", 80, 1500, int(FREQ_BASS_SPLIT_HZ),
            self._on_freq_changed)
        self._row_cqt_bpo = self._make_slider_row(
            ff, "Bins/octave", 6, 48, CQT_BINS_PER_OCTAVE, self._on_freq_changed)
        pl.addWidget(self._freq_group)
        self._freq_group.setVisible(False)
        self._update_freq_ui(cqt=False)

        # ── Halo Sine ──
        self._halo_sine_group = CollapsibleGroupBox("Halo Sine")
        hsf = QFormLayout(self._halo_sine_group)
        self._row_hs_r_base = self._make_slider_row(
            hsf, "Base radius (×0.01)", 5, 70, int(HALO_SINE_R_BASE * 100),
            self._on_params_changed)
        self._row_hs_amplitude = self._make_slider_row(
            hsf, "Amplitude (×0.01)", 1, 40, int(HALO_SINE_AMPLITUDE * 100),
            self._on_params_changed)
        self._row_hs_points = self._make_slider_row(
            hsf, "Spline points", 16, 256, HALO_SINE_N_POINTS, self._on_params_changed)
        self._row_hs_glow = self._make_slider_row(
            hsf, "Glow layers", 1, 8, HALO_SINE_GLOW_LAYERS, self._on_params_changed)
        self._row_hs_decay = self._make_slider_row(
            hsf, "Smoothing (×0.01)", 0, 99, int(HALO_SINE_SMOOTHING_DECAY * 100),
            self._on_params_changed)
        self._row_hs_fill = self._make_slider_row(
            hsf, "Fill (×0.01)", 0, 100, int(HALO_SINE_FILL_OPACITY * 100),
            self._on_params_changed)
        self._row_hs_gap = self._make_slider_row(
            hsf, "Sine spacing (×0.01)", 10, 250, int(HALO_SINE_SPLINE_GAP * 100),
            self._on_params_changed)
        self._row_hs_pixel = self._make_slider_row(
            hsf, "Pixelation", 1, 64, HALO_SINE_PIXEL_SIZE, self._on_params_changed)
        pl.addWidget(self._halo_sine_group)

        # ── Tunnel Arcade ──
        self._tunnel_group = CollapsibleGroupBox("Tunnel Arcade")
        tg_f = QFormLayout(self._tunnel_group)
        self._combo_tunnel_sides = QComboBox()
        for s in ["4", "6", "8", "12"]:
            self._combo_tunnel_sides.addItem(s)
        self._combo_tunnel_sides.setCurrentText(str(TUNNEL_SIDES))
        self._combo_tunnel_sides.currentIndexChanged.connect(self._on_params_changed)
        tg_f.addRow("Sides", self._combo_tunnel_sides)
        self._combo_tunnel_kick_mode = QComboBox()
        self._combo_tunnel_kick_mode.addItems([
            "Delta", "Adaptive threshold", "Spectral kick", "Frequency threshold"])
        self._combo_tunnel_kick_mode.setCurrentIndex(TUNNEL_KICK_MODE)
        self._combo_tunnel_kick_mode.currentIndexChanged.connect(self._on_tunnel_kick_mode_changed)
        tg_f.addRow("Detection", self._combo_tunnel_kick_mode)
        self._row_tunnel_kick_freq_lo = self._make_slider_row(
            tg_f, "Band low (%)", 0, 49, TUNNEL_KICK_FREQ_LO, self._on_params_changed)
        self._row_tunnel_kick_freq_hi = self._make_slider_row(
            tg_f, "Band high (%)", 1, 50, TUNNEL_KICK_FREQ_HI, self._on_params_changed)
        self._row_tunnel_kick_threshold = self._make_slider_row(
            tg_f, "Threshold (×0.01)", 10, 800, TUNNEL_KICK_THRESHOLD, self._on_params_changed)
        self._row_tunnel_kick_cooldown = self._make_slider_row(
            tg_f, "Cooldown (frames)", 5, 120, TUNNEL_KICK_COOLDOWN, self._on_params_changed)
        for _r in (self._row_tunnel_kick_freq_lo, self._row_tunnel_kick_freq_hi,
                   self._row_tunnel_kick_threshold, self._row_tunnel_kick_cooldown):
            self._set_row_visible(_r, False)
        self._row_tunnel_rings = self._make_slider_row(
            tg_f, "Rings", 1, 16, TUNNEL_RINGS, self._on_params_changed)
        self._row_tunnel_speed = self._make_slider_row(
            tg_f, "Speed (×0.01)", 10, 400, int(TUNNEL_SPEED * 100), self._on_params_changed)
        self._row_tunnel_kick_zoom = self._make_slider_row(
            tg_f, "Kick zoom (×0.01)", 0, 300, int(TUNNEL_KICK_ZOOM * 100), self._on_params_changed)
        self._row_tunnel_chroma = self._make_slider_row(
            tg_f, "Aberration (×0.01)", 0, 300, int(TUNNEL_CHROMA * 100), self._on_params_changed)
        self._row_tunnel_kick_sens = self._make_slider_row(
            tg_f, "Kick sens. (×0.01)", 10, 500, int(TUNNEL_KICK_SENSITIVITY * 100), self._on_params_changed)
        self._row_tunnel_bass_speed = self._make_slider_row(
            tg_f, "Audio speed (×0.01)", 0, 1000, int(TUNNEL_BASS_SPEED * 100), self._on_params_changed)
        pl.addWidget(self._tunnel_group)

        # ── Nuclear Shockwave (mode 10) ──
        self._nuke_group = CollapsibleGroupBox("Nuclear Shockwave")
        nk_f = QFormLayout(self._nuke_group)
        self._row_nuke_thresh = self._make_slider_row(
            nk_f, "Kick threshold (×0.001)", 5, 150, NUKE_KICK_THRESHOLD, self._on_params_changed)
        self._row_nuke_speed = self._make_slider_row(
            nk_f, "Wave speed (×0.01)", 20, 300, int(NUKE_SPEED * 100), self._on_params_changed)
        self._row_nuke_life = self._make_slider_row(
            nk_f, "Lifetime (×0.1 s)", 5, 60, int(NUKE_LIFE * 10), self._on_params_changed)
        self._row_nuke_width = self._make_slider_row(
            nk_f, "Width (×0.01)", 30, 400, int(NUKE_WIDTH * 100), self._on_params_changed)
        self._row_nuke_flash = self._make_slider_row(
            nk_f, "Flash (×0.01)", 0, 100, int(NUKE_FLASH * 100), self._on_params_changed)
        self._row_nuke_bg = self._make_slider_row(
            nk_f, "Background (×0.01)", 0, 300, int(NUKE_BG * 100), self._on_params_changed)
        pl.addWidget(self._nuke_group)

        # ── Void Pull (mode 11) ──
        self._void_group = CollapsibleGroupBox("Void Pull")
        vd_f = QFormLayout(self._void_group)
        self._row_void_speed = self._make_slider_row(
            vd_f, "Rotation speed (×0.01)", 0, 100, int(VOID_SPEED * 100), self._on_params_changed)
        self._row_void_pull = self._make_slider_row(
            vd_f, "Kick/bass distortion (×0.01)", 0, 150, int(VOID_PULL * 100), self._on_params_changed)
        self._row_void_rays = self._make_slider_row(
            vd_f, "Treble rays (×0.01)", 0, 600, int(VOID_RAYS * 100), self._on_params_changed)
        self._row_void_arms = self._make_slider_row(
            vd_f, "Spiral arms", 1, 8, VOID_ARMS, self._on_params_changed)
        pl.addWidget(self._void_group)

    # ── Beat effects (per layer) ──
        self._beats_group = CollapsibleGroupBox("Beat Effects")
        beats_f = QFormLayout(self._beats_group)
        self._chk_flash = QCheckBox("Flash")
        self._chk_flash.stateChanged.connect(self._on_params_changed)
        beats_f.addRow("", self._chk_flash)
        self._row_flash_intensity = self._make_slider_row(
            beats_f, "Intensity (×0.01)", 1, 100, int(FLASH_INTENSITY * 100),
            self._on_params_changed)
        self._chk_flash.stateChanged.connect(
            lambda s: self._set_row_visible(self._row_flash_intensity, bool(s)))
        self._set_row_visible(self._row_flash_intensity, False)
        pl.addWidget(self._beats_group)

        # ── Export ──
        eg = QGroupBox("Export")
        ef = QFormLayout(eg)
        self._combo_res = QComboBox()
        for r in RESOLUTIONS:
            self._combo_res.addItem(r)
        self._combo_res.setCurrentText("1080p")
        ef.addRow("Resolution", self._combo_res)
        self._combo_fps = QComboBox()
        self._combo_fps.addItems(["30", "60"])
        self._combo_fps.setCurrentText("60")
        ef.addRow("FPS", self._combo_fps)
        pl.addWidget(eg)

        self._btn_play = QPushButton("▶ Preview")
        self._btn_play.setEnabled(False)
        self._btn_play.clicked.connect(self._toggle_playback)
        pl.addWidget(self._btn_play)

        # ── Transport ──
        self._seek_slider = QSlider(Qt.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(False)
        self._seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self._seek_slider.sliderReleased.connect(self._on_seek_released)
        pl.addWidget(self._seek_slider)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume"))
        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(100)
        self._vol_slider.valueChanged.connect(self._on_volume_changed)
        # Restarting playback is what actually applies a new volume, and it tears down
        # the audio stream and the FFT. Do it once the slider is let go, never on every
        # value emitted during a drag.
        self._vol_slider.sliderReleased.connect(self._on_volume_released)
        self._vol_label = QLabel("100%")
        self._vol_label.setFixedWidth(36)
        vol_row.addWidget(self._vol_slider)
        vol_row.addWidget(self._vol_label)
        pl.addLayout(vol_row)

        self._btn_export = QPushButton("⬇ Export MP4")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._start_export)
        pl.addWidget(self._btn_export)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        pl.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        pl.addWidget(self._status)

        self._preview = PreviewWidget()
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.layer_moved.connect(self._on_layer_moved)

        preview_column = QWidget()
        preview_layout = QVBoxLayout(preview_column)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(AspectRatioContainer(self._preview), 1)
        preview_layout.addWidget(lg)
        root.addWidget(preview_column, 1)

    # ── Layer list management ────────────────────────────────────
    def _layer_item_label(self, layer: Layer) -> str:
        vis  = "👁" if layer.visible else "🚫"
        solo = " ⓢ" if layer.solo else ""
        return f"{vis} {layer.name}{solo}"

    def _refresh_layer_list(self):
        self._loading = True
        self._layer_list.clear()
        # Display topmost layer first (render order is bottom→top)
        for layer in reversed(self._lm.layers):
            item = QListWidgetItem(self._layer_item_label(layer))
            item.setData(Qt.UserRole, layer.id)
            self._layer_list.addItem(item)
        bg_item = QListWidgetItem("🔒 Background")
        bg_item.setData(Qt.UserRole, 0)
        bg_item.setFlags(bg_item.flags() & ~Qt.ItemIsDragEnabled)
        self._layer_list.addItem(bg_item)
        # Sync selection
        sel = self._lm.selected_id
        for row in range(self._layer_list.count()):
            if self._layer_list.item(row).data(Qt.UserRole) == sel:
                self._layer_list.setCurrentRow(row)
                break
        self._loading = False

    def _refresh_selected_item_label(self):
        layer = self._lm.selected()
        if layer is None:
            return
        row = self._layer_list.currentRow()
        if row >= 0:
            self._layer_list.item(row).setText(self._layer_item_label(layer))

    def _on_layer_selected(self, row: int):
        if self._loading or row < 0:
            return
        item = self._layer_list.item(row)
        if item is None:
            return
        selected_id = item.data(Qt.UserRole)
        self._lm.selected_id = selected_id
        if selected_id == 0:
            self._show_bg_editor()
            return
        self._load_layer_into_ui(self._lm.selected())

    def _on_layers_reordered(self, *args):
        if self._loading:
            return
        # Rebuild render order from list (top item = topmost = last in layers)
        ids_top_to_bottom = [self._layer_list.item(r).data(Qt.UserRole)
                             for r in range(self._layer_list.count())]
        id_to_layer = {l.id: l for l in self._lm.layers}
        self._lm.layers = [id_to_layer[i] for i in reversed(ids_top_to_bottom)
                           if i in id_to_layer]
        self._refresh_layer_list()

    def _hide_layer_editors(self):
        for group in (
            self._bg_group, self._text_group, self._visual_group,
            self._params_group, self._freq_group, self._beats_group,
            self._center_group, self._custom_amp_group, self._custom_freq_group,
            self._halo_sine_group, self._tunnel_group, self._nuke_group,
            self._void_group,
            self._image_group, self._particle_group,
        ):
            group.setVisible(False)

    def _show_bg_editor(self):
        self._hide_layer_editors()
        self._loading = True
        bg = self._lm.bg
        self._sl(self._row_bg_opacity).setValue(round(bg.opacity * 100))
        self._chk_bg_zoom.setChecked(bg.zoom_reactive)
        self._sl(self._row_bg_zoom_int).setValue(round(bg.zoom_intensity * 100))
        self._combo_bg_fit.setCurrentIndex(bg.fit_mode)
        self._sl(self._row_bg_crop_zoom).setValue(round(bg.crop_zoom * 100))
        self._sl(self._row_bg_crop_x).setValue(round(bg.crop_x * 100))
        self._sl(self._row_bg_crop_y).setValue(round(bg.crop_y * 100))
        crop = bg.fit_mode == 2
        for row in (self._row_bg_crop_zoom, self._row_bg_crop_x, self._row_bg_crop_y):
            self._set_row_visible(row, crop)
        self._loading = False
        self._bg_group.setVisible(True)
        self._bg_group.setChecked(True)

    def _add_layer(self):
        if self._lm.add_layer(mode=DEFAULT_VIZ_TYPE) is None:
            self._status.setText(f"Maximum of {MAX_VISUAL_LAYERS} layers reached.")
            return
        self._refresh_layer_list()
        self._load_layer_into_ui(self._lm.selected())

    def _add_text_layer(self):
        if self._lm.add_text_layer() is None:
            self._status.setText(f"Maximum of {MAX_TEXT_LAYERS} text layers reached.")
            return
        self._refresh_layer_list()
        self._load_layer_into_ui(self._lm.selected())

    def _add_image_layer(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Add Image Overlay", "",
            "Images (*.png *.svg *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if not path:
            return
        layer = self._lm.add_image_layer(path)
        layer.name = Path(path).stem[:24] or layer.name
        self._refresh_layer_list()
        self._load_layer_into_ui(layer)

    def _add_particle_layer(self):
        layer = self._lm.add_particle_layer()
        self._refresh_layer_list()
        self._load_layer_into_ui(layer)

    def _remove_selected_layer(self):
        layer = self._lm.selected()
        if layer is None:
            return
        self._preview.release_layer_state(layer.id)
        self._lm.remove_layer(layer.id)
        self._refresh_layer_list()
        if self._lm.selected():
            self._load_layer_into_ui(self._lm.selected())
        else:
            self._lm.selected_id = 0
            self._refresh_layer_list()
            self._show_bg_editor()

    def _toggle_selected_visible(self):
        layer = self._lm.selected()
        if layer:
            layer.visible = not layer.visible
            self._refresh_selected_item_label()

    def _toggle_selected_solo(self):
        layer = self._lm.selected()
        if layer:
            layer.solo = not layer.solo
            self._refresh_selected_item_label()

    def _on_layer_moved(self):
        layer = self._lm.selected()
        if layer and layer.mode == TEXT_MODE:
            self._loading = True
            self._sl(self._row_bg_text_x).setValue(round((layer.x + 1.0) * 50))
            self._sl(self._row_bg_text_y).setValue(round((1.0 - layer.y) * 50))
            self._loading = False
        elif layer and layer.mode == IMAGE_MODE:
            self._loading = True
            self._sl(self._row_image_x).setValue(round(layer.x * 100))
            self._sl(self._row_image_y).setValue(round(layer.y * 100))
            self._sl(self._row_image_width).setValue(round(layer.image_width * 100))
            self._sl(self._row_image_height).setValue(round(layer.image_height * 100))
            self._loading = False

    # ── Load / write selected layer ──────────────────────────────
    def _load_layer_into_ui(self, layer: Layer | None):
        if layer is None:
            return
        self._loading = True
        self._hide_layer_editors()

        # transform
        self._sl(self._row_scale).setValue(int(layer.scale * 100))
        self._sl(self._row_opacity).setValue(int(layer.opacity * 100))
        self._combo_blend.setCurrentIndex(0 if layer.blend_mode == BLEND_ADDITIVE else 1)

        if layer.mode == TEXT_MODE:
            self._load_text_layer_into_ui(layer)
            self._loading = False
            self._text_group.setVisible(True)
            self._text_group.setChecked(True)
            return
        if layer.mode == IMAGE_MODE:
            self._image_label.setText(Path(layer.image_path).name if layer.image_path else "No asset")
            self._sl(self._row_image_x).setValue(round(layer.x * 100))
            self._sl(self._row_image_y).setValue(round(layer.y * 100))
            self._sl(self._row_image_width).setValue(round(layer.image_width * 100))
            self._sl(self._row_image_height).setValue(round(layer.image_height * 100))
            self._sl(self._row_image_opacity).setValue(round(layer.opacity * 100))
            self._chk_image_beat.setChecked(layer.image_beat_reactive)
            self._sl(self._row_image_beat).setValue(round(layer.image_beat_intensity * 100))
            self._set_row_visible(self._row_image_beat, layer.image_beat_reactive)
            self._loading = False
            self._image_group.setVisible(True)
            self._image_group.setChecked(True)
            return
        if layer.mode == PARTICLE_MODE:
            self._sl(self._row_particle_count).setValue(layer.particle_count)
            self._sl(self._row_particle_speed).setValue(round(layer.particle_speed * 100))
            self._sl(self._row_particle_size).setValue(round(layer.particle_size * 10))
            self._sl(self._row_particle_opacity).setValue(round(layer.opacity * 100))
            self._sl(self._row_particle_glossy).setValue(round(layer.particle_glossy * 100))
            self._combo_particle_direction.setCurrentText(layer.particle_direction)
            self._combo_particle_origin.setCurrentText(layer.particle_origin)
            self._chk_particle_random_size.setChecked(layer.particle_random_size)
            self._combo_particle_source.setCurrentText("none")
            self._combo_particle_color_mode.setCurrentText(layer.particle_color_mode)
            self._particle_colors = [list(color) for color in layer.particle_colors]
            self._refresh_particle_color_buttons()
            self._loading = False
            self._on_particle_changed()
            self._particle_group.setVisible(True)
            self._particle_group.setChecked(True)
            return

        self._visual_group.setVisible(True)
        self._params_group.setVisible(True)
        self._beats_group.setVisible(True)

        # viz + palette
        preset_name = layer.preset_name if layer.preset_name in VISUAL_PRESETS else ""
        if not preset_name:
            preset_name = next(
                (name for name, preset in VISUAL_PRESETS.items()
                 if preset["mode"] == layer.mode), VIZ_TYPES_VISIBLE[0])
        self._combo_viz.setCurrentText(preset_name)
        self._custom_palette_amp  = [list(c) for c in layer.palette] if layer.palette_name == _PERSO_AMP else self._custom_palette_amp
        self._custom_palette_freq = [list(c) for c in layer.palette] if layer.palette_name == _PERSO_FREQ else self._custom_palette_freq
        self._refresh_pal_buttons(self._amp_buttons, self._custom_palette_amp)
        self._refresh_pal_buttons(self._freq_buttons, self._custom_palette_freq)
        idx = self._combo_palette.findText(layer.palette_name)
        self._combo_palette.setCurrentIndex(idx if idx >= 0 else 0)

        # generic
        self._sl(self._row_bars).setValue(layer.num_bars)
        self._sl(self._row_height).setValue(int(layer.max_bar_height * 100))
        self._sl(self._row_radius).setValue(int(layer.circle_radius * 100))
        self._sl(self._row_bar_width).setValue(int(layer.bar_width * 100))
        self._combo_bar_shape.setCurrentIndex(layer.bar_shape)
        self._sl(self._row_pulse).setValue(int(layer.pulse_intensity * 100))
        self._sl(self._row_sensitivity).setValue(int(layer.sensitivity * 100))
        self._sl(self._row_rotation).setValue(int(math.degrees(layer.rotation)))
        self._chk_mirror.setChecked(layer.mirror)

        # halo
        self._sl(self._row_hs_r_base).setValue(int(layer.halo_r_base * 100))
        self._sl(self._row_hs_amplitude).setValue(int(layer.halo_amplitude * 100))
        self._sl(self._row_hs_points).setValue(layer.halo_n_points)
        self._sl(self._row_hs_glow).setValue(layer.halo_glow_layers)
        self._sl(self._row_hs_decay).setValue(int(layer.halo_smoothing_decay * 100))
        self._sl(self._row_hs_fill).setValue(int(layer.halo_fill_opacity * 100))
        self._sl(self._row_hs_gap).setValue(int(layer.halo_spline_gap * 100))
        self._sl(self._row_hs_pixel).setValue(layer.halo_pixel_size)

        # tunnel
        self._combo_tunnel_sides.setCurrentText(str(layer.tunnel_sides))
        self._combo_tunnel_kick_mode.setCurrentIndex(layer.tunnel_kick_mode)
        self._sl(self._row_tunnel_kick_freq_lo).setValue(layer.tunnel_kick_freq_lo)
        self._sl(self._row_tunnel_kick_freq_hi).setValue(layer.tunnel_kick_freq_hi)
        self._sl(self._row_tunnel_kick_threshold).setValue(int(layer.tunnel_kick_threshold * 100))
        self._sl(self._row_tunnel_kick_cooldown).setValue(layer.tunnel_kick_cooldown)
        self._sl(self._row_tunnel_rings).setValue(layer.tunnel_rings)
        self._sl(self._row_tunnel_speed).setValue(int(layer.tunnel_speed * 100))
        self._sl(self._row_tunnel_kick_zoom).setValue(int(layer.tunnel_kick_zoom * 100))
        self._sl(self._row_tunnel_chroma).setValue(int(layer.tunnel_chroma * 100))
        self._sl(self._row_tunnel_kick_sens).setValue(int(layer.tunnel_kick_sensitivity * 100))
        self._sl(self._row_tunnel_bass_speed).setValue(int(layer.tunnel_bass_speed * 100))

        # nuclear shockwave
        self._sl(self._row_nuke_thresh).setValue(int(round(layer.nuke_kick_threshold * 1000)))
        self._sl(self._row_nuke_speed).setValue(int(layer.nuke_speed * 100))
        self._sl(self._row_nuke_life).setValue(int(layer.nuke_life * 10))
        self._sl(self._row_nuke_width).setValue(int(layer.nuke_width * 100))
        self._sl(self._row_nuke_flash).setValue(int(layer.nuke_flash * 100))
        self._sl(self._row_nuke_bg).setValue(int(layer.nuke_bg * 100))

        # void pull
        self._sl(self._row_void_speed).setValue(int(layer.void_speed * 100))
        self._sl(self._row_void_pull).setValue(int(layer.void_pull * 100))
        self._sl(self._row_void_rays).setValue(int(layer.void_rays * 100))
        self._sl(self._row_void_arms).setValue(layer.void_arms)

        # flash
        self._chk_flash.setChecked(layer.flash)
        self._sl(self._row_flash_intensity).setValue(int(layer.flash_intensity * 100))

        self._loading = False
        self._update_ui_for_viz(layer.mode)
        self._update_tunnel_kick_visibility()
        self._set_row_visible(self._row_flash_intensity, layer.flash)

    def _write_ui_to_layer(self, layer: Layer):
        layer.scale = self._sl(self._row_scale).value() / 100
        layer.opacity = self._sl(self._row_opacity).value() / 100
        layer.blend_mode = BLEND_ADDITIVE if self._combo_blend.currentIndex() == 0 else BLEND_NORMAL
        if layer.mode == TEXT_MODE:
            self._write_text_ui_to_layer(layer)
            return
        preset_name = self._combo_viz.currentText()
        layer.mode = VISUAL_PRESETS[preset_name]["mode"]
        layer.preset_name = preset_name
        layer.name = preset_name
        layer.palette = self._current_palette()
        layer.palette_name = self._combo_palette.currentText()
        layer.pal_mode = self._current_pal_mode()
        layer.num_bars = self._sl(self._row_bars).value()
        layer.max_bar_height = self._sl(self._row_height).value() / 100
        layer.circle_radius = self._sl(self._row_radius).value() / 100
        layer.bar_width = self._sl(self._row_bar_width).value() / 100
        layer.bar_shape = self._combo_bar_shape.currentIndex()
        layer.pulse_intensity = self._sl(self._row_pulse).value() / 100
        layer.sensitivity = self._sl(self._row_sensitivity).value() / 100
        layer.rotation = math.radians(self._sl(self._row_rotation).value())
        layer.mirror = self._chk_mirror.isChecked()
        layer.halo_r_base = self._sl(self._row_hs_r_base).value() / 100
        layer.halo_amplitude = self._sl(self._row_hs_amplitude).value() / 100
        layer.halo_n_points = self._sl(self._row_hs_points).value()
        layer.halo_glow_layers = self._sl(self._row_hs_glow).value()
        layer.halo_smoothing_decay = self._sl(self._row_hs_decay).value() / 100
        layer.halo_fill_opacity = self._sl(self._row_hs_fill).value() / 100
        layer.halo_spline_gap = self._sl(self._row_hs_gap).value() / 100
        layer.halo_pixel_size = self._sl(self._row_hs_pixel).value()
        layer.tunnel_sides = int(self._combo_tunnel_sides.currentText())
        layer.tunnel_kick_mode = self._combo_tunnel_kick_mode.currentIndex()
        layer.tunnel_kick_freq_lo = self._sl(self._row_tunnel_kick_freq_lo).value()
        layer.tunnel_kick_freq_hi = self._sl(self._row_tunnel_kick_freq_hi).value()
        layer.tunnel_kick_threshold = self._sl(self._row_tunnel_kick_threshold).value() / 100.0
        layer.tunnel_kick_cooldown = self._sl(self._row_tunnel_kick_cooldown).value()
        layer.tunnel_rings = self._sl(self._row_tunnel_rings).value()
        layer.tunnel_speed = self._sl(self._row_tunnel_speed).value() / 100
        layer.tunnel_kick_zoom = self._sl(self._row_tunnel_kick_zoom).value() / 100
        layer.tunnel_chroma = self._sl(self._row_tunnel_chroma).value() / 100
        layer.tunnel_kick_sensitivity = self._sl(self._row_tunnel_kick_sens).value() / 100
        layer.tunnel_bass_speed = self._sl(self._row_tunnel_bass_speed).value() / 100
        layer.nuke_kick_threshold = self._sl(self._row_nuke_thresh).value() / 1000.0
        layer.nuke_speed = self._sl(self._row_nuke_speed).value() / 100
        layer.nuke_life = self._sl(self._row_nuke_life).value() / 10
        layer.nuke_width = self._sl(self._row_nuke_width).value() / 100
        layer.nuke_flash = self._sl(self._row_nuke_flash).value() / 100
        layer.nuke_bg = self._sl(self._row_nuke_bg).value() / 100
        layer.void_speed = self._sl(self._row_void_speed).value() / 100
        layer.void_pull = self._sl(self._row_void_pull).value() / 100
        layer.void_rays = self._sl(self._row_void_rays).value() / 100
        layer.void_arms = self._sl(self._row_void_arms).value()
        layer.flash = self._chk_flash.isChecked()
        layer.flash_intensity = self._sl(self._row_flash_intensity).value() / 100

    def _load_text_layer_into_ui(self, layer: Layer):
        self._bg_text_color = tuple(layer.text_color)
        self._bg_text_stroke_color = tuple(layer.text_stroke_color)
        self._bg_text_shadow_color = tuple(layer.text_shadow_color)
        self._bg_text_font = layer.text_font
        self._bg_text.setPlainText(layer.text)
        self._sl(self._row_bg_text_size).setValue(layer.text_size)
        self._sl(self._row_bg_text_x).setValue(round((layer.x + 1.0) * 50))
        self._sl(self._row_bg_text_y).setValue(round((1.0 - layer.y) * 50))
        self._sl(self._row_bg_text_opacity).setValue(round(layer.text_opacity * 100))
        font_index = self._combo_bg_text_font.findText(layer.text_font_label)
        self._combo_bg_text_font.setCurrentIndex(font_index if font_index >= 0 else self._combo_bg_text_font.count() - 1)
        self._chk_bg_text_bold.setChecked(layer.text_bold)
        self._chk_bg_text_italic.setChecked(layer.text_italic)
        self._combo_bg_text_align.setCurrentText(layer.text_align.title())
        self._sl(self._row_bg_text_letter_spacing).setValue(layer.text_letter_spacing)
        self._sl(self._row_bg_text_line_spacing).setValue(layer.text_line_spacing)
        self._sl(self._row_bg_text_stroke).setValue(layer.text_stroke)
        self._sl(self._row_bg_text_rotation).setValue(round(layer.text_rotation))
        self._chk_bg_text_shadow.setChecked(layer.text_shadow)
        self._sl(self._row_bg_text_shadow_x).setValue(layer.text_shadow_x)
        self._sl(self._row_bg_text_shadow_y).setValue(layer.text_shadow_y)

    def _write_text_ui_to_layer(self, layer: Layer):
        layer.text = self._bg_text.toPlainText()
        layer.text_size = self._sl(self._row_bg_text_size).value()
        layer.x = self._sl(self._row_bg_text_x).value() / 50.0 - 1.0
        layer.y = 1.0 - self._sl(self._row_bg_text_y).value() / 50.0
        layer.text_opacity = self._sl(self._row_bg_text_opacity).value() / 100
        layer.text_font = self._bg_text_font
        layer.text_font_label = self._combo_bg_text_font.currentText()
        layer.text_bold = self._chk_bg_text_bold.isChecked()
        layer.text_italic = self._chk_bg_text_italic.isChecked()
        layer.text_align = self._combo_bg_text_align.currentText().lower()
        layer.text_letter_spacing = self._sl(self._row_bg_text_letter_spacing).value()
        layer.text_line_spacing = self._sl(self._row_bg_text_line_spacing).value()
        layer.text_stroke = self._sl(self._row_bg_text_stroke).value()
        layer.text_stroke_color = self._bg_text_stroke_color
        layer.text_rotation = self._sl(self._row_bg_text_rotation).value()
        layer.text_shadow = self._chk_bg_text_shadow.isChecked()
        layer.text_shadow_color = self._bg_text_shadow_color
        layer.text_shadow_x = self._sl(self._row_bg_text_shadow_x).value()
        layer.text_shadow_y = self._sl(self._row_bg_text_shadow_y).value()
        layer.text_color = self._bg_text_color
        first_line = next((line.strip() for line in layer.text.splitlines() if line.strip()), "Text")
        layer.name = first_line[:24]

    # ── Visibility helpers ───────────────────────────────────────
    def _update_freq_ui(self, cqt: bool):
        self._set_row_visible(self._row_bass_split, not cqt)
        self._set_row_visible(self._row_bass_hz,    not cqt)
        self._set_row_visible(self._row_cqt_bpo,    cqt)

    def _update_tunnel_kick_visibility(self):
        advanced = self._combo_tunnel_kick_mode.currentIndex() in (2, 3)
        self._set_row_visible(self._row_tunnel_kick_freq_lo,   advanced)
        self._set_row_visible(self._row_tunnel_kick_freq_hi,   advanced)
        self._set_row_visible(self._row_tunnel_kick_threshold, advanced)
        self._set_row_visible(self._row_tunnel_kick_cooldown,  advanced)

    def _update_ui_for_viz(self, viz_type: int):
        is_radial    = viz_type in _RADIAL_MODES
        has_rotation = viz_type in _ROTATION_MODES
        is_bar       = viz_type in _BAR_MODES
        has_bar_shape = viz_type in _BAR_SHAPE_MODES
        has_height   = viz_type in _HEIGHT_MODES
        is_halo      = viz_type in _HALO_MODES
        is_halo_sine = viz_type in _HALO_SINE_MODES
        is_tunnel    = viz_type in _TUNNEL_MODES
        has_mirror   = viz_type in _MIRROR_MODES

        self._set_row_visible(self._row_bars,      is_bar)
        self._set_row_visible(self._row_height,    has_height)
        self._set_row_visible(self._row_radius,    is_radial)
        self._set_row_visible(self._row_bar_width, is_bar)
        self._combo_bar_shape.setVisible(has_bar_shape)
        bar_shape_label = self._params_group.layout().labelForField(self._combo_bar_shape)
        if bar_shape_label is not None:
            bar_shape_label.setVisible(has_bar_shape)
        self._set_row_visible(self._row_pulse,     is_radial)
        self._set_row_visible(self._row_rotation,  has_rotation)
        self._center_group.setVisible(is_halo)
        self._halo_sine_group.setVisible(is_halo_sine)
        is_halo_circle = viz_type == 6
        self._set_row_visible(self._row_hs_r_base, is_halo_circle)
        self._set_row_visible(self._row_hs_gap,    is_halo_circle)
        self._set_row_visible(self._row_hs_pixel,  is_halo_circle)
        self._tunnel_group.setVisible(is_tunnel)
        self._nuke_group.setVisible(viz_type in _NUKE_MODES)
        self._void_group.setVisible(viz_type in _VOID_MODES)
        self._chk_mirror.setVisible(has_mirror)
        pal_name = self._combo_palette.currentText()
        self._custom_amp_group.setVisible(pal_name == _PERSO_AMP)
        self._custom_freq_group.setVisible(pal_name == _PERSO_FREQ)

    # ── Accessors ────────────────────────────────────────────────
    def _sl(self, row): return row[0]

    def _current_palette(self):
        name = self._combo_palette.currentText()
        if name == _PERSO_AMP:
            return [list(c) for c in self._custom_palette_amp]
        if name == _PERSO_FREQ:
            return [list(c) for c in self._custom_palette_freq]
        return [list(c) for c in PALETTES[name]]

    def _current_pal_mode(self) -> int:
        return 1 if self._combo_palette.currentText() == _PERSO_FREQ else 0

    def _refresh_pal_buttons(self, buttons: list, palette: list):
        for i, c in enumerate(palette):
            r, g, b = c[0], c[1], c[2]
            hex_col = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
            lum     = 0.299*r + 0.587*g + 0.114*b
            txt_col = "#000000" if lum > 0.5 else "#ffffff"
            buttons[i].setStyleSheet(
                f"background-color:{hex_col};color:{txt_col};border:none;padding:4px;")

    def _pick_amp_color(self, idx: int):
        c = self._custom_palette_amp[idx]
        initial = QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))
        col = QColorDialog.getColor(initial, self, f"Amplitude — stop {idx + 1}")
        if col.isValid():
            self._custom_palette_amp[idx] = [col.redF(), col.greenF(), col.blueF()]
            self._refresh_pal_buttons(self._amp_buttons, self._custom_palette_amp)
            self._on_params_changed()

    def _pick_freq_color(self, idx: int):
        c = self._custom_palette_freq[idx]
        initial = QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))
        col = QColorDialog.getColor(initial, self, f"Frequency — stop {idx + 1}")
        if col.isValid():
            self._custom_palette_freq[idx] = [col.redF(), col.greenF(), col.blueF()]
            self._refresh_pal_buttons(self._freq_buttons, self._custom_palette_freq)
            self._on_params_changed()

    def _on_palette_changed(self):
        pal_name = self._combo_palette.currentText()
        self._custom_amp_group.setVisible(pal_name == _PERSO_AMP)
        self._custom_freq_group.setVisible(pal_name == _PERSO_FREQ)
        self._on_params_changed()

    def _use_cqt(self) -> bool:
        return self._cqt_check.isChecked()

    def _make_fft(self) -> FFTProcessor:
        return FFTProcessor(
            sr=self._audio.sr,
            num_bars=FFT_ANALYSIS_BARS,
            smoothing_decay=self._sl(self._row_smooth).value() / 100,
            bass_split=self._sl(self._row_bass_split).value() / 100,
            bass_split_hz=float(self._sl(self._row_bass_hz).value()),
            use_cqt=self._use_cqt(),
            bins_per_octave=self._sl(self._row_cqt_bpo).value(),
        )

    # ── Slots ────────────────────────────────────────────────────
    def _template_settings(self) -> dict:
        return {
            "resolution": self._combo_res.currentText(),
            "fps": int(self._combo_fps.currentText()),
            "smoothing_decay": self._sl(self._row_smooth).value() / 100,
            "bass_split": self._sl(self._row_bass_split).value() / 100,
            "bass_split_hz": float(self._sl(self._row_bass_hz).value()),
            "use_cqt": self._use_cqt(),
            "bins_per_octave": self._sl(self._row_cqt_bpo).value(),
        }

    def _save_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", self._template_path or "visualizer-template.json",
            "Templates (*.json *.yaml *.yml *.toml)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".json"
        try:
            save_template(path, self._lm, self._center_image_path,
                          self._template_settings())
            self._template_path = path
            self._status.setText(f"Template saved:\n{path}")
        except Exception as exc:
            self._status.setText(f"Template save error:\n{exc}")

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Template", self._template_path or "",
            "Templates (*.json *.yaml *.yml *.toml)")
        if not path:
            return
        try:
            lm, center, settings = load_template(path)
            for layer in self._lm.layers:
                self._preview.release_layer_state(layer.id)
            self._lm = lm
            self._center_image_path = center
            self._bg_image_path = lm.bg.image_path
            self._template_path = path
            self._preview.set_layer_manager(lm)
            if self._bg_image_path:
                self._preview.load_background(self._bg_image_path)
            else:
                self._preview.clear_background()
            if center:
                self._preview.load_center_image(center)
            else:
                self._preview.clear_center_image()
            self._loading = True
            self._combo_res.setCurrentText(str(settings.get("resolution", "1080p")))
            self._combo_fps.setCurrentText(str(settings.get("fps", 60)))
            self._sl(self._row_smooth).setValue(round(float(settings.get("smoothing_decay", .85)) * 100))
            self._sl(self._row_bass_split).setValue(round(float(settings.get("bass_split", .60)) * 100))
            self._sl(self._row_bass_hz).setValue(round(float(settings.get("bass_split_hz", 300))))
            self._cqt_check.setChecked(bool(settings.get("use_cqt", False)))
            self._sl(self._row_cqt_bpo).setValue(
                int(settings.get("bins_per_octave", CQT_BINS_PER_OCTAVE)))
            self._loading = False
            self._update_freq_ui(self._use_cqt())
            self._bg_label.setText(Path(self._bg_image_path).name if self._bg_image_path else "No image selected")
            self._btn_bg_clear.setEnabled(bool(self._bg_image_path))
            self._center_label.setText(Path(center).name if center else "No image selected")
            self._btn_ctr_clear.setEnabled(bool(center))
            self._refresh_layer_list()
            if lm.selected():
                self._load_layer_into_ui(lm.selected())
            else:
                self._show_bg_editor()
            self._status.setText(f"Template loaded:\n{path}")
        except Exception as exc:
            self._loading = False
            self._status.setText(f"Template load error:\n{exc}")

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "", "Audio (*.wav *.flac *.mp3 *.ogg *.aiff)")
        if not path:
            return
        self._audio = AudioFile(path)
        self._fft   = self._make_fft()
        self._file_label.setText(Path(path).name)
        self._btn_play.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._seek_frame = 0
        self._seek_slider.setValue(0)
        self._seek_slider.setEnabled(True)
        self._status.setText(f"Duration: {self._audio.duration:.1f}s · {self._audio.sr} Hz")

    def _open_bg_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Background Media", "",
            "Media (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.mp4 *.mov *.mkv *.webm *.avi *.m4v)")
        if not path:
            return
        self._bg_image_path = path
        self._lm.bg.image_path = path
        self._bg_label.setText(Path(path).name)
        self._btn_bg_clear.setEnabled(True)
        self._preview.load_background(path)

    def _clear_bg_image(self):
        self._bg_image_path = None
        self._lm.bg.image_path = None
        self._bg_label.setText("No image selected")
        self._btn_bg_clear.setEnabled(False)
        self._preview.clear_background()

    def _on_bg_changed(self):
        if self._loading:
            return
        self._lm.bg.opacity = self._sl(self._row_bg_opacity).value() / 100
        self._lm.bg.zoom_reactive = self._chk_bg_zoom.isChecked()
        self._lm.bg.zoom_intensity = self._sl(self._row_bg_zoom_int).value() / 100
        self._lm.bg.fit_mode = self._combo_bg_fit.currentIndex()
        self._lm.bg.crop_zoom = self._sl(self._row_bg_crop_zoom).value() / 100
        self._lm.bg.crop_x = self._sl(self._row_bg_crop_x).value() / 100
        self._lm.bg.crop_y = self._sl(self._row_bg_crop_y).value() / 100
        crop = self._lm.bg.fit_mode == 2
        for row in (self._row_bg_crop_zoom, self._row_bg_crop_x, self._row_bg_crop_y):
            self._set_row_visible(row, crop)

    def _on_text_changed(self):
        if self._loading:
            return
        layer = self._lm.selected()
        if layer is None or layer.mode != TEXT_MODE:
            return
        self._write_text_ui_to_layer(layer)
        self._refresh_selected_item_label()

    def _pick_bg_text_color(self):
        initial = QColor(*(round(c * 255) for c in self._bg_text_color))
        color = QColorDialog.getColor(initial, self, "Background text color")
        if color.isValid():
            self._bg_text_color = (color.redF(), color.greenF(), color.blueF())
            self._btn_bg_text_color.setStyleSheet(
                f"background-color: {color.name()}; color: {'black' if color.lightnessF() > 0.5 else 'white'}")
            self._on_text_changed()

    def _on_bg_text_font_changed(self, name: str):
        custom = name == "Custom font file"
        self._btn_bg_text_font.setVisible(custom)
        if not custom:
            self._bg_text_font = _TEXT_FONTS[name]
            self._on_text_changed()

    def _pick_bg_text_font(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Font", "", "Fonts (*.ttf *.otf *.ttc)")
        if path:
            self._bg_text_font = path
            self._btn_bg_text_font.setText(Path(path).name)
            self._btn_bg_text_font.setToolTip(path)
            self._on_text_changed()

    def _pick_bg_text_stroke_color(self):
        color = QColorDialog.getColor(
            QColor(*(round(c * 255) for c in self._bg_text_stroke_color)),
            self, "Text outline color")
        if color.isValid():
            self._bg_text_stroke_color = (color.redF(), color.greenF(), color.blueF())
            self._btn_bg_text_stroke_color.setStyleSheet(
                f"background-color: {color.name()}")
            self._on_text_changed()

    def _pick_bg_text_shadow_color(self):
        color = QColorDialog.getColor(
            QColor(*(round(c * 255) for c in self._bg_text_shadow_color)),
            self, "Text shadow color")
        if color.isValid():
            self._bg_text_shadow_color = (color.redF(), color.greenF(), color.blueF())
            self._btn_bg_text_shadow_color.setStyleSheet(
                f"background-color: {color.name()}")
            self._on_text_changed()

    def _open_center_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Center Image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)")
        if not path:
            return
        self._center_image_path = path
        self._center_label.setText(Path(path).name)
        self._btn_ctr_clear.setEnabled(True)
        self._preview.load_center_image(path)

    def _clear_center_image(self):
        self._center_image_path = None
        self._center_label.setText("No image selected")
        self._btn_ctr_clear.setEnabled(False)
        self._preview.clear_center_image()

    def _on_viz_changed(self):
        if self._loading:
            return
        preset = VISUAL_PRESETS[self._combo_viz.currentText()]
        self._loading = True
        if "num_bars" in preset:
            self._sl(self._row_bars).setValue(preset["num_bars"])
        if "max_bar_height" in preset:
            self._sl(self._row_height).setValue(round(preset["max_bar_height"] * 100))
        if "circle_radius" in preset:
            self._sl(self._row_radius).setValue(round(preset["circle_radius"] * 100))
        if "bar_width" in preset:
            self._sl(self._row_bar_width).setValue(round(preset["bar_width"] * 100))
        if "bar_shape" in preset:
            self._combo_bar_shape.setCurrentIndex(preset["bar_shape"])
        self._loading = False
        self._update_ui_for_viz(preset["mode"])
        self._on_params_changed()
        self._refresh_selected_item_label()

    def _on_params_changed(self):
        if self._loading:
            return
        layer = self._lm.selected()
        if layer is not None:
            self._write_ui_to_layer(layer)

    def _on_image_changed(self):
        if self._loading:
            return
        layer = self._lm.selected()
        if layer is None or layer.mode != IMAGE_MODE:
            return
        layer.x = self._sl(self._row_image_x).value() / 100
        layer.y = self._sl(self._row_image_y).value() / 100
        layer.image_width = self._sl(self._row_image_width).value() / 100
        layer.image_height = self._sl(self._row_image_height).value() / 100
        layer.opacity = self._sl(self._row_image_opacity).value() / 100
        layer.image_beat_reactive = self._chk_image_beat.isChecked()
        layer.image_beat_intensity = self._sl(self._row_image_beat).value() / 100
        self._set_row_visible(self._row_image_beat, layer.image_beat_reactive)

    def _on_particle_changed(self):
        if self._loading:
            return
        layer = self._lm.selected()
        if layer is None or layer.mode != PARTICLE_MODE:
            return
        layer.particle_count = self._sl(self._row_particle_count).value()
        layer.particle_speed = self._sl(self._row_particle_speed).value() / 100
        layer.particle_size = self._sl(self._row_particle_size).value() / 10
        layer.opacity = self._sl(self._row_particle_opacity).value() / 100
        layer.particle_glossy = self._sl(self._row_particle_glossy).value() / 100
        layer.particle_direction = self._combo_particle_direction.currentText()
        layer.particle_origin = self._combo_particle_origin.currentText()
        layer.particle_random_size = self._chk_particle_random_size.isChecked()
        layer.particle_beat_reactive = False
        layer.particle_reactive_source = "none"
        layer.particle_color_mode = self._combo_particle_color_mode.currentText()
        layer.particle_colors = [list(color) for color in self._particle_colors]
        solid = layer.particle_color_mode == "solid"
        visible = layer.particle_color_mode != "rainbow"
        for index, button in enumerate(self._particle_color_buttons):
            button.setVisible(visible and (not solid or index == 0))
            label = self._particle_group.layout().labelForField(button)
            if label is not None:
                label.setVisible(visible and (not solid or index == 0))

    def _pick_particle_color(self, index: int):
        red, green, blue = self._particle_colors[index]
        initial = QColor(round(red * 255), round(green * 255), round(blue * 255))
        color = QColorDialog.getColor(initial, self, f"Particle color {index + 1}")
        if not color.isValid():
            return
        self._particle_colors[index] = [color.redF(), color.greenF(), color.blueF()]
        self._refresh_particle_color_buttons()
        self._on_particle_changed()

    def _refresh_particle_color_buttons(self):
        for button, color in zip(self._particle_color_buttons, self._particle_colors):
            red, green, blue = color
            button.setStyleSheet(
                "background-color:#{:02x}{:02x}{:02x};".format(
                    round(red * 255), round(green * 255), round(blue * 255)))

    def _on_global_smoothing_changed(self):
        if self._loading:
            return
        if self._fft:
            self._fft.smoothing_decay = self._sl(self._row_smooth).value() / 100

    def _on_freq_mode_changed(self):
        self._update_freq_ui(cqt=self._use_cqt())
        self._on_freq_changed()

    def _on_freq_changed(self):
        if self._loading:
            return
        if self._audio:
            new_fft = self._make_fft()
            if self._playback_thread and self._playback_thread.isRunning():
                self._playback_thread.update_fft(new_fft)
            self._fft = new_fft

    def _on_tunnel_kick_mode_changed(self):
        self._update_tunnel_kick_visibility()
        self._on_params_changed()

    def _on_volume_changed(self, val: int):
        self._volume = val / 100.0
        self._vol_label.setText(f"{val}%")
        # No restart here: a drag emits dozens of these per second, and each restart
        # stops and recreates the playback thread and the shared audio stream.
        if not self._vol_slider.isSliderDown():
            self._on_volume_released()

    def _on_volume_released(self):
        if self._playback_thread and self._playback_thread.isRunning():
            self._restart_playback_from(self._seek_frame)

    def _on_seek_pressed(self):
        self._seeking = True

    def _on_seek_released(self):
        if not self._audio:
            self._seeking = False
            return
        hop   = max(1, int(self._audio.sr / int(self._combo_fps.currentText())))
        total = int(len(self._audio.mono) / hop)
        self._seek_frame = int(self._seek_slider.value() / 1000 * total)
        self._seeking = False
        if self._playback_thread and self._playback_thread.isRunning():
            self._restart_playback_from(self._seek_frame)

    def _on_position_changed(self, frame: int, total: int):
        if not self._seeking:
            self._seek_slider.setValue(int(frame / max(total, 1) * 1000))
        self._seek_frame = frame

    def _restart_playback_from(self, frame: int):
        if self._playback_thread:
            self._playback_thread.stop()
            # Bounded wait: the GUI thread must never be able to hang here, whatever
            # state the audio stream is in.
            self._playback_thread.wait(_THREAD_STOP_TIMEOUT_MS)
        fps = int(self._combo_fps.currentText())
        self._fft = self._make_fft()
        self._playback_thread = AudioPlaybackThread(
            self._audio, self._fft, fps, start_frame=frame, volume=self._volume)
        self._playback_thread.frame_ready.connect(self._preview.update_audio_data)
        self._playback_thread.position_changed.connect(self._on_position_changed)
        self._playback_thread.finished.connect(self._on_playback_done)
        self._playback_thread.start()

    def _toggle_playback(self):
        if self._playback_thread and self._playback_thread.isRunning():
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self._audio:
            return
        self._fft = self._make_fft()
        fps = int(self._combo_fps.currentText())
        self._playback_thread = AudioPlaybackThread(
            self._audio, self._fft, fps, start_frame=self._seek_frame, volume=self._volume)
        self._playback_thread.frame_ready.connect(self._preview.update_audio_data)
        self._playback_thread.position_changed.connect(self._on_position_changed)
        self._playback_thread.finished.connect(self._on_playback_done)
        self._playback_thread.start()
        self._btn_play.setText("⏹ Stop")

    def _stop_playback(self):
        if self._playback_thread:
            self._playback_thread.stop()
            self._playback_thread.wait(_THREAD_STOP_TIMEOUT_MS)
        self._btn_play.setText("▶ Preview")

    def _on_playback_done(self):
        self._btn_play.setText("▶ Preview")
        self._seek_frame = 0
        self._seek_slider.setValue(0)

    def _start_export(self):
        if not self._audio:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Output Folder")
        if not out_dir:
            return

        self._stop_playback()
        self._btn_export.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Exporting…")

        worker = ExportWorker(
            audio=self._audio,
            layer_manager=copy.deepcopy(self._lm),
            output_dir=out_dir,
            resolution=self._combo_res.currentText(),
            fps=int(self._combo_fps.currentText()),
            smoothing_decay=self._sl(self._row_smooth).value() / 100,
            bass_split=self._sl(self._row_bass_split).value() / 100,
            bass_split_hz=float(self._sl(self._row_bass_hz).value()),
            use_cqt=self._use_cqt(),
            bins_per_octave=self._sl(self._row_cqt_bpo).value(),
            bg_image_path=self._bg_image_path,
            center_image_path=self._center_image_path,
        )
        self._export_thread = QThread()
        worker.moveToThread(self._export_thread)
        self._export_thread.started.connect(worker.run)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_done)
        worker.error.connect(self._on_export_error)
        worker.finished.connect(self._export_thread.quit)
        worker.error.connect(self._export_thread.quit)
        self._export_worker = worker
        self._export_thread.start()

    def _on_export_progress(self, cur, total):
        self._progress.setMaximum(total)
        self._progress.setValue(cur)

    def _on_export_done(self, path):
        self._progress.setVisible(False)
        self._btn_export.setEnabled(True)
        self._status.setText(f"Export complete:\n{path}")

    def _on_export_error(self, msg):
        self._progress.setVisible(False)
        self._btn_export.setEnabled(True)
        self._status.setText(f"Export error:\n{msg}")

    def closeEvent(self, event):
        self._stop_playback()
        super().closeEvent(event)
