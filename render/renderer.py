import moderngl
import numpy as np
import math
import os
from collections import deque
from pathlib import Path
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
from config.defaults import PALETTES, SENSITIVITY
from core.layer import (
    Layer, BGLayer, LayerManager, BLEND_ADDITIVE, TEXT_MODE, IMAGE_MODE,
    PARTICLE_MODE,
)
from render.modes import HaloSineMode, FlatSineMode
from render.video import VideoFrameSource


SHADER_DIR = Path(__file__).parent / "shaders"
_DEFAULT_PALETTE = list(PALETTES.values())[0]

_BASS_HIST_N  = 64    # history frames
_BASS_HIST_W  = 256   # fixed width = u_bars max
_BASS2_DECAY  = 0.93  # EMA decay for Halo Bass 2 (bidirectional smooth)


def _color_framebuffer(ctx: moderngl.Context, width: int, height: int):
    """Create a framebuffer whose color texture scales without nearest-pixel blocks."""
    texture = ctx.texture((width, height), 3)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False
    return ctx.framebuffer(color_attachments=[texture])


def _render_scale(width: int, height: int, requested: int,
                  max_pixels: int = 3840 * 2160) -> int:
    scale = max(1, int(requested))
    while scale > 1 and (width * scale) * (height * scale) > max_pixels:
        scale -= 1
    return scale


def _load_rgba_image(path: str) -> Image.Image:
    """Load raster formats with Pillow and SVG through Qt's bundled renderer."""
    if Path(path).suffix.lower() != ".svg":
        return Image.open(path).convert("RGBA")
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    svg = QSvgRenderer(path)
    if not svg.isValid():
        raise ValueError(f"Invalid SVG asset: {path}")
    size = svg.defaultSize()
    width, height = max(size.width(), 1), max(size.height(), 1)
    canvas = QImage(width, height, QImage.Format_RGBA8888)
    canvas.fill(0)
    painter = QPainter(canvas)
    svg.render(painter, QRectF(0, 0, width, height))
    painter.end()
    return Image.frombytes("RGBA", (width, height), canvas.bits().tobytes())


def _font_variant(font_path: str, bold: bool, italic: bool) -> str:
    """Resolve style variants for bundled/common Windows font filenames."""
    if not (bold or italic) or Path(font_path).is_absolute():
        return font_path
    variants = {
        "arial.ttf": ("arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
        "times.ttf": ("timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
        "cour.ttf": ("courbd.ttf", "couri.ttf", "courbi.ttf"),
        "verdana.ttf": ("verdanab.ttf", "verdanai.ttf", "verdanaz.ttf"),
        "georgia.ttf": ("georgiab.ttf", "georgiai.ttf", "georgiaz.ttf"),
        "comic.ttf": ("comicbd.ttf", "comici.ttf", "comicz.ttf"),
    }
    choices = variants.get(Path(font_path).name.lower())
    if not choices:
        return font_path
    return choices[2] if bold and italic else choices[0] if bold else choices[1]


def _load_shader(name: str) -> str:
    return (SHADER_DIR / name).read_text()


class LayerState:
    """Per-layer mutable render state: audio history, PIL mode instances, FBO.

    History textures are canvas-size-independent (256×64) so they survive a
    canvas resize; only the layer FBO is recreated.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.prev_bass: float = 0.0
        self.kick_accum: float = 0.0
        self.kick_bass_buf: deque | None = None
        self.kick_cooldown: int = 0
        self.kick_bg_ema: float = 0.0

        # Mode 10 shockwaves: ring buffer of spawn times (seconds) + adaptive
        # bass-onset detector (independent of the Tunnel kick settings, so it
        # fires reliably on sustained techno).
        self.shock_times = np.full(8, -100.0, dtype="f4")
        self.shock_ptr: int = 0
        self.shock_avg: float = 0.0       # slow-tracking background pulse level
        self.shock_armed: bool = True     # hysteresis gate — one wave per kick
        self.shock_cooldown: int = 0

        self.bass_smooth2 = np.zeros(_BASS_HIST_W, dtype=np.float32)
        self.bass_history  = np.zeros((_BASS_HIST_N, _BASS_HIST_W), dtype=np.float32)
        self.bass_history2 = np.zeros((_BASS_HIST_N, _BASS_HIST_W), dtype=np.float32)

        self.bass_hist_tex = ctx.texture(
            (_BASS_HIST_W, _BASS_HIST_N), 1, self.bass_history.tobytes(), dtype='f4')
        self.bass_hist_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bass_hist_tex.repeat_x = False
        self.bass_hist_tex.repeat_y = False

        self.bass_hist2_tex = ctx.texture(
            (_BASS_HIST_W, _BASS_HIST_N), 1, self.bass_history2.tobytes(), dtype='f4')
        self.bass_hist2_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bass_hist2_tex.repeat_x = False
        self.bass_hist2_tex.repeat_y = False

        self.halo_sine = HaloSineMode()
        self.flat_sine = FlatSineMode()

        self.fbo: moderngl.Framebuffer | None = None
        self.resize(width, height)

    def resize(self, width: int, height: int) -> None:
        if self.fbo:
            self.fbo.color_attachments[0].release()
            self.fbo.release()
        self.fbo = _color_framebuffer(self.ctx, width, height)

    def release(self) -> None:
        self.bass_hist_tex.release()
        self.bass_hist2_tex.release()
        if self.fbo:
            self.fbo.color_attachments[0].release()
            self.fbo.release()
            self.fbo = None


class Renderer:
    def __init__(self, width: int, height: int, ctx: moderngl.Context = None,
                 supersampling: int = 2):
        self.output_width = width
        self.output_height = height
        self.width = width
        self.height = height
        self.supersampling = max(1, int(supersampling))
        self._bg_texture: moderngl.Texture | None = None
        self._bg_img_aspect: float = 1.0
        self._bg_video: VideoFrameSource | None = None
        self._bg_video_start_time: float | None = None
        self._text_textures: dict[int, moderngl.Texture] = {}
        self._text_signatures: dict[int, tuple] = {}
        self._image_textures: dict[int, moderngl.Texture] = {}
        self._image_paths: dict[int, str] = {}
        self._particle_textures: dict[int, moderngl.Texture] = {}
        self._center_texture: moderngl.Texture | None = None
        self._center_pil: "Image.Image | None" = None

        if ctx is None:
            # Colab and other headless Linux runners have no X11 display.  Let
            # callers select EGL while preserving the desktop default.
            backend = os.environ.get("MGL_BACKEND")
            self.ctx = moderngl.create_standalone_context(
                **({"backend": backend} if backend else {}))
            self._owns_ctx = True
        else:
            self.ctx = ctx
            self._owns_ctx = False

        self._states: dict[int, LayerState] = {}

        self._build_programs()
        self._build_quad()
        self.scene_fbo: moderngl.Framebuffer | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.resize(width, height)

    # ── Setup ────────────────────────────────────────────────────
    def _build_programs(self):
        self.prog = self.ctx.program(
            vertex_shader=_load_shader("radial.vert"),
            fragment_shader=_load_shader("radial.frag"),
        )
        self.composite_prog = self.ctx.program(
            vertex_shader=_load_shader("composite.vert"),
            fragment_shader=_load_shader("composite.frag"),
        )
        self.bg_prog = self.ctx.program(
            vertex_shader=_load_shader("radial.vert"),
            fragment_shader=_load_shader("bg.frag"),
        )
        self.fxaa_prog = self.ctx.program(
            vertex_shader=_load_shader("radial.vert"),
            fragment_shader=_load_shader("fxaa.frag"),
        )
        self.overlay_prog = self.ctx.program(
            vertex_shader=_load_shader("radial.vert"),
            fragment_shader=_load_shader("overlay.frag"),
        )
        self.image_prog = self.ctx.program(
            vertex_shader=_load_shader("composite.vert"),
            fragment_shader=_load_shader("overlay.frag"),
        )
        self.line_prog = self.ctx.program(
            vertex_shader=_load_shader("composite.vert"),
            fragment_shader=(
                "#version 330 core\n"
                "out vec4 fragColor;\n"
                "uniform vec3 u_color;\n"
                "void main() { fragColor = vec4(u_color, 1.0); }\n"
            ),
        )

    def _build_quad(self):
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype="f4")
        self.vbo = self.ctx.buffer(vertices)
        self.vao           = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_position")
        self.composite_vao = self.ctx.simple_vertex_array(self.composite_prog, self.vbo, "in_position")
        self.bg_vao        = self.ctx.simple_vertex_array(self.bg_prog, self.vbo, "in_position")
        self.fxaa_vao      = self.ctx.simple_vertex_array(self.fxaa_prog, self.vbo, "in_position")
        self.overlay_vao   = self.ctx.simple_vertex_array(self.overlay_prog, self.vbo, "in_position")
        self.image_vao     = self.ctx.simple_vertex_array(self.image_prog, self.vbo, "in_position")

        # Line-loop quad (CCW order) for the selection outline
        line_verts = np.array([
            -1.0, -1.0,  1.0, -1.0,  1.0, 1.0,  -1.0, 1.0,
        ], dtype="f4")
        self.line_vbo = self.ctx.buffer(line_verts)
        self.line_vao = self.ctx.simple_vertex_array(self.line_prog, self.line_vbo, "in_position")

    def resize(self, width: int, height: int) -> None:
        self.output_width = max(int(width), 1)
        self.output_height = max(int(height), 1)

        # Render up to 2x in each dimension, then downsample. Cap the working
        # surface near 8.3 MP so 4K exports stay within a practical GPU budget.
        max_render_pixels = 3840 * 2160
        scale = _render_scale(
            self.output_width, self.output_height, self.supersampling,
            max_render_pixels)
        self.render_scale = scale
        self.width = self.output_width * scale
        self.height = self.output_height * scale
        for framebuffer in (self.scene_fbo, self.fbo):
            if framebuffer:
                framebuffer.color_attachments[0].release()
                framebuffer.release()
        self.scene_fbo = _color_framebuffer(self.ctx, self.width, self.height)
        self.fbo = _color_framebuffer(self.ctx, self.output_width, self.output_height)
        for st in self._states.values():
            st.resize(self.width, self.height)

    def _state_for(self, layer_id: int) -> LayerState:
        st = self._states.get(layer_id)
        if st is None:
            st = LayerState(self.ctx, self.width, self.height)
            self._states[layer_id] = st
        return st

    def release_layer_state(self, layer_id: int) -> None:
        st = self._states.pop(layer_id, None)
        if st:
            st.release()
        texture = self._text_textures.pop(layer_id, None)
        if texture:
            texture.release()
        self._text_signatures.pop(layer_id, None)
        image_texture = self._image_textures.pop(layer_id, None)
        if image_texture:
            image_texture.release()
        self._image_paths.pop(layer_id, None)
        particle_texture = self._particle_textures.pop(layer_id, None)
        if particle_texture:
            particle_texture.release()

    # ── Image management (global bg + center) ────────────────────
    def load_background(self, path: str, fps: int = 60) -> None:
        if self._bg_texture:
            self._bg_texture.release()
            self._bg_texture = None
        if self._bg_video:
            self._bg_video.close()
            self._bg_video = None
        self._bg_video_start_time = None
        if Path(path).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            self._bg_video = VideoFrameSource(path, fps=fps)
            self._bg_img_aspect = self._bg_video.width / max(self._bg_video.height, 1)
            self._bg_texture = self.ctx.texture(
                (self._bg_video.width, self._bg_video.height), 3)
            self._bg_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._bg_texture.repeat_x = False
            self._bg_texture.repeat_y = False
            return
        img = Image.open(path).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
        self._bg_img_aspect = img.size[0] / max(img.size[1], 1)
        self._bg_texture = self.ctx.texture(img.size, 3, img.tobytes())
        self._bg_texture.build_mipmaps()
        self._bg_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

    def clear_background(self) -> None:
        if self._bg_texture:
            self._bg_texture.release()
            self._bg_texture = None
        if self._bg_video:
            self._bg_video.close()
            self._bg_video = None
        self._bg_video_start_time = None

    def load_center_image(self, path: str) -> None:
        if self._center_texture:
            self._center_texture.release()
        pil_src = Image.open(path).convert("RGBA")
        self._center_pil = pil_src
        img = pil_src.transpose(Image.FLIP_TOP_BOTTOM)
        self._center_texture = self.ctx.texture(img.size, 4, img.tobytes())
        self._center_texture.build_mipmaps()
        self._center_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

    def clear_center_image(self) -> None:
        if self._center_texture:
            self._center_texture.release()
            self._center_texture = None
        self._center_pil = None

    # ── Composition pipeline ─────────────────────────────────────
    def render_composition(self, lm: LayerManager, bars: np.ndarray, pulse: float,
                           time: float = 0.0) -> None:
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._render_bg(lm.bg, pulse, time)
        self._render_text(lm.bg, 0)
        for layer in lm.active_layers():
            if layer.mode == TEXT_MODE:
                self._render_text(layer, layer.id)
                continue
            if layer.mode == IMAGE_MODE:
                self._render_image(layer, pulse)
                continue
            if layer.mode == PARTICLE_MODE:
                self._render_particles(layer, bars, pulse, time)
                continue
            st = self._state_for(layer.id)
            self._render_layer(st, layer, bars, pulse, time)
            self._composite(st, layer)
        self._apply_fxaa()

    def render_frame(self, bars: np.ndarray, pulse: float, time: float = 0.0) -> None:
        """Backward-compatible single-layer render entry point."""
        if not hasattr(self, "_single_layer_manager"):
            self._single_layer_manager = LayerManager()
            self._single_layer_manager.add_layer()
        self.render_composition(self._single_layer_manager, bars, pulse, time)

    def _render_bg(self, bg: BGLayer, pulse: float, time: float) -> None:
        self.scene_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        has_img = 1 if (bg.image_path and self._bg_texture) else 0
        if has_img:
            if self._bg_video:
                if self._bg_video_start_time is None:
                    self._bg_video_start_time = time
                frame = self._bg_video.frame_at(time - self._bg_video_start_time)
                if frame:
                    self._bg_texture.write(frame)
            self._bg_texture.use(location=0)
            self.bg_prog["u_tex"].value = 0
        self.bg_prog["u_has_image"].value     = has_img
        self.bg_prog["u_opacity"].value       = float(bg.opacity)
        self.bg_prog["u_img_aspect"].value    = float(self._bg_img_aspect)
        self.bg_prog["u_canvas_aspect"].value = float(self.width / self.height)
        self.bg_prog["u_fit_mode"].value      = int(bg.fit_mode)
        self.bg_prog["u_crop_zoom"].value    = float(bg.crop_zoom)
        self.bg_prog["u_crop_offset"].value  = (float(bg.crop_x), float(bg.crop_y))
        zoom = 1.0 + (pulse * bg.zoom_intensity * 0.10 if bg.zoom_reactive else 0.0)
        self.bg_prog["u_zoom"].value = float(zoom)
        self.bg_vao.render(moderngl.TRIANGLE_STRIP)

    def _render_text(self, config, cache_key: int) -> None:
        if not config.text.strip():
            return
        is_layer = isinstance(config, Layer)
        scale = config.scale if is_layer else 1.0
        x_pos = 0.5 + config.x * 0.5 if is_layer else config.text_x
        y_pos = 0.5 - config.y * 0.5 if is_layer else config.text_y
        opacity = config.text_opacity * (config.opacity if is_layer else 1.0)
        signature = (
            self.width, self.height, config.text, config.text_size, x_pos, y_pos,
            tuple(config.text_color), opacity, config.text_stroke, scale,
            config.text_font, config.text_bold, config.text_italic, config.text_align,
            config.text_letter_spacing, config.text_line_spacing,
            tuple(config.text_stroke_color), config.text_rotation, config.text_shadow,
            tuple(config.text_shadow_color), config.text_shadow_x, config.text_shadow_y)
        if signature != self._text_signatures.get(cache_key):
            canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
            scaled_size = max(8, round(config.text_size * scale * self.height / 1080))
            try:
                font = ImageFont.truetype(
                    _font_variant(config.text_font, config.text_bold, config.text_italic),
                    scaled_size)
            except OSError:
                try:
                    font = ImageFont.truetype(config.text_font, scaled_size)
                except OSError:
                    font = ImageFont.load_default(size=scaled_size)
            color = tuple(round(c * 255) for c in config.text_color) + (round(opacity * 255),)
            stroke_color = tuple(round(c * 255) for c in config.text_stroke_color) + (color[3],)
            shadow_color = tuple(round(c * 255) for c in config.text_shadow_color) + (color[3],)
            stroke = max(0, round(config.text_stroke * scale * self.height / 1080))
            letter_spacing = round(config.text_letter_spacing * scale * self.height / 1080)
            line_spacing = round(config.text_line_spacing * scale * self.height / 1080)
            measure = ImageDraw.Draw(canvas)
            lines = config.text.splitlines() or [""]
            widths = [sum(measure.textlength(ch, font=font) for ch in line)
                      + letter_spacing * max(len(line) - 1, 0) for line in lines]
            font_box = measure.textbbox((0, 0), "Ag", font=font, stroke_width=stroke)
            line_height = max(1, font_box[3] - font_box[1])
            shadow_pad = max(abs(config.text_shadow_x), abs(config.text_shadow_y)) if config.text_shadow else 0
            pad = max(8, stroke * 3 + shadow_pad)
            content_width = max(1, round(max(widths, default=1)))
            content_height = line_height * len(lines) + line_spacing * max(len(lines) - 1, 0)
            patch = Image.new("RGBA", (content_width + pad * 2,
                                       content_height + pad * 2), (0, 0, 0, 0))
            draw = ImageDraw.Draw(patch)

            def draw_lines(offset_x: int, offset_y: int, fill, outline):
                for line_index, (line, width) in enumerate(zip(lines, widths)):
                    if config.text_align == "left":
                        cursor_x = pad
                    elif config.text_align == "right":
                        cursor_x = pad + content_width - width
                    else:
                        cursor_x = pad + (content_width - width) / 2
                    baseline_y = pad + line_index * (line_height + line_spacing) - font_box[1]
                    for char in line:
                        draw.text((cursor_x + offset_x, baseline_y + offset_y), char,
                                  font=font, fill=fill, stroke_width=stroke,
                                  stroke_fill=outline)
                        cursor_x += draw.textlength(char, font=font) + letter_spacing

            if config.text_shadow:
                draw_lines(config.text_shadow_x, config.text_shadow_y,
                           shadow_color, shadow_color)
            draw_lines(0, 0, color, stroke_color)
            if config.text_rotation:
                patch = patch.rotate(config.text_rotation, expand=True, resample=Image.Resampling.BICUBIC)
            x = round(x_pos * self.width - patch.width / 2)
            y = round(y_pos * self.height - patch.height / 2)
            canvas.alpha_composite(patch, (x, y))
            canvas = canvas.transpose(Image.FLIP_TOP_BOTTOM)
            old_texture = self._text_textures.get(cache_key)
            if old_texture:
                old_texture.release()
            texture = self.ctx.texture(canvas.size, 4, canvas.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._text_textures[cache_key] = texture
            self._text_signatures[cache_key] = signature

        self.scene_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._text_textures[cache_key].use(location=0)
        self.overlay_prog["u_tex"].value = 0
        self.overlay_prog["u_opacity"].value = 1.0
        self.overlay_vao.render(moderngl.TRIANGLE_STRIP)

    def _render_image(self, layer: Layer, pulse: float) -> None:
        if not layer.image_path:
            return
        if self._image_paths.get(layer.id) != layer.image_path:
            old = self._image_textures.pop(layer.id, None)
            if old:
                old.release()
            img = _load_rgba_image(layer.image_path).transpose(Image.FLIP_TOP_BOTTOM)
            texture = self.ctx.texture(img.size, 4, img.tobytes())
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            self._image_textures[layer.id] = texture
            self._image_paths[layer.id] = layer.image_path

        beat_scale = (1.0 + pulse * layer.image_beat_intensity
                      if layer.image_beat_reactive else 1.0)
        self.scene_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._image_textures[layer.id].use(location=0)
        self.image_prog["u_tex"].value = 0
        self.image_prog["u_offset"].value = (float(layer.x), float(layer.y))
        self.image_prog["u_scale"].value = (
            float(max(layer.image_width * beat_scale, 0.001)),
            float(max(layer.image_height * beat_scale, 0.001)))
        self.image_prog["u_opacity"].value = float(layer.opacity)
        self.image_vao.render(moderngl.TRIANGLE_STRIP)

    def _render_particles(self, layer: Layer, bars: np.ndarray,
                          pulse: float, time: float) -> None:
        """Render the reference's floating-particle effect as a real RGBA layer."""
        scale = min(1.0, 1920 / max(self.width, 1), 1080 / max(self.height, 1))
        width = max(1, round(self.width * scale))
        height = max(1, round(self.height * scale))
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        particles = ImageDraw.Draw(canvas)
        glow_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        glow = ImageDraw.Draw(glow_canvas)

        beat_boost = 1.0
        count = max(10, min(200, int(layer.particle_count)))

        directions = {
            "up": (0.0, -1.0), "down": (0.0, 1.0),
            "left": (-1.0, 0.0), "right": (1.0, 0.0),
            "up-left": (-0.707, -0.707), "up-right": (0.707, -0.707),
            "down-left": (-0.707, 0.707), "down-right": (0.707, 0.707),
        }
        colors = layer.particle_colors or [[1.0, 1.0, 1.0]]

        for i in range(count):
            px = ((i * 47) % 100) / 100.0
            py = ((i * 83) % 100) / 100.0
            particle_size = (0.6 + (i % 5) * 0.5
                             if layer.particle_random_size else 1.0)
            particle_speed = 0.15 + (i % 7) * 0.04
            angle = i * 2.399963229728653
            dx, dy = directions.get(layer.particle_direction,
                                    (math.cos(angle), math.sin(angle)))

            if layer.particle_direction == "center-out":
                px, py = 0.5 + math.cos(angle) * 0.04, 0.5 + math.sin(angle) * 0.04
                dx, dy = math.cos(angle), math.sin(angle)
            elif layer.particle_direction == "edges-in":
                px, py = 0.5 + math.cos(angle) * 0.52, 0.5 + math.sin(angle) * 0.52
                dx, dy = -math.cos(angle), -math.sin(angle)
            else:
                origin = layer.particle_origin
                if origin == "random":
                    origin = ("top", "bottom", "left", "right", "center", "full")[i % 6]
                if origin == "top":
                    py *= 0.12
                elif origin == "bottom":
                    py = 0.88 + py * 0.12
                elif origin == "left":
                    px *= 0.12
                elif origin == "right":
                    px = 0.88 + px * 0.12
                elif origin == "center":
                    px, py = 0.42 + px * 0.16, 0.42 + py * 0.16

            travel = time * particle_speed * layer.particle_speed * 0.18
            x = ((px + dx * travel + 2.0) % 1.0) * width
            y = ((py + dy * travel + 2.0) % 1.0) * height
            random_scale = (0.35 + ((i * 37) % 100) / 100.0 * 0.65
                            if layer.particle_random_size else 1.0)
            radius = max(0.5, particle_size * layer.particle_size
                         * random_scale * beat_boost * scale)

            if layer.particle_color_mode == "rainbow":
                hue = (i * 31 + time * layer.particle_speed * 60) % 360
                color = ImageColor.getrgb(f"hsl({round(hue)}, 90%, 65%)")
            else:
                selected = colors[i % len(colors)] if layer.particle_color_mode == "palette" else colors[0]
                color = tuple(max(0, min(255, round(component * 255))) for component in selected[:3])
            alpha = max(0, min(255, round(
                255 * layer.opacity * beat_boost
                * (0.40 + (i % 4) * 0.18))))
            bounds = (x - radius, y - radius, x + radius, y + radius)
            particles.ellipse(bounds, fill=(*color, alpha))
            if layer.particle_glossy > 0:
                glow.ellipse(bounds, fill=(*color, round(alpha * 0.72)))

        if layer.particle_glossy > 0:
            blur = max(0.5, layer.particle_glossy * 7.0 * scale)
            canvas = Image.alpha_composite(
                glow_canvas.filter(ImageFilter.GaussianBlur(blur)), canvas)

        canvas = canvas.transpose(Image.FLIP_TOP_BOTTOM)
        texture = self._particle_textures.get(layer.id)
        if texture is None or texture.size != canvas.size:
            if texture:
                texture.release()
            texture = self.ctx.texture(canvas.size, 4)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._particle_textures[layer.id] = texture
        texture.write(canvas.tobytes())

        self.scene_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = ((moderngl.SRC_ALPHA, moderngl.ONE)
                               if layer.blend_mode == BLEND_ADDITIVE
                               else (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA))
        texture.use(location=0)
        self.image_prog["u_tex"].value = 0
        self.image_prog["u_offset"].value = (float(layer.x), float(layer.y))
        self.image_prog["u_scale"].value = (float(layer.scale), float(layer.scale))
        self.image_prog["u_opacity"].value = 1.0
        self.image_vao.render(moderngl.TRIANGLE_STRIP)

    def _composite(self, st: LayerState, layer: Layer) -> None:
        self.scene_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        if layer.blend_mode == BLEND_ADDITIVE:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        else:
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        st.fbo.color_attachments[0].use(location=0)
        self.composite_prog["u_tex"].value     = 0
        self.composite_prog["u_offset"].value  = (float(layer.x), float(layer.y))
        self.composite_prog["u_scale"].value   = (float(layer.scale), float(layer.scale))
        self.composite_prog["u_opacity"].value = float(layer.opacity)
        self.composite_vao.render(moderngl.TRIANGLE_STRIP)

    def _apply_fxaa(self) -> None:
        """Smooth high-contrast diagonal and curved edges in the composed frame."""
        self.fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.scene_fbo.color_attachments[0].use(location=0)
        self.fxaa_prog["u_tex"].value = 0
        self.fxaa_prog["u_inv_resolution"].value = (
            1.0 / max(self.width, 1), 1.0 / max(self.height, 1))
        self.fxaa_vao.render(moderngl.TRIANGLE_STRIP)

    # ── Single-layer viz render (into st.fbo, on black) ──────────
    def _render_layer(self, st: LayerState, layer: Layer,
                      bars: np.ndarray, pulse: float, time: float) -> None:
        palette = layer.palette if layer.palette else _DEFAULT_PALETTE
        viz_type = layer.mode

        # Resample shared FFT bars to this layer's bar count
        if layer.num_bars != len(bars) and len(bars) > 1:
            xs = np.linspace(0, len(bars) - 1, layer.num_bars)
            lbars = np.interp(xs, np.arange(len(bars)), bars).astype(np.float32)
        else:
            lbars = bars

        st.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        bar_data = np.zeros(256, dtype="f4")
        n = min(len(lbars), 256)
        bar_data[:n] = lbars[:n]

        # Visual layers never draw the bg image; render on pure black for clean additive
        self.prog["u_has_bg"].value = 0
        self.prog["u_force_black_bg"].value = 1
        self.prog["u_bg_pulse_enabled"].value = 0
        self.prog["u_bg_pulse_intensity"].value = 0.0

        if self._center_texture:
            self._center_texture.use(location=1)
            self.prog["u_center_texture"].value = 1
            self.prog["u_has_center"].value = 1
        else:
            self.prog["u_has_center"].value = 0

        for i, rgb in enumerate(palette):
            self.prog[f"u_pal{i}"].value = tuple(rgb)

        self.prog["u_num_bars"].value = int(layer.num_bars)
        self.prog["u_bars"].value = tuple(bar_data.tolist())
        self.prog["u_pulse"].value = float(pulse)
        self.prog["u_pulse_intensity"].value = float(layer.pulse_intensity)
        self.prog["u_max_bar_height"].value = float(layer.max_bar_height)
        self.prog["u_circle_radius"].value = float(layer.circle_radius)
        self.prog["u_bar_width"].value = float(layer.bar_width)
        self.prog["u_bar_shape"].value = int(layer.bar_shape)
        self.prog["u_aspect"].value = float(self.width / self.height)
        self.prog["u_sensitivity"].value = float(layer.sensitivity)
        self.prog["u_viz_type"].value = int(viz_type)
        self.prog["u_rotation"].value = float(layer.rotation)
        self.prog["u_mirror"].value = int(layer.mirror)
        self.prog["u_pal_mode"].value = int(layer.pal_mode)
        self.prog["u_halo_r_base"].value = float(layer.halo_r_base)
        self.prog["u_flash_enabled"].value = int(layer.flash)
        self.prog["u_flash_intensity"].value = float(layer.flash_intensity)

        # Nuclear Shockwave (mode 10) params
        self.prog["u_nuke_speed"].value = float(layer.nuke_speed)
        self.prog["u_nuke_life"].value = float(layer.nuke_life)
        self.prog["u_nuke_width"].value = float(layer.nuke_width)
        self.prog["u_nuke_flash"].value = float(layer.nuke_flash)
        self.prog["u_nuke_bg"].value = float(layer.nuke_bg)

        # Void Pull (mode 11) params
        self.prog["u_void_speed"].value = float(layer.void_speed)
        self.prog["u_void_pull"].value = float(layer.void_pull)
        self.prog["u_void_rays"].value = float(layer.void_rays)
        self.prog["u_void_arms"].value = int(layer.void_arms)

        # Lissajous / Halo animation uniforms
        bar_slice = lbars[:n]
        total_energy = float(bar_slice.sum()) + 1e-6
        centroid = float(np.sum(bar_slice * np.arange(n)) / total_energy) / max(n, 1)
        lissa_energy = float(np.clip(bar_slice.mean() * layer.sensitivity * 1.5 + 0.15, 0.1, 0.85))
        self.prog["u_time"].value = float(time)
        self.prog["u_lissa_a"].value = 1.0 + centroid * 2.0
        self.prog["u_lissa_b"].value = 1.0 + (1.0 - centroid) * 2.0
        self.prog["u_lissa_energy"].value = lissa_energy

        # Bass waterfall history (mode 5)
        st.bass_history = np.roll(st.bass_history, 1, axis=0)
        st.bass_history[0, :n] = lbars[:n]
        st.bass_hist_tex.write(st.bass_history.tobytes())
        st.bass_hist_tex.use(location=2)
        self.prog["u_bass_history"].value = 2

        # Bass history 2 (mode 7) — bidirectional EMA
        st.bass_smooth2[:n] = (st.bass_smooth2[:n] * _BASS2_DECAY
                               + lbars[:n] * (1.0 - _BASS2_DECAY))
        st.bass_history2 = np.roll(st.bass_history2, 1, axis=0)
        st.bass_history2[0, :n] = st.bass_smooth2[:n]
        st.bass_hist2_tex.write(st.bass_history2.tobytes())
        st.bass_hist2_tex.use(location=3)
        self.prog["u_bass_history2"].value = 3

        # Audio band decomposition + kick detection (Tunnel Arcade)
        self._update_kick(st, layer, lbars, time, pulse)

        # Mode 6: center image composited AFTER spline in PIL — skip in GLSL
        if viz_type == 6 and self._center_pil is not None:
            self.prog["u_has_center"].value = 0

        self.vao.render(moderngl.TRIANGLE_STRIP)

        # PIL post passes
        if viz_type == 6:
            self._post_halo_sine(st, layer, lbars, pulse, palette)
        elif viz_type == 9:
            self._post_flat_sine(st, layer, lbars, palette)

    def _update_kick(self, st: LayerState, layer: Layer, bars: np.ndarray,
                     time: float, pulse: float) -> None:
        n_used = min(len(bars), layer.num_bars)
        bar_sl = bars[:n_used] if n_used > 0 else np.zeros(1, dtype="f4")
        n_bass = max(1, int(n_used * 0.30))
        n_mid  = max(n_bass + 1, int(n_used * 0.70))
        bass_v = float(np.mean(bar_sl[:n_bass]))
        mid_v  = float(np.mean(bar_sl[n_bass:n_mid]))
        high_v = float(np.mean(bar_sl[n_mid:]) if n_mid < n_used else 0.0)

        i_lo = max(0, int(n_used * layer.tunnel_kick_freq_lo / 100.0))
        i_hi = max(i_lo + 1, int(n_used * layer.tunnel_kick_freq_hi / 100.0))
        band = bar_sl[i_lo:i_hi] if i_hi <= n_used else bar_sl[i_lo:]
        band_energy = float(band.max()) if len(band) > 0 else 0.0

        sens = layer.tunnel_kick_sensitivity
        thr  = layer.tunnel_kick_threshold
        mode = layer.tunnel_kick_mode

        if mode == 1:
            if st.kick_bass_buf is None:
                st.kick_bass_buf = deque(maxlen=90)
            st.kick_bass_buf.append(bass_v)
            if st.kick_cooldown > 0:
                st.kick_cooldown -= 1
            rolling_max = max(st.kick_bass_buf) if st.kick_bass_buf else 1e-6
            if bass_v >= rolling_max * 0.75 and st.kick_cooldown == 0 and rolling_max > 1e-4:
                kick_n = float(np.clip(bass_v / max(rolling_max, 1e-6) * sens, 0.0, 1.0))
                st.kick_cooldown = 15
            else:
                kick_n = 0.0
        elif mode == 2:
            bg_alpha = 0.03
            st.kick_bg_ema = st.kick_bg_ema * (1.0 - bg_alpha) + band_energy * bg_alpha
            if st.kick_cooldown > 0:
                st.kick_cooldown -= 1
            ratio = band_energy / max(st.kick_bg_ema, 1e-5)
            if ratio >= thr and st.kick_cooldown == 0:
                kick_n = float(np.clip((ratio / max(thr, 1e-3) - 1.0) * sens, 0.0, 1.0))
                st.kick_cooldown = layer.tunnel_kick_cooldown
            else:
                kick_n = 0.0
        elif mode == 3:
            if st.kick_cooldown > 0:
                st.kick_cooldown -= 1
            if band_energy >= thr and st.kick_cooldown == 0:
                kick_n = float(np.clip(band_energy * sens, 0.0, 1.0))
                st.kick_cooldown = layer.tunnel_kick_cooldown
            else:
                kick_n = 0.0
        else:
            kick_n = float(np.clip((bass_v - st.prev_bass * 1.3) * 4.0 * sens, 0.0, 1.0))

        st.prev_bass = bass_v
        st.kick_accum = max(kick_n, st.kick_accum * 0.88)

        # Mode 10: spawn exactly one shockwave per kick. Adaptive onset on the
        # FFT pulse (sub-bass envelope) with hysteresis: the detector arms when
        # pulse falls back near its slow-tracking average, then fires once when it
        # rises a margin above that average. The slow-release pulse tail can't
        # re-trigger because the gate stays disarmed until pulse drops again.
        st.shock_avg = st.shock_avg * 0.96 + pulse * 0.04
        margin = layer.nuke_kick_threshold        # margin above background level
        hi = st.shock_avg + margin
        lo = st.shock_avg + margin * 0.4
        if st.shock_cooldown > 0:
            st.shock_cooldown -= 1
        if not st.shock_armed and pulse < lo:
            st.shock_armed = True
        if st.shock_armed and pulse > hi and pulse > 0.05 and st.shock_cooldown == 0:
            st.shock_times[st.shock_ptr] = float(time)
            st.shock_ptr = (st.shock_ptr + 1) % len(st.shock_times)
            st.shock_armed = False
            st.shock_cooldown = 6                  # safety floor (~0.1 s)

        self.prog["u_bass"].value = bass_v
        self.prog["u_mid"].value = mid_v
        self.prog["u_high"].value = high_v
        self.prog["u_kick"].value = kick_n
        self.prog["u_kick_accum"].value = float(st.kick_accum)
        self.prog["u_shock_times"].value = tuple(st.shock_times.tolist())
        self.prog["u_tunnel_sides"].value = int(layer.tunnel_sides)
        self.prog["u_tunnel_rings"].value = int(layer.tunnel_rings)
        self.prog["u_tunnel_speed"].value = float(layer.tunnel_speed)
        self.prog["u_tunnel_kick_zoom"].value = float(layer.tunnel_kick_zoom)
        self.prog["u_tunnel_chroma"].value = float(layer.tunnel_chroma)
        self.prog["u_tunnel_bass_speed"].value = float(layer.tunnel_bass_speed)

    def _post_halo_sine(self, st: LayerState, layer: Layer,
                        bars: np.ndarray, pulse: float, palette: list) -> None:
        raw = st.fbo.read(components=3)
        pixels_bt = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        pixels_tb = np.ascontiguousarray(pixels_bt[::-1])
        result_tb = st.halo_sine.draw_overlay(
            pixels_tb, bars=bars,
            r_base=layer.halo_r_base, amplitude_max=layer.halo_amplitude,
            n_points=layer.halo_n_points, glow_layers=layer.halo_glow_layers,
            smoothing_decay=layer.halo_smoothing_decay,
            sensitivity=layer.sensitivity, palette=palette,
            rotation=layer.rotation,
            fill_opacity=layer.halo_fill_opacity,
            pal_mode=layer.pal_mode,
            spline_gap=layer.halo_spline_gap,
        )
        r_px = layer.halo_r_base * self.height / 2
        if self._center_pil is not None:
            result_tb = self._composite_center_circle(
                result_tb, self._center_pil, r_px,
                pulse=pulse, pulse_intensity=layer.pulse_intensity,
                pixel_size=layer.halo_pixel_size,
            )
        result_tb = self._draw_ring_glow(
            result_tb, r_px, palette,
            pulse=pulse, pulse_intensity=layer.pulse_intensity,
        )
        result_bt = np.ascontiguousarray(result_tb[::-1])
        st.fbo.color_attachments[0].write(result_bt.tobytes())

    def _post_flat_sine(self, st: LayerState, layer: Layer,
                        bars: np.ndarray, palette: list) -> None:
        raw = st.fbo.read(components=3)
        pixels_bt = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        pixels_tb = np.ascontiguousarray(pixels_bt[::-1])
        result_tb = st.flat_sine.draw_overlay(
            pixels_tb, bars=bars,
            amplitude_max=layer.halo_amplitude,
            n_points=layer.halo_n_points,
            glow_layers=layer.halo_glow_layers,
            smoothing_decay=layer.halo_smoothing_decay,
            sensitivity=layer.sensitivity,
            palette=palette,
            fill_opacity=layer.halo_fill_opacity,
            pal_mode=layer.pal_mode,
        )
        result_bt = np.ascontiguousarray(result_tb[::-1])
        st.fbo.color_attachments[0].write(result_bt.tobytes())

    # ── PIL helpers (stateless) ──────────────────────────────────
    def _composite_center_circle(self, frame_tb: np.ndarray,
                                  center_pil: "Image.Image",
                                  r_px: float,
                                  pulse: float = 0.0,
                                  pulse_intensity: float = 1.0,
                                  pixel_size: int = 1) -> np.ndarray:
        H, W = frame_tb.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        r = max(1, int(r_px * (1.0 + 0.10 * pulse * pulse_intensity)))
        diam = r * 2
        resized = center_pil.resize((diam, diam), Image.LANCZOS)
        if pixel_size > 1:
            small_d = max(1, diam // pixel_size)
            resized = resized.resize((small_d, small_d), Image.NEAREST).resize((diam, diam), Image.NEAREST)
        OVR = 4
        mask_big = Image.new("L", (diam * OVR, diam * OVR), 0)
        draw = ImageDraw.Draw(mask_big)
        draw.ellipse([0, 0, diam * OVR - 1, diam * OVR - 1], fill=255)
        del draw
        mask = mask_big.resize((diam, diam), Image.LANCZOS)
        resized.putalpha(mask)
        base = Image.fromarray(frame_tb, "RGB").convert("RGBA")
        base.paste(resized, (int(cx - r), int(cy - r)), resized)
        return np.array(base.convert("RGB"))

    def _draw_ring_glow(self, frame_tb: np.ndarray,
                        r_px: float,
                        palette: list,
                        pulse: float = 0.0,
                        pulse_intensity: float = 1.0) -> np.ndarray:
        H, W = frame_tb.shape[:2]
        cx, cy = int(W / 2), int(H / 2)
        r = max(4, int(r_px * (1.0 + 0.10 * pulse * pulse_intensity)))

        pal0 = tuple(int(c * 255) for c in palette[0][:3])
        pal1 = tuple(int(c * 255) for c in palette[1][:3])

        OVR     = 2
        max_ext = 7 * max(1, r // 35)
        pad     = max_ext + 4
        patch_r = r + pad
        ps      = patch_r * 2 * OVR
        pc      = patch_r * OVR
        r2      = r * OVR
        rg2     = int(r * 1.5) * OVR

        patch = Image.new("RGBA", (ps, ps), (0, 0, 0, 0))
        draw  = ImageDraw.Draw(patch)

        draw.ellipse([pc - rg2, pc - rg2, pc + rg2, pc + rg2], fill=(*pal0, 20))
        draw.ellipse([pc - r2,  pc - r2,  pc + r2,  pc + r2],  fill=(*pal0, 12))

        for i in range(7, 0, -1):
            extra2 = i * max(1, r // 35) * OVR
            draw.ellipse(
                [pc - r2 - extra2, pc - r2 - extra2, pc + r2 + extra2, pc + r2 + extra2],
                outline=(*pal0, i * 5), width=extra2 + 1,
            )

        rw2 = max(2, r // 20) * OVR
        draw.ellipse([pc - r2, pc - r2, pc + r2, pc + r2], outline=(*pal1, 200), width=rw2)
        draw.ellipse([pc - r2, pc - r2, pc + r2, pc + r2],
                     outline=(255, 255, 255, 85), width=max(1, rw2 // 2))
        del draw

        patch_1x = patch.resize((patch_r * 2, patch_r * 2), Image.LANCZOS)

        # Composite over the patch's bounding box only: the overlay is fully transparent
        # everywhere else, so converting the whole 1920×1080 frame to RGBA and back was
        # pure overhead. The patch still goes through a transparent RGBA overlay first —
        # that squares its alpha, and the glow's look depends on it.
        x0, y0 = cx - patch_r, cy - patch_r
        side   = patch_r * 2
        bx0, by0 = max(0, x0), max(0, y0)
        bx1, by1 = min(W, x0 + side), min(H, y0 + side)
        base = Image.fromarray(frame_tb, "RGB")
        if bx0 >= bx1 or by0 >= by1:
            return np.array(base)

        overlay = Image.new("RGBA", (bx1 - bx0, by1 - by0), (0, 0, 0, 0))
        overlay.paste(patch_1x, (x0 - bx0, y0 - by0), patch_1x)
        crop = base.crop((bx0, by0, bx1, by1)).convert("RGBA")
        base.paste(Image.alpha_composite(crop, overlay).convert("RGB"), (bx0, by0))
        return np.array(base)

    def draw_selection_outline(self, layer: Layer) -> None:
        """Preview-only: draw a highlight box around the selected layer."""
        self.fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.line_prog["u_offset"].value = (float(layer.x), float(layer.y))
        scale = ((float(layer.image_width), float(layer.image_height))
                 if layer.mode == IMAGE_MODE
                 else (float(layer.scale), float(layer.scale)))
        self.line_prog["u_scale"].value = scale
        self.line_prog["u_color"].value = (0.0, 0.9, 1.0)
        self.line_vao.render(moderngl.LINE_LOOP)
        if layer.mode == IMAGE_MODE:
            handle_scale = (
                10.0 / max(self.output_width, 1),
                10.0 / max(self.output_height, 1))
            for dx, dy in (
                (-layer.image_width, -layer.image_height),
                (0.0, -layer.image_height),
                (layer.image_width, -layer.image_height),
                (-layer.image_width, 0.0),
                (layer.image_width, 0.0),
                (-layer.image_width, layer.image_height),
                (0.0, layer.image_height),
                (layer.image_width, layer.image_height),
            ):
                self.line_prog["u_offset"].value = (
                    float(layer.x + dx), float(layer.y + dy))
                self.line_prog["u_scale"].value = handle_scale
                self.line_vao.render(moderngl.LINE_LOOP)

    # ── Output ───────────────────────────────────────────────────
    def read_frame(self) -> bytes:
        raw = self.fbo.read(components=3)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.output_height, self.output_width, 3)
        return np.ascontiguousarray(pixels[::-1]).tobytes()

    def release(self):
        for st in self._states.values():
            st.release()
        self._states.clear()
        if self._bg_texture:
            self._bg_texture.release()
        if self._center_texture:
            self._center_texture.release()
        for texture in self._text_textures.values():
            texture.release()
        self._text_textures.clear()
        self._text_signatures.clear()
        for texture in self._image_textures.values():
            texture.release()
        self._image_textures.clear()
        self._image_paths.clear()
        for texture in self._particle_textures.values():
            texture.release()
        self._particle_textures.clear()
        if self._bg_video:
            self._bg_video.close()
        self.vao.release()
        self.composite_vao.release()
        self.bg_vao.release()
        self.fxaa_vao.release()
        self.overlay_vao.release()
        self.image_vao.release()
        self.line_vao.release()
        self.vbo.release()
        self.line_vbo.release()
        if self.fbo:
            self.fbo.color_attachments[0].release()
            self.fbo.release()
        if self.scene_fbo:
            self.scene_fbo.color_attachments[0].release()
            self.scene_fbo.release()
        self.prog.release()
        self.composite_prog.release()
        self.bg_prog.release()
        self.fxaa_prog.release()
        self.overlay_prog.release()
        self.image_prog.release()
        self.line_prog.release()
        if self._owns_ctx:
            self.ctx.release()
