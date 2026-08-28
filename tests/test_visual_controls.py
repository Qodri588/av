from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config.defaults import (
    BAR_WIDTH, BAR_SHAPES, CIRCLE_RADIUS_RATIO, RESOLUTIONS, VISUAL_PRESETS,
    VIZ_TYPES_VISIBLE,
)
from core.layer import (
    Layer, LayerManager, TEXT_MODE, IMAGE_MODE, PARTICLE_MODE, BLEND_NORMAL,
)
from ui.preview import (
    _physical_size, _image_rect_pixels, _image_resize_handle, _apply_image_resize,
)
from render.renderer import Renderer, _render_scale
from render.renderer import _load_rgba_image


def test_layer_visual_geometry_defaults_are_per_layer():
    first = Layer()
    second = Layer()

    assert first.circle_radius == CIRCLE_RADIUS_RATIO
    assert first.bar_width == BAR_WIDTH

    first.circle_radius = 0.28
    first.bar_width = 0.35

    assert second.circle_radius == CIRCLE_RADIUS_RATIO
    assert second.bar_width == BAR_WIDTH


def test_preview_uses_physical_pixels_on_high_dpi_displays():
    assert _physical_size(800, 450, 1.0) == (800, 450)
    assert _physical_size(800, 450, 1.5) == (1200, 675)
    assert _physical_size(800, 450, 2.0) == (1600, 900)


def test_renderer_supersamples_without_changing_output_dimensions():
    renderer = Renderer(320, 180)
    try:
        assert (renderer.width, renderer.height) == (640, 360)
        assert renderer.fbo.size == (320, 180)
        assert renderer.scene_fbo.size == (640, 360)
    finally:
        renderer.release()


def test_renderer_caps_supersampling_for_4k_output():
    assert _render_scale(1920, 1080, 2) == 2
    assert _render_scale(3840, 2160, 2) == 1


def test_removed_visual_types_are_not_selectable():
    removed = {
        "Halo Sine", "Flat Sine", "Tunnel Arcade",
        "Halo Classic", "Halo Fine", "Halo Bold",
        "Linear Fine", "Linear Wide",
    }
    assert removed.isdisjoint(VIZ_TYPES_VISIBLE)


def test_remaining_visual_types_have_multiple_real_variations():
    modes = [preset["mode"] for preset in VISUAL_PRESETS.values()]
    for mode in (0, 1, 2, 4):
        assert modes.count(mode) >= 3

    manager = LayerManager()
    layer = manager.add_layer()
    default = VISUAL_PRESETS[layer.preset_name]
    assert layer.mode == default["mode"]
    assert layer.num_bars == default["num_bars"]


def test_studio_spectrum_presets_match_reference_proportions():
    spectrum = VISUAL_PRESETS["Studio Spectrum"]
    mirror = VISUAL_PRESETS["Studio Mirror"]

    assert spectrum == {
        "mode": 2, "num_bars": 72, "max_bar_height": 0.42, "bar_width": 0.52,
        "bar_shape": 3,
    }
    assert mirror == {
        "mode": 1, "num_bars": 72, "max_bar_height": 0.58, "bar_width": 0.52,
        "bar_shape": 3,
    }
    assert "Studio Spectrum" in VIZ_TYPES_VISIBLE
    assert "Studio Mirror" in VIZ_TYPES_VISIBLE


def test_led_presets_and_bar_shapes_are_available():
    assert BAR_SHAPES == (
        "Square", "Sharp", "Rounded", "Pill", "Classic LED", "Dot LED",
    )
    assert VISUAL_PRESETS["Classic LED"]["bar_shape"] == 4
    assert VISUAL_PRESETS["Dot LED"]["bar_shape"] == 5
    assert "Classic LED" in VIZ_TYPES_VISIBLE
    assert "Dot LED" in VIZ_TYPES_VISIBLE


def test_flat_spectrum_shader_uses_rounded_glowing_bars():
    shader = (Path(__file__).parents[1] / "render" / "shaders" / "radial.frag").read_text()

    assert "rounded_box_sdf" in shader
    assert "studio_bar" in shader
    assert "studio_bar_sdf" in shader
    assert "u_bar_shape" in shader
    assert "row_count" in shader
    assert "lit_height" in shader
    assert "glow_strength = (u_bar_shape >= 4) ? 0.10 : 0.16" in shader
    assert "rim_strength = (u_bar_shape >= 4) ? 0.0 : 0.12" in shader
    assert "float baseline = exp(" in shader


def test_all_export_resolutions_are_16_by_9():
    for width, height in RESOLUTIONS.values():
        assert width * 9 == height * 16


def test_multiple_text_layers_have_independent_settings():
    manager = LayerManager()
    first = manager.add_text_layer()
    second = manager.add_text_layer()

    first.text = "Title"
    first.text_size = 120
    second.text = "Subtitle"
    second.text_size = 48

    assert first.mode == second.mode == TEXT_MODE
    assert first.id != second.id
    assert (first.text, first.text_size) == ("Title", 120)
    assert (second.text, second.text_size) == ("Subtitle", 48)


def test_particle_layer_has_reference_parameters_and_renders_pixels():
    manager = LayerManager()
    layer = manager.add_particle_layer()
    assert layer.mode == PARTICLE_MODE
    assert layer.particle_direction == "random"
    assert layer.particle_origin == "full"
    assert layer.particle_color_mode == "palette"
    assert layer.particle_reactive_source == "none"
    assert layer.particle_beat_reactive is False
    assert layer.particle_glossy == 0
    assert layer.particle_size == 3.0
    assert layer.opacity == 0.9
    assert layer.particle_count == 50
    assert layer.particle_random_size is False
    assert len(layer.particle_colors) == 3

    renderer = Renderer(320, 180, supersampling=1)
    try:
        renderer.render_composition(
            manager, np.linspace(0.1, 0.9, 128, dtype=np.float32),
            pulse=0.5, time=1.25)
        frame = np.frombuffer(renderer.read_frame(), dtype=np.uint8).reshape(180, 320, 3)
        assert frame.max() > 30
        assert np.count_nonzero(frame) > 100
    finally:
        renderer.release()


def test_image_overlay_has_independent_free_size_and_beat_controls():
    manager = LayerManager()
    first = manager.add_image_layer("first.png")
    second = manager.add_image_layer("second.svg")

    first.image_width = 0.72
    first.image_height = 0.24
    first.image_beat_intensity = 0.65

    assert first.mode == second.mode == IMAGE_MODE
    assert first.blend_mode == second.blend_mode == BLEND_NORMAL
    assert (first.image_width, first.image_height) == (0.72, 0.24)
    assert (second.image_width, second.image_height) == (0.35, 0.35)
    assert first.image_beat_reactive is True
    assert first.image_beat_intensity == 0.65


def test_image_overlays_do_not_consume_visual_layer_limit():
    manager = LayerManager()
    manager.add_image_layer("overlay.svg")
    assert all(manager.add_layer() is not None for _ in range(6))
    assert manager.add_layer() is None


def test_svg_overlay_loader_preserves_transparency(tmp_path):
    asset = tmp_path / "overlay.svg"
    asset.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
        '<rect x="10" width="20" height="20" fill="#ff0000"/></svg>',
        encoding="utf-8")

    image = _load_rgba_image(str(asset))

    assert image.size == (40, 20)
    assert image.getpixel((0, 0))[3] == 0
    assert image.getpixel((20, 10))[:3] == (255, 0, 0)


def test_renderer_composites_image_overlay_with_alpha(tmp_path):
    asset = tmp_path / "overlay.png"
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for x in range(4, 12):
        for y in range(4, 12):
            image.putpixel((x, y), (255, 0, 0, 255))
    image.save(asset)
    manager = LayerManager()
    layer = manager.add_image_layer(str(asset))
    layer.image_width = 0.5
    layer.image_height = 0.5

    renderer = Renderer(160, 90, supersampling=1)
    try:
        renderer.render_composition(
            manager, np.zeros(256, dtype=np.float32), pulse=1.0)
        frame = np.frombuffer(renderer.read_frame(), dtype=np.uint8).reshape(90, 160, 3)
        assert frame[:, :, 0].max() > 200
        assert frame[0, 0].max() == 0
    finally:
        renderer.release()


def test_image_edge_resize_keeps_opposite_edge_anchored():
    layer = Layer(mode=IMAGE_MODE, image_width=0.5, image_height=0.4)
    width, height = 800, 450
    before = _image_rect_pixels(layer, width, height)
    origin = (400.0, 225.0, layer.x, layer.y,
              layer.image_width, layer.image_height)

    assert _image_resize_handle(layer, before[0], 225, width, height) == "w"
    _apply_image_resize(layer, "w", origin, 80, 0, width, height)
    after = _image_rect_pixels(layer, width, height)

    assert after[0] == pytest.approx(before[0] + 80)
    assert after[2] == pytest.approx(before[2])
    assert layer.image_width == pytest.approx(0.4)


def test_image_corner_resize_changes_both_axes_independently():
    layer = Layer(mode=IMAGE_MODE, image_width=0.5, image_height=0.4)
    width, height = 800, 450
    before = _image_rect_pixels(layer, width, height)
    origin = (before[2], before[3], layer.x, layer.y,
              layer.image_width, layer.image_height)

    assert _image_resize_handle(layer, before[2], before[3], width, height) == "se"
    _apply_image_resize(layer, "se", origin, 120, -30, width, height)
    after = _image_rect_pixels(layer, width, height)

    assert after[0] == pytest.approx(before[0])
    assert after[1] == pytest.approx(before[1])
    assert after[2] == pytest.approx(before[2] + 120)
    assert after[3] == pytest.approx(before[3] - 30)
