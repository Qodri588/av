from pathlib import Path

import pytest

from core.layer import IMAGE_MODE, TEXT_MODE, LayerManager
from core.template import load_template, save_template
from main import _cli_parser


@pytest.mark.parametrize("extension", ["json", "yaml", "yml", "toml"])
def test_template_round_trip_all_supported_formats(tmp_path, extension):
    assets = tmp_path / "assets"
    assets.mkdir()
    bg = assets / "background.mp4"
    overlay = assets / "logo.svg"
    center = assets / "center.png"
    for asset in (bg, overlay, center):
        asset.touch()

    manager = LayerManager()
    visual = manager.add_layer()
    visual.x = 0.25
    visual.opacity = 0.7
    visual.bar_shape = 5
    text = manager.add_text_layer()
    text.text = "Saved title"
    image = manager.add_image_layer(str(overlay))
    image.image_width = 0.63
    image.image_height = 0.27
    image.image_beat_intensity = 0.8
    particles = manager.add_particle_layer()
    particles.particle_direction = "center-out"
    particles.particle_count = 80
    particles.particle_color_mode = "rainbow"
    particles.particle_glossy = 0.8
    manager.bg.image_path = str(bg)
    manager.bg.crop_zoom = 1.4
    manager.selected_id = image.id
    settings = {"resolution": "720p", "fps": 30, "use_cqt": True,
                "smoothing_decay": .9, "bass_split": .55,
                "bass_split_hz": 400, "bins_per_octave": 24}
    target = tmp_path / f"template.{extension}"

    save_template(str(target), manager, str(center), settings)
    loaded, loaded_center, loaded_settings = load_template(str(target))

    assert loaded.bg.image_path == str(bg.resolve())
    assert loaded.bg.crop_zoom == 1.4
    assert [layer.mode for layer in loaded.layers] == [
        visual.mode, TEXT_MODE, IMAGE_MODE, particles.mode,
    ]
    assert loaded.layers[0].bar_shape == 5
    assert loaded.layers[1].text == "Saved title"
    assert loaded.layers[2].image_width == 0.63
    assert loaded.layers[2].image_height == 0.27
    assert loaded.layers[3].particle_direction == "center-out"
    assert loaded.layers[3].particle_count == 80
    assert loaded.layers[3].particle_color_mode == "rainbow"
    assert loaded.layers[3].particle_glossy == 0.8
    assert loaded.selected().mode == IMAGE_MODE
    assert loaded_center == str(center.resolve())
    assert loaded_settings == settings


def test_cli_accepts_positional_and_flag_forms():
    positional = _cli_parser().parse_args(
        ["-cli", "template.json", "song.wav", "bg.mp4", "result.mp4"])
    flagged = _cli_parser().parse_args(
        ["--cli", "template.toml", "song.flac", "--bg", "bg.png", "-o", "out.mp4"])

    assert (positional.template, positional.audio, positional.background,
            positional.output) == ("template.json", "song.wav", "bg.mp4", "result.mp4")
    assert flagged.background_option == "bg.png"
    assert flagged.output_option == "out.mp4"


def test_cli_accepts_colab_overrides():
    args = _cli_parser().parse_args([
        "--cli", "template.json", "song.wav", "--bg", "background.png",
        "--center", "logo.png", "--resolution", "720p", "--fps", "30",
        "--supersampling", "1", "--duration", "10",
    ])

    assert args.center == "logo.png"
    assert args.resolution == "720p"
    assert args.fps == 30
    assert args.supersampling == 1
    assert args.duration == 10


def test_load_template_accepts_windows_path_separators(tmp_path, monkeypatch):
    import core.template as template_module

    template = tmp_path / "template.json"
    template.write_text(
        '{"version":1,"background":{"image_path":"assets\\\\bg.png"},'
        '"center_image_path":null,"layers":[],"selected_layer":-1,"settings":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(template_module.os, "sep", "/")

    loaded, _, _ = load_template(str(template))

    assert loaded.bg.image_path == str((tmp_path / "assets" / "bg.png").resolve())
