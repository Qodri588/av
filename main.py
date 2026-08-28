import argparse
import sys
from pathlib import Path


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an audio visualizer from a saved GUI template.")
    parser.add_argument("-cli", "--cli", dest="template", required=True,
                        help="Template file (.json, .yaml, .yml, or .toml)")
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("background", nargs="?",
                        help="Optional background image/video override")
    parser.add_argument("output", nargs="?", help="Output MP4 path")
    parser.add_argument("--bg", dest="background_option",
                        help="Background image/video override")
    parser.add_argument("-o", "--output", dest="output_option",
                        help="Output MP4 path")
    parser.add_argument("--center", help="Center image override")
    parser.add_argument("--resolution", choices=("720p", "1080p", "4K"),
                        help="Override the resolution saved in the template")
    parser.add_argument("--fps", type=int, choices=(30, 60),
                        help="Override the FPS saved in the template")
    return parser


def cli_main(argv: list[str]) -> int:
    args = _cli_parser().parse_args(argv)
    from config.defaults import CQT_BINS_PER_OCTAVE
    from core.audio import AudioFile
    from core.template import load_template
    from export.ffmpeg_export import FFmpegExporter

    template_path = Path(args.template).resolve()
    audio_path = Path(args.audio).resolve()
    background = args.background_option or args.background
    output = args.output_option or args.output
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if background:
        background = str(Path(background).resolve())
        if not Path(background).is_file():
            raise FileNotFoundError(f"Background not found: {background}")
    center = args.center
    if center:
        center = str(Path(center).resolve())
        if not Path(center).is_file():
            raise FileNotFoundError(f"Center image not found: {center}")
    if not output:
        output = str(Path.cwd() / f"{audio_path.stem}.mp4")
    output_path = Path(output).resolve()
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    lm, center_image, settings = load_template(str(template_path))
    if background:
        lm.bg.image_path = background
    bg_path = background or lm.bg.image_path
    center_image = center or center_image
    audio = AudioFile(str(audio_path))

    def progress(current: int, total: int) -> None:
        percent = round(current * 100 / max(total, 1))
        print(f"\rRendering: {percent:3d}%", end="", flush=True)

    exporter = FFmpegExporter(
        audio=audio, layer_manager=lm,
        resolution=args.resolution or str(settings.get("resolution", "1080p")),
        fps=args.fps or int(settings.get("fps", 60)),
        smoothing_decay=float(settings.get("smoothing_decay", .85)),
        bass_split=float(settings.get("bass_split", .60)),
        bass_split_hz=float(settings.get("bass_split_hz", 300)),
        use_cqt=bool(settings.get("use_cqt", False)),
        bins_per_octave=int(settings.get("bins_per_octave", CQT_BINS_PER_OCTAVE)),
        bg_image_path=bg_path, center_image_path=center_image,
        output_path=str(output_path), progress_cb=progress,
    )
    result = exporter.export()
    print(f"\nExport complete: {result}")
    return 0


def gui_main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ui.mainwindow import MainWindow

    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Visualizer")
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> int:
    if "-cli" in sys.argv or "--cli" in sys.argv:
        return cli_main(sys.argv[1:])
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
