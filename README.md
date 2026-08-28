# Audio Visualizer

Python desktop audio visualizer with GPU-accelerated rendering (ModernGL / GLSL) and FFmpeg video export.

---

## Install

### ⭐ Easiest — packaged app (single file)

A self-contained build with Python, Qt, ModernGL and the shaders all bundled. Nothing to install except FFmpeg (see note).

| OS | File | Run |
|----|------|-----|
| **Linux** | `AudioVisualizer-x86_64.AppImage` | `chmod +x AudioVisualizer-x86_64.AppImage` then double-click or `./AudioVisualizer-x86_64.AppImage` |
| **Windows** | `AudioVisualizer.exe` | double-click (SmartScreen: *More info → Run anyway*) |
| **macOS** | *(build from source for now)* | — |

Grab it from the **[Releases](https://github.com/BouleMagique/audio-visualizer/releases)** page, or build it yourself (below).

> **⚠️ FFmpeg required** for video export (not bundled): `sudo pacman -S ffmpeg` / `sudo apt install ffmpeg` / `winget install Gyan.FFmpeg`.
> **GPU:** rendering uses the system's OpenGL drivers (OpenGL 3.3+), which is normal — they are not bundled.

### 🔨 Build the package

```bash
bash build-linux.sh      # -> dist/AudioVisualizer-x86_64.AppImage
```

Needs Python 3.10+. The script creates an isolated venv, installs the deps + PyInstaller, bundles the GLSL shaders, and wraps everything with `appimagetool` (downloaded automatically on first run). Build happens in a local cache dir (`~/.cache/audio-visualizer-build`) to avoid PyInstaller issues on network mounts.

*Windows (`.exe`) and macOS (`.app`) build scripts follow the same PyInstaller approach, per-OS (PyInstaller does not cross-compile). macOS needs Python 3.10+ — the system Python 3.9 is too old for PySide6.*

### 🐍 From source (developer mode)

```bash
git clone https://github.com/BouleMagique/audio-visualizer
cd audio-visualizer
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**System requirements:** Python 3.10+, FFmpeg in PATH, GPU with OpenGL 3.3+.

---

## Run

- **Packaged app:** just launch the AppImage.
- **From source:** `python main.py`

### Templates and CLI export

Use **Template > Save** in the GUI to save the complete composition as JSON,
YAML, or TOML. Use **Template > Load** to restore it. Asset paths are stored
relative to the template when possible.

Render headlessly using positional arguments:

```bash
python main.py -cli template.json song.wav background.mp4 output.mp4
```

Or use explicit background/output flags:

```bash
python main.py --cli template.toml song.wav --bg background.png -o output.mp4
```

Headless Linux/Google Colab rendering uses EGL:

```bash
MGL_BACKEND=egl python main.py --cli template.json song.wav \
  --bg background.png --center logo.png --resolution 1080p --fps 30 -o output.mp4
```

For an interactive upload-and-render workflow, open `colab_render.ipynb` in
Google Colab. It clones this repository, then lets you upload audio, template,
background, and center image. It can also use `coba/temp.json` directly.
The notebook installs the render-only `requirements-colab.txt`, whose NumPy pin
is compatible with the Numba version preinstalled by Colab.

The notebook uses NVIDIA EGL when a GPU runtime is available and automatically
falls back to Mesa CPU rendering otherwise. Its testing defaults are 720p,
30 FPS, 1x supersampling, and the first 10 seconds of audio. Set
`TEST_SECONDS = None` in the notebook to render the complete audio.

The background is optional. Without an override, the image or video background
saved in the template is used.

---

## Visualization modes

| Mode | Description |
|------|-------------|
| Radial | Circular bars radiating outward |
| Mirror | Symmetric vertical bars |
| Linear | Bottom-up bars |
| Oscilloscope | Lissajous figure |
| Halo | Smooth waveform ring |
| Halo Bass | Polar bass waterfall (12 rings) |
| Halo Sine | Catmull-Rom spline, PIL-rendered |
| Halo Bass 2 | EMA-smoothed neon waterfall |
| Tunnel Arcade | SDF polygon tunnel, kick warp, chromatic aberration |
| Flat Sine | Horizontal mirrored waveform |
| Nuclear Shockwave | Concentric kick-triggered shockwaves |
| Void Pull | Spiral with gravity-pull on kick |

## Features

- Multi-layer compositing (locked background + up to 6 independent visual layers, per-layer FBO, blend modes, palettes)
- 5 built-in palettes + 2 custom palette modes (amplitude / frequency mapping)
- Background image with bass-reactive zoom pulse
- Center image overlay with glow ring and pulse
- White flash on kick / bass hits, 4 kick-detection modes
- Export to MP4 via FFmpeg — 720p / 1080p / 4K, 30 or 60 fps, H.264 + AAC 320k

## Stack

| Component | Library |
|-----------|---------|
| GPU render | ModernGL 5.x + GLSL 3.30 |
| Audio decode | soundfile |
| Audio playback | sounddevice |
| FFT / DSP | numpy + scipy |
| UI | PySide6 |
| PIL overlay | Pillow |
| Export | FFmpeg (subprocess) |
