"""Portable project-template serialization for GUI and CLI workflows."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, fields
from pathlib import Path

from core.layer import BGLayer, Layer, LayerManager


TEMPLATE_VERSION = 1
_LAYER_FIELDS = {field.name for field in fields(Layer)} - {"id"}
_BG_FIELDS = {field.name for field in fields(BGLayer)} - {"id"}


def _portable_path(value: str | None, base: Path) -> str | None:
    if not value:
        return None
    try:
        return os.path.relpath(Path(value).resolve(), base.resolve())
    except ValueError:  # Different Windows drive.
        return str(Path(value).resolve())


def _resolved_path(value: str | None, base: Path) -> str | None:
    if not value:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def template_data(layer_manager: LayerManager, center_image_path: str | None,
                  settings: dict, template_dir: Path) -> dict:
    background = asdict(layer_manager.bg)
    background.pop("id", None)
    background["image_path"] = _portable_path(background.get("image_path"), template_dir)
    layers = []
    for layer in layer_manager.layers:
        item = asdict(layer)
        item.pop("id", None)
        item["image_path"] = _portable_path(item.get("image_path"), template_dir)
        layers.append(item)
    selected_index = layer_manager.index_of(layer_manager.selected_id)
    return {
        "version": TEMPLATE_VERSION,
        "background": background,
        "center_image_path": _portable_path(center_image_path, template_dir),
        "layers": layers,
        "selected_layer": selected_index,
        "settings": settings,
    }


def save_template(path: str, layer_manager: LayerManager,
                  center_image_path: str | None, settings: dict) -> None:
    target = Path(path)
    data = template_data(layer_manager, center_image_path, settings, target.parent)
    suffix = target.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        # JSON is valid YAML 1.2 and keeps this format dependency-free.
        content = json.dumps(data, indent=2, ensure_ascii=False)
    elif suffix == ".toml":
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        content = "version = 1\npayload = " + json.dumps(payload, ensure_ascii=False) + "\n"
    else:
        raise ValueError("Template must use .json, .yaml, .yml, or .toml")
    target.write_text(content, encoding="utf-8")


def _read_data(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".toml":
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        return json.loads(document["payload"])
    raise ValueError("Template must use .json, .yaml, .yml, or .toml")


def load_template(path: str) -> tuple[LayerManager, str | None, dict]:
    source = Path(path)
    data = _read_data(source)
    if data.get("version") != TEMPLATE_VERSION:
        raise ValueError(f"Unsupported template version: {data.get('version')}")
    base = source.parent
    lm = LayerManager()
    bg_data = {key: value for key, value in data.get("background", {}).items()
               if key in _BG_FIELDS}
    bg_data["image_path"] = _resolved_path(bg_data.get("image_path"), base)
    for key in ("text_color", "text_stroke_color", "text_shadow_color"):
        if key in bg_data:
            bg_data[key] = tuple(bg_data[key])
    lm.bg = BGLayer(**bg_data)
    for raw in data.get("layers", []):
        item = {key: value for key, value in raw.items() if key in _LAYER_FIELDS}
        item["image_path"] = _resolved_path(item.get("image_path"), base)
        # JSON turns tuples into lists; renderer accepts both, but restore model types.
        for key in ("text_color", "text_stroke_color", "text_shadow_color"):
            if key in item:
                item[key] = tuple(item[key])
        lm.layers.append(Layer(**item))
    selected = int(data.get("selected_layer", -1))
    lm.selected_id = lm.layers[selected].id if 0 <= selected < len(lm.layers) else 0
    center = _resolved_path(data.get("center_image_path"), base)
    return lm, center, dict(data.get("settings", {}))
