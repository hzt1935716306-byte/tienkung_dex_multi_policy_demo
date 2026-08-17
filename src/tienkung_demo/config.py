from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _resolve_path(value: str, config_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")

    config["_path"] = str(config_path)
    config_dir = config_path.parent
    config["model"] = _resolve_path(config["model"], config_dir)
    config["walk_policy"] = _resolve_path(config["walk_policy"], config_dir)

    motions = config.get("motions", {})
    if not isinstance(motions, dict):
        raise ValueError("Config field 'motions' must be an object.")
    for key, motion in motions.items():
        if len(key) != 1 or not isinstance(motion, dict):
            raise ValueError(f"Invalid motion entry: {key!r}")
        motion["path"] = _resolve_path(motion["path"], config_dir)

    for field in ("model", "walk_policy"):
        if not Path(config[field]).is_file():
            raise FileNotFoundError(config[field])
    for motion in motions.values():
        if not Path(motion["path"]).is_file():
            raise FileNotFoundError(motion["path"])
    return config
