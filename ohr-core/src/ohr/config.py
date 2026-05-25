"""Simple YAML and JSON configuration helpers for the standalone OHR core."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()

    with config_path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle) or {}
        elif suffix == ".json":
            data = json.load(handle)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path}")

    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping-like configuration object in {config_path}")

    data["_config_path"] = str(config_path.resolve())
    return data


def load_packaged_config(name: str = "default_ohr.yaml") -> dict[str, Any]:
    """Load one of the default YAML configurations bundled in the package."""
    resource = files("ohr.resources.configs").joinpath(*Path(name).parts)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged configuration not found: {name}")

    data = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping-like configuration object in {name}")

    data["_config_path"] = f"package://ohr.resources.configs/{name}"
    return data
