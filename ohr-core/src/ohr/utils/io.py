"""Lightweight persistence helpers for OHR artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def save_json(payload: dict, path: str | Path) -> Path:
    """Write a JSON file with UTF-8 encoding."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return target


def load_json(path: str | Path) -> dict:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
