"""Path helpers that stay valid when the OHR core is installed from PyPI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def get_default_base_dir() -> Path:
    """Return the current working directory as the default base directory."""
    return Path.cwd()


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to the caller workspace or explicit base directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = get_default_base_dir() if base_dir is None else Path(base_dir)
    return (base / candidate).resolve()


def ensure_dir(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Create a directory if it does not exist and return it."""
    directory = resolve_path(path, base_dir=base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for artifact directories."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
