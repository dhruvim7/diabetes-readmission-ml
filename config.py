"""Configuration loading and path management.

The entire pipeline is driven by a single YAML file. No analysis constant is
defined anywhere else. This module resolves that file into an object with
attribute access and guarantees the output directory tree exists.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "load_config"]


class Config:
    """Dot-accessible, immutable-by-convention view over the YAML config."""

    def __init__(self, data: dict[str, Any], root: Path) -> None:
        self._data = data
        self.root = root

    # -- access ------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"No configuration key {key!r}") from exc
        if isinstance(value, dict):
            return Config(value, self.root)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- paths -------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a configured directory to an absolute path, creating it."""
        p = self.root / self._data["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def ensure_tree(self) -> None:
        for key in self._data["paths"]:
            self.path(key)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load the analysis configuration and create the output directory tree."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = Config(data, root=path.parent)
    cfg.ensure_tree()
    return cfg
