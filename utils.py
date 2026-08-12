"""Cross-cutting infrastructure: logging, provenance, artifact IO, gates.

Two ideas carry most of the weight here:

``RunManifest``
    Every stage appends row counts, timings, seeds and file hashes. The
    manifest is the audit trail that lets any number in the manuscript be
    traced back to the run that produced it.

``gate``
    A stage-boundary assertion. The dominant failure mode in a pipeline like
    this is not incorrect code but a wrong intermediate value propagating
    silently downstream. Gates make that failure loud and immediate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

__all__ = [
    "GateError",
    "RunManifest",
    "gate",
    "get_logger",
    "read_table",
    "seed_everything",
    "setup_logging",
    "sha256",
    "stage_timer",
    "write_json",
    "write_table",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Configure root logging to stream to stdout and a timestamped file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f"run_{stamp}.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level, format=_LOG_FORMAT, handlers=handlers, force=True
    )
    # Third-party libraries are noisy at INFO.
    for noisy in ("optuna", "matplotlib", "shap", "numba", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set every seed the pipeline depends on."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class RunManifest:
    """Append-only provenance record for a pipeline run."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._data = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                },
                "packages": _package_versions(),
                "stages": {},
                "row_counts": {},
                "gates": [],
                "files": {},
            }
            self._flush()

    # -- recording ---------------------------------------------------------
    def record_stage(self, stage: str, **payload: Any) -> None:
        entry = self._data["stages"].setdefault(stage, {})
        entry.update(payload)
        entry["completed_utc"] = datetime.now(timezone.utc).isoformat()
        self._flush()

    def record_rows(self, label: str, n: int) -> None:
        self._data["row_counts"][label] = int(n)
        self._flush()

    def record_gate(self, name: str, passed: bool, detail: str) -> None:
        self._data["gates"].append(
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
                "utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._flush()

    def record_file(self, label: str, path: Path) -> None:
        path = Path(path)
        if path.exists():
            self._data["files"][label] = {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            self._flush()

    @property
    def row_counts(self) -> dict[str, int]:
        return dict(self._data["row_counts"])

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


def _package_versions() -> dict[str, str]:
    names = [
        "numpy", "pandas", "scikit-learn", "scipy", "xgboost",
        "lightgbm", "shap", "optuna", "imbalanced-learn", "statsmodels",
    ]
    out: dict[str, str] = {}
    try:
        from importlib.metadata import version

        for n in names:
            try:
                out[n] = version(n)
            except Exception:  # pragma: no cover
                out[n] = "not installed"
    except Exception:  # pragma: no cover
        pass
    return out


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

class GateError(RuntimeError):
    """Raised when a stage-boundary validation check fails."""


def gate(
    name: str,
    condition: bool,
    detail: str,
    manifest: RunManifest | None = None,
    fatal: bool = True,
) -> bool:
    """Assert a stage-boundary condition.

    Parameters
    ----------
    name
        Short identifier recorded in the manifest.
    condition
        The check. ``True`` passes.
    detail
        Human-readable description including the observed value.
    fatal
        When ``True`` a failure raises :class:`GateError` and halts the run.
        When ``False`` it logs a warning and continues.
    """
    log = get_logger("gate")
    if manifest is not None:
        manifest.record_gate(name, condition, detail)
    if condition:
        log.info("GATE PASS  [%s] %s", name, detail)
        return True
    message = f"GATE FAIL  [{name}] {detail}"
    if fatal:
        log.error(message)
        raise GateError(message)
    log.warning(message)
    return False


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def stage_timer(stage: str, manifest: RunManifest | None = None) -> Iterator[None]:
    log = get_logger("pipeline")
    log.info("=" * 78)
    log.info("STAGE START  %s", stage)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("STAGE END    %s  (%.1fs)", stage, elapsed)
        if manifest is not None:
            manifest.record_stage(stage, elapsed_seconds=round(elapsed, 2))


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------

def write_json(obj: Any, path: Path) -> Path:
    """Serialise metrics to JSON, coercing numpy scalars."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    get_logger("io").info("wrote %s", path.name)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def write_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Persist a table as CSV. Publication tables are always written to disk
    so that figures and manuscript text read from the same source."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    get_logger("io").info("wrote %s  (%d rows x %d cols)", path.name, *df.shape)
    return path


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
