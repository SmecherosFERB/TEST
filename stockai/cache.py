"""Cache pe disc, ca să nu consumăm de două ori aceeași cerere într-o zi."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


class DiskCache:
    def __init__(self, root: str | os.PathLike | None = None) -> None:
        self.root = Path(root or os.getenv("STOCKAI_CACHE", ".cache"))

    def _path(self, key: str, ext: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        return self.root / f"{safe}.{ext}"

    def _fresh(self, path: Path, max_age_hours: float) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600

    def get_json(self, key: str, max_age_hours: float) -> Any | None:
        path = self._path(key, "json")
        if not self._fresh(path, max_age_hours):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set_json(self, key: str, value: Any) -> None:
        path = self._path(key, "json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")

    def get_frame(self, key: str, max_age_hours: float) -> pd.DataFrame | None:
        path = self._path(key, "csv")
        if not self._fresh(path, max_age_hours):
            return None
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except (OSError, ValueError):
            return None

    def set_frame(self, key: str, frame: pd.DataFrame) -> None:
        path = self._path(key, "csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)


class RateLimiter:
    """Păstrează o pauză minimă între cereri către același furnizor."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()
