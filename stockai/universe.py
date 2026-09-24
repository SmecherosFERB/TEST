"""Lista de acțiuni importante (aceeași ca în pagina web): simbol, nume, sector, ordinea importanței."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UNIVERSE_PATH = Path(__file__).with_name("universe.json")


def load_universe() -> list[dict[str, Any]]:
    rows = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    return sorted(rows, key=lambda r: r["rank"])
