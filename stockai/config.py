"""Setări centrale: orizont, praguri de decizie și ponderi."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # Câte zile de tranzacționare înainte măsurăm dacă prețul a crescut.
    horizon_days: int = 20
    # Cât istoric descărcăm (acoperă SMA200 + mulți ani pentru statistică și model).
    history_period: str = "15y"

    # Scor compus în [-100, 100]: peste +buy_threshold = BUY, sub -buy_threshold = SELL.
    buy_threshold: float = 25.0

    # Ponderile componentelor din scorul compus (se renormalizează dacă lipsește una).
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "technical": 0.35,
            "fundamental": 0.15,
            "sentiment": 0.10,
            "earnings": 0.15,
            "insiders": 0.10,
            "market": 0.15,
        }
    )

    # Calibrare: zilele istorice cu scor tehnic în ±bucket_width față de azi.
    bucket_width: float = 15.0
    min_samples: int = 60
    # Sub acest avantaj față de rata de bază, istoricul nu ne spune nimic util.
    min_edge: float = 0.03
    # Două componente cu semne opuse și cel puțin această intensitate = conflict.
    conflict_strength: float = 30.0

    # Modelul de probabilitate antrenat cu `python -m stockai --train`.
    model_path: str = field(default_factory=lambda: os.getenv("STOCKAI_MODEL", ".cache/model.pkl"))
    # Al doilea model: șansele ca acțiunea să bată S&P 500 (python -m stockai --train --target beat).
    beat_model_path: str = field(default_factory=lambda: os.getenv("STOCKAI_MODEL_BEAT", ".cache/model_beat.pkl"))

    claude_model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-5"))
    claude_effort: str = field(default_factory=lambda: os.getenv("CLAUDE_EFFORT", "high"))
