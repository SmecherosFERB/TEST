"""Procentul istoric: cât de des a urcat prețul când scorul tehnic arăta ca azi."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HistoricalOdds:
    probability_up: float  # frecvența creșterii după `horizon` zile, pentru scoruri similare
    base_rate: float  # frecvența creșterii în toate zilele (referința)
    avg_return: float  # randamentul mediu în zilele similare
    samples: int  # câte zile similare au existat (atenție: ferestrele se suprapun)
    horizon: int

    @property
    def edge(self) -> float:
        return self.probability_up - self.base_rate


def historical_odds(
    close: pd.Series,
    score: pd.Series,
    current_score: float,
    horizon: int,
    bucket_width: float,
) -> HistoricalOdds | None:
    forward = close.shift(-horizon) / close - 1
    frame = pd.DataFrame({"score": score, "forward": forward}).dropna()
    if frame.empty:
        return None
    similar = frame[(frame["score"] - current_score).abs() <= bucket_width]
    if similar.empty:
        return None
    return HistoricalOdds(
        probability_up=float((similar["forward"] > 0).mean()),
        base_rate=float((frame["forward"] > 0).mean()),
        avg_return=float(similar["forward"].mean()),
        samples=len(similar),
        horizon=horizon,
    )
