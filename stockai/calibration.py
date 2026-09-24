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
    total: int = 0  # câte zile au intrat în rata de bază

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
        total=len(frame),
    )


# ---------- estimări oneste ----------
# Ferestrele de `horizon` zile se suprapun: `horizon` zile la rând valorează cam o singură observație independentă.
# Procentul unei singure acțiuni se bazează deci pe puține cazuri reale. Îl tragem spre o estimare mai largă
# (modelul antrenat pe toate acțiunile, dacă există, altfel rata de bază) și arătăm intervalul de încredere de 90%.
Z90 = 1.645
SHRINK = 25  # câte cazuri independente „valorează” estimarea largă
SHRINK_BASE = 100


def wilson_interval(p: float, n: float, z: float = Z90) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z2 = z * z
    d = 1 + z2 / n
    center = (p + z2 / (2 * n)) / d
    margin = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / d
    return max(0.0, center - margin), min(1.0, center + margin)


def honest_estimate(
    odds: HistoricalOdds | None,
    prior_prob: float | None = None,
    prior_base: float | None = None,
    total_samples: int | None = None,
) -> dict | None:
    """Estimarea combinată, cu interval. `prior_*` vin de obicei din modelul antrenat pe toată lista."""
    if odds is None:
        return None
    h = max(1, odds.horizon)
    n_stock = odds.samples / h
    up_stock = odds.probability_up * n_stock
    base = odds.base_rate
    if prior_base is not None:
        n_base = (total_samples or 2520) / h
        base = (odds.base_rate * n_base + SHRINK_BASE * prior_base) / (n_base + SHRINK_BASE)
    prior = prior_prob if prior_prob is not None else base
    p = (up_stock + SHRINK * prior) / (n_stock + SHRINK)
    lo, hi = wilson_interval(p, n_stock + SHRINK)
    return {
        "p": p,
        "lo": lo,
        "hi": hi,
        "base": base,
        "edge": p - base,
        "sure": "up" if lo > base else "down" if hi < base else None,
        "stock_p": odds.probability_up,
        "stock_independent_cases": n_stock,
        "prior": prior,
        "source": "history",
    }


def model_estimate(model_out: dict, odds: HistoricalOdds | None) -> dict | None:
    """Estimarea modelului recalibrat pe anii nevăzuți (are deja intervalul lui); istoricul acțiunii rămâne informativ."""
    if not model_out or model_out.get("lo") is None:
        return None
    p, lo, hi, base = model_out["prob"], model_out["lo"], model_out["hi"], model_out["base_rate"]
    return {
        "p": p,
        "lo": lo,
        "hi": hi,
        "base": base,
        "edge": p - base,
        "sure": "up" if lo > base else "down" if hi < base else None,
        "stock_p": odds.probability_up if odds else None,
        "stock_independent_cases": odds.samples / max(1, odds.horizon) if odds else 0.0,
        "prior": p,
        "source": "model",
    }
