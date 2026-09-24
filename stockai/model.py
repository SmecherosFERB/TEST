"""Model de probabilitate: P(prețul e mai mare peste `horizon` zile), învățat pe toată lista de acțiuni.

Regresie logistică pe câteva semnale cu dovezi publicate. Înainte de a o folosi, o verificăm „walk-forward”:
pentru fiecare an, modelul e antrenat doar pe anii anteriori și testat pe anul respectiv, apoi comparăm
cu cea mai simplă predicție posibilă (rata de bază: cât de des au urcat acțiunile în trecut).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .indicators import compute_all
from .scoring import technical_score
from .signals import earnings_surprise_series, high_52w_gap, momentum_12_1, realized_vol, regime_series

FEATURES = [
    "tech",
    "mom_12_1",
    "ret_1m",
    "high_52w",
    "vol_20d",
    "dist_sma200",
    "rsi",
    "mkt_regime",
    "mkt_ret_1m",
    "earn_surprise",
]
FEATURE_NAMES = {
    "tech": "scor tehnic",
    "mom_12_1": "momentum 12 luni (fără ultima)",
    "ret_1m": "randament ultima lună",
    "high_52w": "distanța față de maximul pe 52 săpt.",
    "vol_20d": "volatilitate 20 zile",
    "dist_sma200": "distanța față de media pe 200 zile",
    "rsi": "RSI",
    "mkt_regime": "trendul pieței (S&P 500)",
    "mkt_ret_1m": "randamentul pieței în ultima lună",
    "earn_surprise": "surpriza la ultimele rezultate",
}
CLIPS = {
    "mom_12_1": (-1.0, 3.0),
    "ret_1m": (-0.5, 0.5),
    "high_52w": (-0.9, 0.0),
    "vol_20d": (0.0, 2.0),
    "dist_sma200": (-0.7, 1.5),
}


def feature_frame(
    prices: pd.DataFrame,
    market_close: pd.Series | None = None,
    quarters: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    close = prices["Close"]
    ind = compute_all(prices)
    f = pd.DataFrame(index=close.index)
    f["tech"] = technical_score(ind) / 100
    f["mom_12_1"] = momentum_12_1(close)
    f["ret_1m"] = close / close.shift(21) - 1
    f["high_52w"] = high_52w_gap(close)
    f["vol_20d"] = realized_vol(close)
    f["dist_sma200"] = close / ind["sma200"] - 1
    f["rsi"] = ind["rsi"] / 100 - 0.5
    if market_close is not None:
        m = market_close.reindex(close.index, method="ffill")
        f["mkt_regime"] = regime_series(m)
        f["mkt_ret_1m"] = m / m.shift(21) - 1
    else:
        f["mkt_regime"] = 0.0
        f["mkt_ret_1m"] = 0.0
    f["earn_surprise"] = earnings_surprise_series(quarters, close.index)
    for col, (lo, hi) in CLIPS.items():
        f[col] = f[col].clip(lo, hi)
    return f


def build_dataset(
    prices: dict[str, pd.DataFrame],
    market_close: pd.Series | None = None,
    earnings: dict[str, list[dict[str, Any]]] | None = None,
    horizon: int = 20,
    step: int = 5,
) -> pd.DataFrame:
    """Un rând la fiecare `step` zile pentru fiecare acțiune: semnalele din ziua respectivă + dacă prețul a urcat."""
    frames = []
    for ticker, px in prices.items():
        f = feature_frame(px, market_close, (earnings or {}).get(ticker))
        close = px["Close"]
        fwd = close.shift(-horizon) / close - 1
        f["label"] = (fwd > 0).astype(float).where(fwd.notna())
        f = f.dropna().iloc[::step]
        if f.empty:
            continue
        f["ticker"] = ticker
        f["date"] = f.index
        frames.append(f.reset_index(drop=True))
    if not frames:
        raise ValueError("Nu există destule date pentru antrenare")
    return pd.concat(frames, ignore_index=True)


class ProbabilityModel:
    def __init__(self, horizon: int = 20, C: float = 0.3) -> None:
        self.horizon = horizon
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        self.base_rate: float | None = None
        self.info: dict[str, Any] = {}

    def fit(self, data: pd.DataFrame) -> ProbabilityModel:
        self.pipe.fit(data[FEATURES].to_numpy(), data["label"].astype(int).to_numpy())
        self.base_rate = float(data["label"].mean())
        self.info = {
            "rows": int(len(data)),
            "tickers": int(data["ticker"].nunique()),
            "from": str(pd.Timestamp(data["date"].min()).date()),
            "to": str(pd.Timestamp(data["date"].max()).date()),
        }
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(features[FEATURES].to_numpy())[:, 1]

    def coefficients(self) -> dict[str, float]:
        """Cât contează fiecare semnal (pe scară standardizată; pozitiv = crește probabilitatea)."""
        coef = self.pipe[-1].coef_[0]
        return dict(sorted(zip(FEATURES, map(float, coef)), key=lambda kv: -abs(kv[1])))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> ProbabilityModel:
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, ProbabilityModel):
            raise TypeError("fișierul nu conține un model StockAI")
        return model


@dataclass
class BacktestReport:
    horizon: int
    years: list[dict[str, Any]] = field(default_factory=list)
    brier_model: float = float("nan")
    brier_base: float = float("nan")
    auc: float = float("nan")
    n: int = 0
    calibration: list[dict[str, Any]] = field(default_factory=list)
    top_decile_up: float = float("nan")
    overall_up: float = float("nan")

    @property
    def skill(self) -> float:
        """Cât de mult reduce modelul eroarea față de rata de bază (0 = deloc, negativ = mai rău)."""
        return 1 - self.brier_model / self.brier_base if self.brier_base else float("nan")

    def render(self) -> str:
        lines = [
            f"Test walk-forward, orizont {self.horizon} zile, {self.n} predicții:",
            f"  Eroare Brier: model {self.brier_model:.4f} · rata de bază {self.brier_base:.4f} · îmbunătățire {self.skill:+.1%}",
            f"  AUC {self.auc:.3f} (0,5 = ghicit la întâmplare)",
            f"  În cele 10% cazuri cu probabilitatea cea mai mare, prețul a urcat în {self.top_decile_up:.1%}"
            f" (media: {self.overall_up:.1%})",
            "  Calibrare (probabilitate prezisă → cât de des a urcat de fapt):",
        ]
        for c in self.calibration:
            lines.append(f"    {c['predicted']:.1%} → {c['actual']:.1%}  (n={c['n']})")
        lines.append("  Pe ani:")
        for y in self.years:
            lines.append(
                f"    {y['year']}: model {y['brier_model']:.4f} vs bază {y['brier_base']:.4f}, AUC {y['auc']:.3f}, "
                f"a urcat în {y['up_rate']:.0%} din cazuri"
            )
        return "\n".join(lines)


def walk_forward(data: pd.DataFrame, horizon: int = 20, min_train_years: int = 3) -> BacktestReport:
    data = data.sort_values("date").reset_index(drop=True)
    years = sorted(pd.DatetimeIndex(data["date"]).year.unique())
    report = BacktestReport(horizon=horizon)
    preds, labels, bases = [], [], []
    for year in years[min_train_years:]:
        # Eliminăm ultimele zile dinaintea anului testat: etichetele lor depind de prețuri din anul testat.
        cutoff = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=int(horizon * 1.5) + 1)
        train = data[data["date"] < cutoff]
        test = data[pd.DatetimeIndex(data["date"]).year == year]
        if len(train) < 500 or len(test) < 50 or train["label"].nunique() < 2:
            continue
        model = ProbabilityModel(horizon).fit(train)
        p = model.predict(test)
        y = test["label"].to_numpy()
        base = np.full_like(p, model.base_rate)
        report.years.append(
            {
                "year": int(year),
                "n": int(len(test)),
                "up_rate": float(y.mean()),
                "brier_model": float(np.mean((p - y) ** 2)),
                "brier_base": float(np.mean((base - y) ** 2)),
                "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            }
        )
        preds.append(p)
        labels.append(y)
        bases.append(base)
    if not preds:
        raise ValueError("Prea puțini ani de date pentru un test walk-forward")
    p, y, b = np.concatenate(preds), np.concatenate(labels), np.concatenate(bases)
    report.n = int(len(p))
    report.brier_model = float(np.mean((p - y) ** 2))
    report.brier_base = float(np.mean((b - y) ** 2))
    report.auc = float(roc_auc_score(y, p))
    report.overall_up = float(y.mean())
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    for k in range(10):
        mask = bins == k
        if mask.any():
            report.calibration.append({"predicted": float(p[mask].mean()), "actual": float(y[mask].mean()), "n": int(mask.sum())})
    report.top_decile_up = float(y[bins == 9].mean()) if (bins == 9).any() else float("nan")
    return report
