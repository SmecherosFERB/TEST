"""Model de probabilitate învățat pe toată lista de acțiuni, pentru două întrebări:

- „up”: va fi prețul mai mare peste `horizon` zile?
- „beat”: va crește acțiunea mai mult decât S&P 500 în același interval?

Regresie logistică cu penalizare pe semnale cu dovezi publicate. Înainte de a o folosi, o verificăm
„walk-forward”: pentru fiecare an, modelul învață doar din anii anteriori și e testat pe anul respectiv.
Pe rezultatele din anii nevăzuți recalibrăm probabilitățile (Platt): dacă ordinea dată de model nu a contat
sigur statistic, probabilitățile rămân aproape de medie, cu intervalul lor de încredere.
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

from .calibration import Z90, wilson_interval
from .indicators import compute_all
from .quality import check_prices, excluded_rows
from .scoring import technical_score
from .signals import earnings_surprise_series, high_52w_gap, momentum_12_1, realized_vol, regime_series

FEATURES = [
    "tech",
    "mom_12_1",
    "ret_1m",
    "high_52w",
    "vol_60d",
    "dist_sma200",
    "volume_trend",
    "mkt_regime",
    "mkt_ret_1m",
    "mom_bear",
    "beta_1y",
    "earn_surprise",
]
FEATURE_NAMES = {
    "tech": "scor tehnic",
    "mom_12_1": "momentum 12 luni (fără ultima)",
    "ret_1m": "randamentul ultimei luni",
    "high_52w": "apropierea de maximul pe 52 săpt.",
    "vol_60d": "volatilitate 60 zile",
    "dist_sma200": "distanța față de media pe 200 zile",
    "volume_trend": "volum neobișnuit (luna față de an)",
    "mkt_regime": "trendul pieței (S&P 500)",
    "mkt_ret_1m": "randamentul pieței în ultima lună",
    "mom_bear": "momentum după un an slab al pieței",
    "beta_1y": "beta față de S&P 500 (1 an)",
    "earn_surprise": "surpriza la ultimele rezultate",
}
TARGETS = {"up": "creștere", "beat": "bate S&P 500"}
# Două clase de acțiuni ale aceleiași companii au aproape aceleași prețuri: în model contează o singură dată.
SAME_COMPANY = {"GOOG": "GOOGL", "BRK.A": "BRK.B", "FOX": "FOXA", "NWS": "NWSA", "UA": "UAA", "LEN.B": "LEN", "HEI.A": "HEI"}
# Penalizare L2 pe observație (λ = RIDGE · n): efectele reale sunt mici, zgomotul e mare.
RIDGE = 0.1


def feature_frame(
    prices: pd.DataFrame,
    market_close: pd.Series | None = None,
    quarters: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    close = prices["Close"]
    ind = compute_all(prices)
    f = pd.DataFrame(index=close.index)
    f["tech"] = technical_score(ind) / 100
    f["mom_12_1"] = np.log1p(momentum_12_1(close).clip(-0.9, 4.0))
    f["ret_1m"] = (close / close.shift(21) - 1).clip(-0.4, 0.4)
    f["high_52w"] = high_52w_gap(close).clip(-0.8, 0.0)
    f["vol_60d"] = np.log(realized_vol(close, 60).clip(0.05, 2.0))
    f["dist_sma200"] = (close / ind["sma200"] - 1).clip(-0.6, 1.0)
    volume = prices["Volume"] if "Volume" in prices else pd.Series(0.0, index=close.index)
    v20, v250 = volume.rolling(20).mean(), volume.rolling(250).mean()
    trend = np.log(v20.where(v20 > 0) / v250.where(v250 > 0))
    f["volume_trend"] = trend.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.5, 1.5)
    if market_close is not None:
        m = market_close.reindex(close.index, method="ffill")
        f["mkt_regime"] = regime_series(m)
        f["mkt_ret_1m"] = (m / m.shift(21) - 1).clip(-0.3, 0.3)
        # Momentumul se strică după un an slab al pieței (Daniel și Moskowitz, 2016).
        bear = (m / m.shift(252) - 1) < 0
        f["mom_bear"] = f["mom_12_1"].where(bear, 0.0).where(m.shift(252).notna())
        # Beta pe ultimul an (Frazzini și Pedersen, 2014: acțiunile cu beta mare au randamente ajustate la risc mai mici).
        rs, rm = np.log(close).diff(), np.log(m).diff()
        f["beta_1y"] = (rs.rolling(252, min_periods=200).cov(rm) / rm.rolling(252, min_periods=200).var()).clip(-0.5, 3.0)
    else:
        f["mkt_regime"] = 0.0
        f["mkt_ret_1m"] = 0.0
        f["mom_bear"] = 0.0
        f["beta_1y"] = 1.0
    f["earn_surprise"] = earnings_surprise_series(quarters, close.index)
    return f[FEATURES]


def build_dataset(
    prices: dict[str, pd.DataFrame],
    market_close: pd.Series | None = None,
    earnings: dict[str, list[dict[str, Any]]] | None = None,
    horizon: int = 20,
    step: int = 20,
    target: str = "up",
) -> pd.DataFrame:
    """Un rând la fiecare `step` zile pentru fiecare acțiune: semnalele din ziua respectivă + ce a urmat.

    Implicit `step = horizon`, deci ferestrele nu se suprapun și fiecare rând e un caz nou.
    Semnalul e calculat la închiderea zilei; intrarea realistă e la închiderea următoare, ieșirea după
    `horizon` zile. Seriile cu date slabe nu intră deloc, iar rândurile din jurul unui split neajustat sunt scoase.
    """
    if target not in TARGETS:
        raise ValueError(f"țintă necunoscută: {target}")
    if target == "beat" and market_close is None:
        raise ValueError("Pentru ținta „bate S&P 500” e nevoie de prețurile S&P 500")
    frames, excluded, dropped = [], {}, 0
    for ticker, px in prices.items():
        if SAME_COMPANY.get(ticker) in prices:
            excluded[ticker] = f"aceeași companie cu {SAME_COMPANY[ticker]}"
            continue
        quality = check_prices(px, today=px.index[-1].date())
        if quality.grade == "slabă":
            excluded[ticker] = "date slabe: " + ", ".join(c.label.lower() for c in quality.failed)
            continue
        f = feature_frame(px, market_close, (earnings or {}).get(ticker))
        close = px["Close"]
        fwd = close.shift(-(horizon + 1)) / close.shift(-1) - 1
        if target == "beat":
            m = market_close.reindex(close.index, method="ffill")
            mfwd = m.shift(-(horizon + 1)) / m.shift(-1) - 1
            f["label"] = (fwd > mfwd).astype(float).where(fwd.notna() & mfwd.notna())
        else:
            f["label"] = (fwd > 0).astype(float).where(fwd.notna())
        if quality.suspect:
            bad = excluded_rows(close.index, quality.suspect, horizon=horizon)
            dropped += int(bad.sum())
            f.loc[bad, "label"] = np.nan
        f = f.dropna().iloc[::-1].iloc[::step].iloc[::-1]  # pornim de la cel mai recent rând cu rezultat
        if f.empty:
            continue
        f["ticker"] = ticker
        f["date"] = f.index
        frames.append(f.reset_index(drop=True))
    if not frames:
        raise ValueError("Nu există destule date pentru antrenare")
    data = pd.concat(frames, ignore_index=True)
    data.attrs["step"] = step
    data.attrs["excluded"] = excluded
    data.attrs["days_dropped_near_splits"] = dropped
    return data


# ---------- recalibrare pe anii nevăzuți ----------

def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def design_effect(residuals: np.ndarray, groups: np.ndarray) -> float:
    """Acțiunile urcă și coboară împreună: câte observații „valorează” una independentă (ANOVA pe luni)."""
    frame = pd.DataFrame({"e": residuals, "g": groups})
    sizes = frame.groupby("g")["e"].size()
    G, N = len(sizes), len(frame)
    if G < 3 or N <= G:
        return 1.0
    means = frame.groupby("g")["e"].transform("mean")
    ssb = float(((means - frame["e"].mean()) ** 2).sum())
    ssw = float(((frame["e"] - means) ** 2).sum())
    msb, msw = ssb / (G - 1), ssw / (N - G)
    m0 = (N - float((sizes**2).sum()) / N) / (G - 1)
    rho = max(0.0, (msb - msw) / (msb + (m0 - 1) * msw)) if msb + (m0 - 1) * msw > 0 else 0.0
    return 1 + (N / G - 1) * rho


@dataclass
class Calibration:
    a: float
    b: float  # panta după micșorare (0 = modelul nu a ajutat sigur)
    raw_b: float
    se_b: float
    cov: list[list[float]]
    zm: float
    deff: float
    overall: float  # cât de des s-a întâmplat, în anii de test

    @property
    def helps(self) -> bool:
        """Ordinea dată de model a contat sigur statistic în anii nevăzuți."""
        return self.b > 0 and self.raw_b > Z90 * self.se_b

    def apply(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zc = _logit(np.asarray(p, dtype=float)) - self.zm
        eta = self.a + self.b * zc
        var = self.cov[0][0] + 2 * zc * self.cov[0][1] + zc * zc * self.cov[1][1]
        sd = np.sqrt(np.maximum(var, 0))
        return _sigmoid(eta), _sigmoid(eta - Z90 * sd), _sigmoid(eta + Z90 * sd)


def fit_calibration(p: np.ndarray, y: np.ndarray, groups: np.ndarray, overlap: float = 1.0) -> Calibration:
    """Platt pe predicțiile din anii nevăzuți; incertitudinea ține cont de efectul de grup.

    `overlap` > 1 când ferestrele se suprapun (rânduri la mai puțin de `horizon` zile): atunci și rândurile
    aceleiași acțiuni din luni vecine se repetă, așa că incertitudinea crește în consecință.
    """
    z = _logit(p)
    zm = float(z.mean())
    X = np.column_stack([np.ones_like(z), z - zm])
    w = np.array([float(_logit(np.array([y.mean()]))[0]), 0.0])
    H = np.eye(2)
    for _ in range(50):
        mu = _sigmoid(X @ w)
        g = X.T @ (y - mu)
        H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    deff = design_effect(y - p, groups) * max(1.0, overlap)
    cov = np.linalg.inv(H) * deff
    b, vb = float(w[1]), float(cov[1, 1])
    keep = max(0.0, 1 - vb / (b * b)) if b > 0 else 0.0
    return Calibration(a=float(w[0]), b=b * keep, raw_b=b, se_b=float(np.sqrt(vb)), cov=cov.tolist(), zm=zm,
                       deff=float(deff), overall=float(y.mean()))


class ProbabilityModel:
    def __init__(self, horizon: int = 20, target: str = "up", C: float | None = None) -> None:
        self.horizon = horizon
        self.target = target
        self.C = C
        self.features = list(FEATURES)
        self.pipe = None
        self.base_rate: float | None = None
        self.info: dict[str, Any] = {}
        self.calibration: Calibration | None = None

    def fit(self, data: pd.DataFrame) -> ProbabilityModel:
        C = self.C if self.C is not None else 1 / (RIDGE * len(data))
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        self.pipe.fit(data[self.features].to_numpy(), data["label"].astype(int).to_numpy())
        self.base_rate = float(data["label"].mean())
        self.info = {
            "rows": int(len(data)),
            "tickers": int(data["ticker"].nunique()),
            "from": str(pd.Timestamp(data["date"].min()).date()),
            "to": str(pd.Timestamp(data["date"].max()).date()),
            "target": self.target,
        }
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Probabilitatea brută a modelului."""
        return self.pipe.predict_proba(features[self.features].to_numpy())[:, 1]

    def estimate(self, features: pd.DataFrame) -> dict[str, np.ndarray | float]:
        """Probabilitatea recalibrată pe anii nevăzuți, cu intervalul de 90%. Fără recalibrare: cea brută."""
        raw = self.predict(features)
        if self.calibration is None:
            nan = np.full_like(raw, np.nan)
            return {"p": raw, "lo": nan, "hi": nan, "base": self.base_rate, "raw": raw}
        p, lo, hi = self.calibration.apply(raw)
        return {"p": p, "lo": lo, "hi": hi, "base": self.calibration.overall, "raw": raw}

    def coefficients(self) -> dict[str, float]:
        """Cât contează fiecare semnal (pe scară standardizată; pozitiv = crește probabilitatea)."""
        coef = self.pipe[-1].coef_[0]
        return dict(sorted(zip(self.features, map(float, coef)), key=lambda kv: -abs(kv[1])))

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
        if getattr(model, "features", None) != FEATURES:
            raise TypeError("modelul e dintr-o versiune mai veche; rulează din nou python -m stockai --train")
        return model


@dataclass
class BacktestReport:
    horizon: int
    target: str = "up"
    years: list[dict[str, Any]] = field(default_factory=list)
    brier_model: float = float("nan")
    brier_base: float = float("nan")
    auc: float = float("nan")
    n: int = 0
    calibration: list[dict[str, Any]] = field(default_factory=list)
    groups: list[dict[str, Any]] = field(default_factory=list)
    top_decile_up: float = float("nan")
    overall_up: float = float("nan")
    recalibration: Calibration | None = None

    @property
    def skill(self) -> float:
        """Cât de mult reduce modelul eroarea față de rata de bază (0 = deloc, negativ = mai rău)."""
        return 1 - self.brier_model / self.brier_base if self.brier_base else float("nan")

    @property
    def n_independent(self) -> float:
        return self.n / self.recalibration.deff if self.recalibration else float(self.n)

    def render(self) -> str:
        what = "a urcat" if self.target == "up" else "a bătut S&P 500"
        lines = [
            f"Test walk-forward ({TARGETS[self.target]}), orizont {self.horizon} zile, {self.n} predicții"
            f" (~{self.n_independent:.0f} cazuri independente):",
            f"  Eroare Brier: model {self.brier_model:.4f} · rata de bază {self.brier_base:.4f} · îmbunătățire {self.skill:+.1%}",
            f"  AUC {self.auc:.3f} (0,5 = ghicit la întâmplare)",
            f"  În cele 10% cazuri cu probabilitatea cea mai mare, {what} în {self.top_decile_up:.1%}"
            f" (media: {self.overall_up:.1%})",
            "  Pe grupe de câte 20% (de la cele mai slabe la cele mai bune după model):",
        ]
        for k, g in enumerate(self.groups, 1):
            lines.append(f"    grupa {k}: {g['actual']:.1%} (interval 90%: {g['lo']:.1%}–{g['hi']:.1%}, n={g['n']})")
        if self.recalibration:
            r = self.recalibration
            lines.append(
                f"  Recalibrare: pantă {r.raw_b:.2f} ± {r.se_b:.2f} → "
                + ("ordinea dată de model a contat sigur statistic." if r.helps
                   else "ordinea dată de model NU a contat sigur statistic; probabilitățile rămân aproape de medie.")
            )
        lines.append("  Calibrare brută (probabilitate prezisă → cât de des s-a întâmplat):")
        for c in self.calibration:
            lines.append(f"    {c['predicted']:.1%} → {c['actual']:.1%}  (n={c['n']})")
        lines.append("  Pe ani:")
        for y in self.years:
            lines.append(
                f"    {y['year']}: model {y['brier_model']:.4f} vs bază {y['brier_base']:.4f}, AUC {y['auc']:.3f}, "
                f"s-a întâmplat în {y['up_rate']:.0%} din cazuri"
            )
        return "\n".join(lines)


def walk_forward(data: pd.DataFrame, horizon: int = 20, min_train_years: int = 3, target: str = "up") -> BacktestReport:
    overlap = horizon / max(1, int(data.attrs.get("step", horizon)))
    data = data.sort_values("date").reset_index(drop=True)
    years = sorted(pd.DatetimeIndex(data["date"]).year.unique())
    report = BacktestReport(horizon=horizon, target=target)
    preds, labels, bases, months = [], [], [], []
    for year in years[min_train_years:]:
        # Eliminăm ultimele zile dinaintea anului testat: etichetele lor depind de prețuri din anul testat.
        cutoff = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=int(horizon * 1.6) + 3)
        train = data[data["date"] < cutoff]
        test = data[pd.DatetimeIndex(data["date"]).year == year]
        if len(train) < 400 or len(test) < 20 or train["label"].nunique() < 2:
            continue
        model = ProbabilityModel(horizon, target).fit(train)
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
        months.append(pd.DatetimeIndex(test["date"]).strftime("%Y-%m").to_numpy())
    if not preds:
        raise ValueError("Prea puțini ani de date pentru un test walk-forward")
    p, y, b, g = np.concatenate(preds), np.concatenate(labels), np.concatenate(bases), np.concatenate(months)
    report.n = int(len(p))
    report.brier_model = float(np.mean((p - y) ** 2))
    report.brier_base = float(np.mean((b - y) ** 2))
    report.auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    report.overall_up = float(y.mean())
    report.recalibration = fit_calibration(p, y, g, overlap)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    for k in range(10):
        mask = bins == k
        if mask.any():
            report.calibration.append({"predicted": float(p[mask].mean()), "actual": float(y[mask].mean()), "n": int(mask.sum())})
    report.top_decile_up = float(y[bins == 9].mean()) if (bins == 9).any() else float("nan")
    order = np.argsort(p, kind="stable")
    for part in np.array_split(order, 5):
        if len(part):
            act = float(y[part].mean())
            lo, hi = wilson_interval(act, len(part) / report.recalibration.deff)
            report.groups.append({"n": int(len(part)), "actual": act, "lo": lo, "hi": hi})
    return report
