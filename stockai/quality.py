"""Calitatea datelor: verificările unui desk profesionist înainte de a folosi o serie de prețuri.

- salturi de o zi care arată ca un split neajustat (2:1, 3:1, 10:1 ...): perioadele din jur nu intră în model;
- date vechi, zile lipsă, cotații înghețate, volum lipsă, istoric prea scurt;
- lichiditate și preț minim (regulile de selecție);
- verificare încrucișată cu maximul pe 52 de săptămâni raportat separat de furnizor;
- doar bare închise: bara zilei curente e incompletă până la închiderea bursei din New York.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SPLIT_RATIOS = (1.5, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50)
MIN_ADV = 20e6  # dolari tranzacționați pe zi, media ultimelor 20 de zile
MIN_PRICE = 5.0
NEW_YORK = ZoneInfo("America/New_York")
CLOSE = time(16, 15)  # 16:00 plus 15 minute, ca prețul de închidere să fie publicat


@dataclass
class Check:
    ok: bool
    label: str
    detail: str


@dataclass
class Quality:
    score: int
    grade: str
    checks: list[Check]
    suspect: list[pd.Timestamp] = field(default_factory=list)
    adv: float = 0.0
    price: float = 0.0
    stale: bool = False

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def liquid(self) -> bool:
        return self.adv >= MIN_ADV and self.price >= MIN_PRICE

    @property
    def tradable(self) -> bool:
        """Poate intra într-o selecție profesionistă: date folosibile, la zi, lichide."""
        return self.grade != "slabă" and not self.stale and self.liquid

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "grade": self.grade,
            "avg_daily_dollar_volume": round(self.adv),
            "tradable": self.tradable,
            "failed_checks": [f"{c.label}: {c.detail}" for c in self.failed],
            "checks": [asdict(c) for c in self.checks],
        }


def _money(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.1f} mld. $".replace(".", ",")
    if x >= 1e6:
        return f"{x / 1e6:.0f} mil. $"
    return f"{x / 1e3:.0f} mii $"


def split_like_jumps(close: pd.Series) -> tuple[list[pd.Timestamp], int]:
    """Zilele cu salturi de tip split (raport apropiat de 2, 3, 4, 10...) și câte alte salturi extreme există."""
    ratio = (close / close.shift(1)).dropna()
    big = ratio[np.abs(np.log(ratio.where(ratio > 0))) > np.log(1.45)]
    suspect, extreme = [], 0
    for day, r in big.items():
        k = 1 / r if r < 1 else r
        if any(abs(k / s - 1) < 0.03 for s in SPLIT_RATIOS):
            suspect.append(day)
        else:
            extreme += 1
    return suspect, extreme


def check_prices(
    prices: pd.DataFrame,
    today: date | None = None,
    fifty_two_week_high: float | None = None,
) -> Quality:
    today = today or date.today()
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float) if "Volume" in prices else pd.Series(0.0, index=close.index)
    n = len(close)
    checks: list[Check] = []
    penalty = 0

    suspect, extreme = split_like_jumps(close)
    three_years_ago = close.index[-1] - pd.Timedelta(days=3 * 365)
    if any(d > three_years_ago for d in suspect):
        penalty += 40
    elif suspect:
        penalty += 10
    checks.append(Check(
        not suspect, "Fără split-uri neajustate",
        f"{len(suspect)} zile cu salturi de tip split (ultima: {suspect[-1].date()}); perioadele din jur nu intră în model"
        if suspect else "nicio mișcare de o zi de tip 2:1, 3:1, 10:1",
    ))

    last = close.index[-1].date()
    late = int(np.busday_count(last + pd.Timedelta(days=1), today + pd.Timedelta(days=1))) if today > last else 0
    stale = late > 3
    if stale:
        penalty += 30
    checks.append(Check(not stale, "Date la zi", f"ultima zi: {last}" + (f" (acum {late} zile lucrătoare)" if stale else "")))

    recent = close[close.index > three_years_ago]
    gaps = int((pd.Series(recent.index).diff().dt.days > 6).sum())
    frozen_runs = (recent.diff() == 0).astype(int)
    run_ids = (frozen_runs == 0).cumsum()
    flat = int((frozen_runs.groupby(run_ids).sum() >= 4).sum())
    penalty += min(20, gaps * 5) + (15 if flat else 0)
    checks.append(Check(not gaps, "Fără zile lipsă",
                        f"{gaps} goluri mai lungi de o săptămână în ultimii 3 ani" if gaps else "seria e completă în ultimii 3 ani"))
    checks.append(Check(not flat, "Prețuri care se mișcă",
                        "prețul a rămas identic mai multe zile la rând (cotație înghețată)" if flat else "nicio cotație înghețată"))

    last_year = volume.iloc[-252:]
    zero_vol = float((~(last_year > 0)).mean()) if len(last_year) else 1.0
    if zero_vol > 0.05:
        penalty += 10
    checks.append(Check(zero_vol <= 0.05, "Volum raportat",
                        f"lipsește în {zero_vol:.0%} din zilele ultimului an" if zero_vol > 0.05 else "volum în fiecare zi"))

    years = n / 252
    if years < 2:
        penalty += 10
    checks.append(Check(years >= 2, "Istoric suficient", f"{years:.1f} ani de prețuri zilnice".replace(".", ",")))

    adv = float((close * volume).iloc[-20:].mean())
    price = float(close.iloc[-1])
    liquid = adv >= MIN_ADV and price >= MIN_PRICE
    checks.append(Check(liquid, "Lichiditate", f"{_money(adv)} tranzacționați pe zi, preț {price:.2f}"
                        + ("" if liquid else f" (minim {_money(MIN_ADV)} pe zi și {MIN_PRICE:.0f} $)")))

    if fifty_two_week_high and fifty_two_week_high > 0 and n > 252:
        hi = float(close.iloc[-252:].max())
        ratio = hi / fifty_two_week_high
        match = 0.85 <= ratio <= 1.03
        if not match:
            penalty += 15
        checks.append(Check(match, "Sursele se potrivesc",
                            f"maximul pe 52 de săptămâni: {hi:.2f} în serie, {fifty_two_week_high:.2f} raportat"
                            + ("" if match else " (diferență mare: prețuri posibil neajustate)")))
    if extreme:
        checks.append(Check(True, "Mișcări extreme",
                            f"{extreme} zile cu mișcări de peste 45% (păstrate: par reale, nu split-uri)"))

    score = max(0, 100 - penalty)
    grade = "bună" if score >= 80 else "acceptabilă" if score >= 50 else "slabă"
    return Quality(score, grade, checks, suspect, adv, price, stale)


def excluded_rows(index: pd.DatetimeIndex, suspect: list[pd.Timestamp], lookback: int = 252, horizon: int = 20) -> np.ndarray:
    """Rândurile al căror istoric (un an) sau rezultat (intrare a doua zi + `horizon`) atinge un salt suspect."""
    bad = np.zeros(len(index), dtype=bool)
    positions = np.arange(len(index))
    for day in suspect:
        k = index.get_indexer([day])[0]
        if k >= 0:
            bad |= (positions > k - horizon - 2) & (positions < k + lookback)
    return bad


def drop_partial_bar(prices: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Scoate bara zilei curente dacă bursa din New York nu s-a închis încă (preț și volum parțiale)."""
    if prices.empty:
        return prices
    now = (now or datetime.now(NEW_YORK)).astimezone(NEW_YORK)
    last = prices.index[-1].date()
    if last == now.date() and now.time() < CLOSE:
        return prices.iloc[:-1]
    return prices
