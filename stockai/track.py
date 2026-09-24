"""Jurnalul predicțiilor: fiecare analiză salvează o predicție, iar `--evaluate` o verifică după ~4 săptămâni.

Doar așa aflăm dacă probabilitățile sunt „reale”: comparăm ce am estimat cu ce s-a întâmplat.
"""

from __future__ import annotations

import csv
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

FIELDS = ["id", "ticker", "made_on", "prob", "base", "source", "resolved_on", "up", "ret"]
DUE_DAYS = 28  # ≈ 20 de zile de tranzacționare


def default_path() -> Path:
    return Path(os.getenv("STOCKAI_CACHE", ".cache")) / "predictions.csv"


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def log_prediction(ticker: str, made_on: date, prob: float, base: float, source: str, path: Path | None = None) -> bool:
    """O predicție pe acțiune, pe săptămână și pe sursă. Întoarce False dacă exista deja."""
    path = path or default_path()
    monday = made_on - timedelta(days=made_on.weekday())
    pid = f"{ticker}|{monday.isoformat()}|{source}"
    rows = _read(path)
    if any(r["id"] == pid for r in rows):
        return False
    rows.append({"id": pid, "ticker": ticker, "made_on": made_on.isoformat(), "prob": f"{prob:.4f}",
                 "base": f"{base:.4f}", "source": source, "resolved_on": "", "up": "", "ret": ""})
    _write(path, rows)
    return True


def resolve(prices_for, path: Path | None = None, today: date | None = None) -> int:
    """Completează rezultatul predicțiilor scadente. `prices_for(ticker)` întoarce un DataFrame cu coloana Close."""
    path = path or default_path()
    today = today or date.today()
    rows = _read(path)
    cache: dict[str, Any] = {}
    done = 0
    for r in rows:
        if r["up"]:
            continue
        made = date.fromisoformat(r["made_on"])
        target = made + timedelta(days=DUE_DAYS)
        if target > today:
            continue
        t = r["ticker"]
        if t not in cache:
            try:
                cache[t] = prices_for(t)["Close"]
            except Exception:
                cache[t] = None
        close = cache[t]
        if close is None or close.empty or close.index[-1].date() < target:
            continue
        start = close[close.index.date <= made]
        end = close[close.index.date >= target]
        if start.empty or end.empty:
            continue
        ret = float(end.iloc[0] / start.iloc[-1] - 1)
        r.update({"resolved_on": str(end.index[0].date()), "up": "1" if ret > 0 else "0", "ret": f"{ret:.4f}"})
        done += 1
    if done:
        _write(path, rows)
    return done


def report(path: Path | None = None) -> str:
    rows = _read(path or default_path())
    lines = []
    pending = sum(1 for r in rows if not r["up"])
    for source, title in (("stat", "Statistica programului"), ("claude", "Claude")):
        done = [r for r in rows if r["source"] == source and r["up"]]
        if not done:
            lines.append(f"{title}: încă nicio predicție verificată.")
            continue
        p = np.array([float(r["prob"]) for r in done])
        b = np.array([float(r["base"]) for r in done])
        y = np.array([int(r["up"]) for r in done])
        brier, brier_base = float(np.mean((p - y) ** 2)), float(np.mean((b - y) ** 2))
        skill = 1 - brier / brier_base if brier_base else 0.0
        lines.append(f"{title}: {len(done)} predicții verificate, prețul a urcat în {y.mean():.0%} din cazuri.")
        lines.append(f"  Eroare Brier {brier:.3f} față de {brier_base:.3f} cu „ca de obicei” ({skill:+.1%}).")
        for lo, hi, label in ((0, 0.5, "sub 50%"), (0.5, 0.55, "50–55%"), (0.55, 0.6, "55–60%"),
                              (0.6, 0.65, "60–65%"), (0.65, 1.01, "peste 65%")):
            mask = (p >= lo) & (p < hi)
            if mask.any():
                lines.append(f"  Am estimat {label}: {mask.sum()} cazuri, a urcat de fapt în {y[mask].mean():.0%}")
    lines.append(f"Așteaptă verificarea: {pending}. Rezultatele devin de încredere de la ~100 de predicții verificate.")
    return "\n".join(lines)
