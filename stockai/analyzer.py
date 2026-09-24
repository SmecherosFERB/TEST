"""Leagă totul: date → indicatori → scoruri → statistică istorică → (la nevoie) Claude."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Literal

import pandas as pd

from .advisor import AdvisorError, ClaudeAdvisor, ClaudeVerdict
from .calibration import HistoricalOdds, historical_odds
from .config import Settings
from .data import DataError, DataProvider
from .indicators import compute_all
from .scoring import (
    composite_score,
    decision_from_score,
    fundamental_score,
    sentiment_score,
    technical_score,
)

log = logging.getLogger(__name__)

ClaudeMode = Literal["auto", "always", "never"]
COMPONENT_NAMES = {"technical": "tehnic", "fundamental": "fundamental", "sentiment": "sentiment"}


@dataclass
class Recommendation:
    ticker: str
    as_of: str
    price: float
    scores: dict[str, float | None]
    rule_decision: str
    odds: HistoricalOdds | None
    ambiguity_reasons: list[str] = field(default_factory=list)
    claude: ClaudeVerdict | None = None
    claude_error: str | None = None

    @property
    def decision(self) -> str:
        return self.claude.decision if self.claude else self.rule_decision

    @property
    def decided_by(self) -> str:
        return "Claude" if self.claude else "reguli"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "price": self.price,
            "decision": self.decision,
            "decided_by": self.decided_by,
            "rule_decision": self.rule_decision,
            "scores": self.scores,
            "historical_odds": asdict(self.odds) if self.odds else None,
            "ambiguity_reasons": self.ambiguity_reasons,
            "claude": self.claude.model_dump() if self.claude else None,
            "claude_error": self.claude_error,
        }


def find_ambiguity(
    scores: dict[str, float | None],
    rule_decision: str,
    odds: HistoricalOdds | None,
    settings: Settings,
) -> list[str]:
    """Motivele pentru care regulile nu sunt suficient de sigure. Listă goală = semnal clar."""
    reasons = []
    composite = scores["composite"]
    if rule_decision == "HOLD":
        reasons.append(f"scor compus neutru ({composite:+.0f}, prag ±{settings.buy_threshold:.0f})")

    parts = {k: v for k, v in scores.items() if k in COMPONENT_NAMES and v is not None}
    for (a, va), (b, vb) in combinations(parts.items(), 2):
        if va * vb < 0 and min(abs(va), abs(vb)) >= settings.conflict_strength:
            reasons.append(f"{COMPONENT_NAMES[a]} ({va:+.0f}) contrazice {COMPONENT_NAMES[b]} ({vb:+.0f})")

    if odds is None or odds.samples < settings.min_samples:
        reasons.append("prea puține situații similare în istoric")
    elif abs(odds.edge) < settings.min_edge:
        reasons.append(
            f"istoric fără avantaj: {odds.probability_up:.0%} față de rata de bază {odds.base_rate:.0%}"
        )
    elif (rule_decision == "BUY" and odds.edge < 0) or (rule_decision == "SELL" and odds.edge > 0):
        reasons.append(
            f"regulile spun {rule_decision}, dar istoric prețul a urcat în {odds.probability_up:.0%} "
            f"din cazuri (rata de bază {odds.base_rate:.0%})"
        )
    return reasons


class Analyzer:
    def __init__(
        self,
        data: DataProvider,
        settings: Settings | None = None,
        advisor: ClaudeAdvisor | None = None,
        claude_mode: ClaudeMode = "auto",
    ) -> None:
        self.data = data
        self.settings = settings or Settings()
        self.advisor = advisor
        self.claude_mode = claude_mode

    def analyze(self, ticker: str) -> Recommendation:
        s = self.settings
        prices = self.data.prices(ticker, s.history_period)
        ind = compute_all(prices)
        tech = technical_score(ind)
        if pd.isna(tech.iloc[-1]):
            raise DataError(f"{ticker}: istoric prea scurt (sunt necesare cel puțin ~200 de zile)")

        fundamentals = self.data.fundamentals(ticker)
        news = self.data.news(ticker)
        scores: dict[str, float | None] = {
            "technical": float(tech.iloc[-1]),
            "fundamental": fundamental_score(fundamentals),
            "sentiment": sentiment_score(news),
        }
        scores["composite"] = composite_score(scores, s.weights)
        rule_decision = decision_from_score(scores["composite"], s.buy_threshold)
        odds = historical_odds(ind["close"], tech, scores["technical"], s.horizon_days, s.bucket_width)

        rec = Recommendation(
            ticker=ticker.upper(),
            as_of=str(prices.index[-1].date()),
            price=float(prices["Close"].iloc[-1]),
            scores=scores,
            rule_decision=rule_decision,
            odds=odds,
            ambiguity_reasons=find_ambiguity(scores, rule_decision, odds, s),
        )

        ask = self.claude_mode == "always" or (self.claude_mode == "auto" and rec.ambiguity_reasons)
        if ask and self.advisor:
            try:
                rec.claude = self.advisor.advise(self._claude_context(rec, ind, fundamentals, news))
            except AdvisorError as exc:
                log.warning("%s: %s", ticker, exc)
                rec.claude_error = str(exc)
        return rec

    def _claude_context(
        self,
        rec: Recommendation,
        ind: pd.DataFrame,
        fundamentals: dict[str, Any],
        news: list[dict[str, Any]],
    ) -> dict[str, Any]:
        last = ind.iloc[-1]
        close = ind["close"]

        def change(days: int) -> float | None:
            return round(float(close.iloc[-1] / close.iloc[-days - 1] - 1), 4) if len(close) > days else None

        return {
            "ticker": rec.ticker,
            "horizon_days": self.settings.horizon_days,
            "ambiguity_reasons": rec.ambiguity_reasons or ["utilizatorul a cerut explicit opinia ta"],
            "as_of": rec.as_of,
            "price": round(rec.price, 2),
            "price_change": {"1m": change(21), "3m": change(63), "6m": change(126), "1y": change(252)},
            "indicators": {
                "rsi14": round(float(last["rsi"]), 1),
                "price_vs_sma50": round(float(last["close"] / last["sma50"] - 1), 4),
                "price_vs_sma200": round(float(last["close"] / last["sma200"] - 1), 4),
                "macd_histogram": round(float(last["macd_hist"]), 4),
                "bollinger_pct_b": round(float(last["pct_b"]), 2),
                "volume_vs_20d_avg": round(float(last["volume_ratio"]), 2),
            },
            "scores_-100_to_100": {k: None if v is None else round(v, 1) for k, v in rec.scores.items()},
            "rule_based_decision": rec.rule_decision,
            "historical_odds_technical_only": (
                {
                    "prob_up_when_score_similar": round(rec.odds.probability_up, 3),
                    "base_rate_prob_up": round(rec.odds.base_rate, 3),
                    "avg_return_when_similar": round(rec.odds.avg_return, 4),
                    "overlapping_samples": rec.odds.samples,
                }
                if rec.odds
                else None
            ),
            "fundamentals": fundamentals,
            "news": news,
        }
