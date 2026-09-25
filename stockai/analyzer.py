"""Leagă totul: date → indicatori → scoruri → statistică istorică și model → (la nevoie) Claude."""

from __future__ import annotations

import logging
from datetime import date
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Literal

import numpy as np
import pandas as pd

from .advisor import AdvisorError, ClaudeAdvisor, ClaudeVerdict
from .calibration import HistoricalOdds, historical_odds, honest_estimate, model_estimate
from .config import Settings
from .data import DataError, DataProvider
from .decision import Decision, base_decision, holding_plan, with_claude
from .indicators import compute_all
from .quality import check_prices
from .scoring import (
    composite_score,
    decision_from_score,
    fundamental_score,
    sentiment_score,
    technical_score,
)
from .signals import earnings_signal, insider_signal, macro_signal, market_signal, realized_vol, revision_signal

log = logging.getLogger(__name__)

ClaudeMode = Literal["auto", "always", "never"]
COMPONENT_NAMES = {
    "technical": "tehnic",
    "fundamental": "fundamental",
    "sentiment": "sentiment",
    "earnings": "rezultate",
    "insiders": "insideri",
    "revisions": "analiști",
    "market": "piață",
}


@dataclass
class Recommendation:
    ticker: str
    as_of: str
    price: float
    scores: dict[str, float | None]
    rule_decision: str
    odds: HistoricalOdds | None
    ambiguity_reasons: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] | None = None
    model_beat: dict[str, Any] | None = None
    model_trade: dict[str, Any] | None = None
    honest: dict[str, Any] | None = None
    claude: ClaudeVerdict | None = None
    claude_error: str | None = None
    final: Decision | None = None  # decizia după reguli de dovezi și verificări (vezi decision.py)

    @property
    def decision(self) -> str:
        if self.final:
            return self.final.action
        return self.claude.decision if self.claude else self.rule_decision

    @property
    def decided_by(self) -> str:
        if self.final:
            return self.final.decided_by
        return "Claude" if self.claude else "reguli"

    @property
    def confidence(self) -> str | None:
        if self.final:
            return self.final.confidence
        return self.claude.confidence if self.claude else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "price": self.price,
            "decision": self.decision,
            "decided_by": self.decided_by,
            "confidence": self.confidence,
            "why": self.final.why if self.final else [],
            "position_size": self.final.size if self.final else None,
            "rule_decision": self.rule_decision,
            "scores": self.scores,
            "historical_odds": asdict(self.odds) if self.odds else None,
            "model": self.model,
            "model_beat_sp500": self.model_beat,
            "model_trade": self.model_trade,
            "estimate": self.honest,
            "extras": self.extras,
            "ambiguity_reasons": self.ambiguity_reasons,
            "claude": self.claude.model_dump() if self.claude else None,
            "claude_error": self.claude_error,
        }


def find_ambiguity(
    scores: dict[str, float | None],
    rule_decision: str,
    odds: HistoricalOdds | None,
    settings: Settings,
    model: dict[str, Any] | None = None,
    honest: dict[str, Any] | None = None,
) -> list[str]:
    """Motivele pentru care regulile nu sunt suficient de sigure. Listă goală = semnal clar."""
    reasons = []
    composite = scores["composite"]
    if rule_decision == "HOLD":
        reasons.append(f"scor compus neutru ({composite:+.0f}, prag ±{settings.buy_threshold:.0f})")

    parts = {k: v for k, v in scores.items() if k in COMPONENT_NAMES and v is not None}
    conflicts = sorted(
        ((min(abs(va), abs(vb)), a, va, b, vb) for (a, va), (b, vb) in combinations(parts.items(), 2)
         if va * vb < 0 and min(abs(va), abs(vb)) >= settings.conflict_strength),
        reverse=True,
    )
    # Doar cele mai puternice două contradicții, ca motivul să rămână lizibil.
    for _, a, va, b, vb in conflicts[:2]:
        reasons.append(f"{COMPONENT_NAMES[a]} ({va:+.0f}) contrazice {COMPONENT_NAMES[b]} ({vb:+.0f})")

    if odds is None or odds.samples < settings.min_samples:
        reasons.append("prea puține situații similare în istoric")
    else:
        # Estimarea onestă (trasă spre model sau spre rata de bază), dacă există; altfel procentul brut.
        p, base = (honest["p"], honest["base"]) if honest else (odds.probability_up, odds.base_rate)
        edge = p - base
        if abs(edge) < settings.min_edge:
            reasons.append(f"istoric fără avantaj: {p:.0%} șanse estimate față de {base:.0%} de obicei")
        elif (rule_decision == "BUY" and edge < 0) or (rule_decision == "SELL" and edge > 0):
            reasons.append(
                f"regulile spun {rule_decision}, dar istoric prețul a urcat în {p:.0%} din cazuri (de obicei {base:.0%})"
            )

    if model:
        edge = model["prob"] - model["base_rate"]
        if (rule_decision == "BUY" and edge < -settings.min_edge) or (rule_decision == "SELL" and edge > settings.min_edge):
            reasons.append(
                f"modelul statistic contrazice regulile: {model['prob']:.0%} șanse față de {model['base_rate']:.0%} de obicei"
            )
    return reasons


def _next_earnings(upcoming: Any) -> dict[str, Any] | None:
    """Raportul trimestrial următor, dacă e în următoarele ~2 luni."""
    if not isinstance(upcoming, date):
        return None
    days = (upcoming - date.today()).days
    return {"date": upcoming.isoformat(), "days": days} if 0 <= days <= 60 else None


def _model_context(m: dict[str, Any] | None) -> dict[str, Any] | None:
    if not m:
        return None
    return {
        "prob": round(m["prob"], 3),
        "interval_90": [round(m["lo"], 3), round(m["hi"], 3)] if m.get("lo") is not None else None,
        "base_rate": round(m["base_rate"], 3),
        "ranking_helped_significantly_out_of_sample": m.get("helps"),
        "trained_on": m["trained"],
    }


class Analyzer:
    def __init__(
        self,
        data: DataProvider,
        settings: Settings | None = None,
        advisor: ClaudeAdvisor | None = None,
        claude_mode: ClaudeMode = "auto",
        model: Any | None = None,
        beat_model: Any | None = None,
        claude_worse: bool = False,
        trade_model: Any | None = None,
        hold_models: dict[int, Any] | None = None,
    ) -> None:
        self.data = data
        self.settings = settings or Settings()
        self.advisor = advisor
        self.claude_mode = claude_mode
        self.model = model
        self.beat_model = beat_model
        self.trade_model = trade_model
        self.hold_models = hold_models or {}
        # True când, în predicțiile verificate, Claude a greșit mai des decât statistica (vezi track.claude_worse).
        self.claude_worse = claude_worse

    def _optional(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Surse opționale: dacă lipsesc sau dau eroare, analiza continuă fără ele."""
        fn = getattr(self.data, name, None)
        if not callable(fn):
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # o sursă opțională nu are voie să oprească analiza
            log.warning("%s indisponibil: %s", name, exc)
            return None

    def analyze(self, ticker: str) -> Recommendation:
        s = self.settings
        prices = self.data.prices(ticker, s.history_period)
        ind = compute_all(prices)
        tech = technical_score(ind)
        if pd.isna(tech.iloc[-1]):
            raise DataError(f"{ticker}: istoric prea scurt (sunt necesare cel puțin ~200 de zile)")
        today = prices.index[-1].date()

        fundamentals = self.data.fundamentals(ticker)
        news = self.data.news(ticker)
        market_px = self._optional("market", s.history_period)
        market_close = market_px["Close"] if market_px is not None else None
        quarters = self._optional("earnings", ticker)
        trades = self._optional("insiders", ticker, close=prices["Close"])
        macro_raw = self._optional("macro")
        upcoming = self._optional("next_earnings", ticker)
        trend = self._optional("revisions", ticker)

        f = fundamentals or {}
        quality = check_prices(prices, today=date.today(), fifty_two_week_high=f.get("fiftyTwoWeekHigh"),
                               two_hundred_day_average=f.get("twoHundredDayAverage"),
                               last_split=(f["lastSplitFactor"], f.get("lastSplitDate")) if f.get("lastSplitFactor") else None)
        extras = {
            "quality": quality.to_dict(),
            "market": market_signal(market_close),
            "macro": macro_signal(macro_raw),
            "earnings": earnings_signal(quarters, today),
            "insiders": insider_signal(trades, today),
            "next_earnings": _next_earnings(upcoming),
            "revisions": revision_signal(trend),
        }
        market_parts = [x["score"] for x in (extras["market"], extras["macro"]) if x]
        scores: dict[str, float | None] = {
            "technical": float(tech.iloc[-1]),
            "fundamental": fundamental_score(fundamentals),
            "sentiment": sentiment_score(news),
            "earnings": extras["earnings"]["score"] if extras["earnings"] else None,
            "insiders": extras["insiders"]["score"] if extras["insiders"] else None,
            "revisions": extras["revisions"]["score"] if extras["revisions"] else None,
            "market": float(np.mean(market_parts)) if market_parts else None,
        }
        scores["composite"] = composite_score(scores, s.weights)
        rule_decision = decision_from_score(scores["composite"], s.buy_threshold)
        odds = historical_odds(ind["close"], tech, scores["technical"], s.horizon_days, s.bucket_width)
        model_out = self._model_probability(self.model, prices, market_close, quarters)
        beat_out = self._model_probability(self.beat_model, prices, market_close, quarters)
        trade_out = self._model_probability(self.trade_model, prices, market_close, quarters)
        # Mișcarea tipică pe orizont (o abatere): ținta și stopul trade-ului și baza mărimii poziției.
        width = float(realized_vol(prices["Close"], 60).iloc[-1] * np.sqrt(s.horizon_days / 252))
        if trade_out:
            price = float(prices["Close"].iloc[-1])
            trade_out["take_profit"], trade_out["stop_loss"] = price * np.exp(width), price * np.exp(-width)
        # Modelul recalibrat pe anii nevăzuți are deja intervalul lui; altfel, istoricul acțiunii tras spre model.
        honest = model_estimate(model_out, odds) or honest_estimate(
            odds,
            model_out["prob"] if model_out else None,
            model_out["base_rate"] if model_out else None,
            total_samples=odds.total if odds else None,
        )

        rec = Recommendation(
            ticker=ticker.upper(),
            as_of=str(today),
            price=float(prices["Close"].iloc[-1]),
            scores=scores,
            rule_decision=rule_decision,
            odds=odds,
            extras={k: v for k, v in {**extras, "macro_raw": macro_raw}.items() if v},
            model=model_out,
            model_beat=beat_out,
            model_trade=trade_out,
            honest=honest,
        )
        nxt = extras.get("next_earnings")
        base = base_decision(honest, rule_decision, scores["composite"], s.min_edge, quality.grade,
                             nxt["days"] if nxt else None, trade_est=model_estimate(trade_out, None),
                             beat_est=model_estimate(beat_out, None), move=width if np.isfinite(width) else None)
        rec.final = base
        rec.ambiguity_reasons = list(base.ask)
        if base.confidence != "none":
            # Când statistica cere o poziție, componente care se contrazic puternic merită o a doua privire.
            rec.ambiguity_reasons += [r for r in find_ambiguity(scores, rule_decision, odds, s) if "contrazice" in r]
        if quality.grade == "slabă":
            # Pe date slabe poziția e zero oricum: nu are rost să-l întrebăm pe Claude.
            rec.ambiguity_reasons = []

        ask = self.claude_mode == "always" or (self.claude_mode == "auto" and rec.ambiguity_reasons)
        if ask and self.advisor:
            try:
                rec.claude = self.advisor.advise(self._claude_context(rec, ind, fundamentals, news))
            except AdvisorError as exc:
                log.warning("%s: %s", ticker, exc)
                rec.claude_error = str(exc)
        if rec.claude:
            rec.final = with_claude(base, rec.claude, honest, s.min_edge, quality.grade, self.claude_worse)
        if self.hold_models:
            per_h, helps = {}, {}
            for h, model in self.hold_models.items():
                out = self._model_probability(model, prices, market_close, quarters)
                per_h[h] = model_estimate(out, None) if out and out.get("lo") is not None else (
                    {"p": out["prob"], "base": out["base_rate"], "lo": None, "hi": None} if out else None)
                helps[h] = bool(out and out.get("helps"))
            daily = float(realized_vol(prices["Close"], 60).iloc[-1] / np.sqrt(252))
            plan = holding_plan(per_h, rec.decision, rec.price, daily if np.isfinite(daily) else None, helps, s.horizon_days)
            if plan:
                rec.extras["holding"] = plan
        return rec

    def _model_probability(
        self, model: Any | None, prices: pd.DataFrame, market_close: pd.Series | None, quarters: list[dict[str, Any]] | None
    ) -> dict[str, Any] | None:
        if model is None:
            return None
        from .model import FEATURES, feature_frame

        try:
            last = feature_frame(prices, market_close, quarters).iloc[[-1]]
            if last[FEATURES].isna().any(axis=1).iloc[0]:
                return None
            est = model.estimate(last)
        except Exception as exc:
            log.warning("Modelul nu a putut calcula probabilitatea: %s", exc)
            return None
        lo, hi = float(est["lo"][0]), float(est["hi"][0])
        cal = getattr(model, "calibration", None)
        return {
            "prob": float(est["p"][0]),
            "lo": None if np.isnan(lo) else lo,
            "hi": None if np.isnan(hi) else hi,
            "base_rate": float(est["base"]),
            "raw_prob": float(est["raw"][0]),
            "helps": cal.helps if cal else None,
            "target": getattr(model, "target", "up"),
            "horizon": model.horizon,
            "trained": model.info,
        }

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
            "evidence_based_decision": (
                {"action": rec.final.action, "conviction": rec.final.confidence, "position_size": rec.final.size,
                 "why": rec.final.why}
                if rec.final else None
            ),
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
            "statistical_estimate": (
                {
                    "source": (
                        "model învățat pe toată lista, verificat doar pe ani nevăzuți și recalibrat"
                        if rec.honest.get("source") == "model"
                        else "istoricul acțiunii, tras spre modelul sau rata de bază a listei"
                    ),
                    "prob_up": round(rec.honest["p"], 3),
                    "interval_90": [round(rec.honest["lo"], 3), round(rec.honest["hi"], 3)],
                    "base_rate": round(rec.honest["base"], 3),
                    "this_stock_only": None if rec.honest["stock_p"] is None else round(rec.honest["stock_p"], 3),
                    "this_stock_independent_cases": round(rec.honest["stock_independent_cases"]),
                }
                if rec.honest
                else None
            ),
            "statistical_model": _model_context(rec.model),
            "statistical_model_beat_sp500": _model_context(rec.model_beat),
            "statistical_model_trade_target_before_stop": (
                {**_model_context(rec.model_trade), "take_profit": round(rec.model_trade["take_profit"], 2),
                 "stop_loss": round(rec.model_trade["stop_loss"], 2)} if rec.model_trade else None
            ),
            "analyst_eps_revisions": rec.extras.get("revisions"),
            "next_earnings": rec.extras.get("next_earnings"),
            "data_quality": {k: v for k, v in (rec.extras.get("quality") or {}).items() if k != "checks"} or None,
            "market_trend_sp500": rec.extras.get("market"),
            "macro": rec.extras.get("macro"),
            "earnings": rec.extras.get("earnings"),
            "insider_trades_180d": rec.extras.get("insiders"),
            "fundamentals": fundamentals,
            "news": news,
        }
