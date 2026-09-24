from types import SimpleNamespace

import pytest

from stockai.advisor import AdvisorError, ClaudeVerdict
from stockai.analyzer import Analyzer, find_ambiguity
from stockai.calibration import HistoricalOdds
from stockai.config import Settings
from stockai.data import DataError

from .conftest import FakeData, make_prices


class FakeAdvisor:
    def __init__(self, error: Exception | None = None):
        self.contexts = []
        self.error = error

    def advise(self, context):
        self.contexts.append(context)
        if self.error:
            raise self.error
        return ClaudeVerdict(
            decision="BUY", probability_up_pct=62, confidence="medium", reasoning="motiv", key_risks=[]
        )


def odds(prob: float, base: float = 0.55, samples: int = 200) -> HistoricalOdds:
    return HistoricalOdds(probability_up=prob, base_rate=base, avg_return=0.01, samples=samples, horizon=20)


def scores(technical=50.0, fundamental=40.0, sentiment=None, composite=45.0):
    return {"technical": technical, "fundamental": fundamental, "sentiment": sentiment, "composite": composite}


def test_clear_signal_has_no_ambiguity():
    assert find_ambiguity(scores(), "BUY", odds(0.65), Settings()) == []


def test_neutral_score_is_ambiguous():
    reasons = find_ambiguity(scores(composite=10), "HOLD", odds(0.65), Settings())
    assert any("neutru" in r for r in reasons)


def test_conflicting_components_are_ambiguous():
    reasons = find_ambiguity(scores(technical=60, sentiment=-50), "BUY", odds(0.65), Settings())
    assert any("contrazice" in r for r in reasons)


def test_weak_or_contrary_history_is_ambiguous():
    assert any("fără avantaj" in r for r in find_ambiguity(scores(), "BUY", odds(0.56), Settings()))
    assert any("dar istoric" in r for r in find_ambiguity(scores(), "BUY", odds(0.45), Settings()))
    assert any("prea puține" in r for r in find_ambiguity(scores(), "BUY", odds(0.7, samples=10), Settings()))
    assert any("prea puține" in r for r in find_ambiguity(scores(), "BUY", None, Settings()))


def test_analyze_never_calls_claude_in_never_mode(prices):
    advisor = FakeAdvisor()
    rec = Analyzer(FakeData(prices), advisor=advisor, claude_mode="never").analyze("aapl")
    assert advisor.contexts == []
    assert rec.ticker == "AAPL" and rec.decided_by == "reguli"
    assert rec.decision == rec.rule_decision
    assert set(rec.scores) == {"technical", "fundamental", "sentiment", "composite"}


def test_analyze_always_mode_uses_claude_decision(prices):
    advisor = FakeAdvisor()
    news = [{"title": "t", "sentiment": 0.1, "relevance": 0.5}]
    rec = Analyzer(FakeData(prices, {"revenueGrowth": 0.1}, news), advisor=advisor, claude_mode="always").analyze("MSFT")
    assert rec.decision == "BUY" and rec.decided_by == "Claude"
    ctx = advisor.contexts[0]
    assert ctx["ticker"] == "MSFT" and ctx["news"] == news
    assert ctx["ambiguity_reasons"]
    assert ctx["historical_odds_technical_only"]["overlapping_samples"] > 0
    assert rec.to_dict()["claude"]["probability_up_pct"] == 62


def test_analyze_auto_mode_asks_only_when_ambiguous(prices, monkeypatch):
    advisor = FakeAdvisor()
    analyzer = Analyzer(FakeData(prices), advisor=advisor, claude_mode="auto")

    monkeypatch.setattr("stockai.analyzer.find_ambiguity", lambda *a: [])
    assert analyzer.analyze("X").claude is None
    monkeypatch.setattr("stockai.analyzer.find_ambiguity", lambda *a: ["neclar"])
    assert analyzer.analyze("X").claude is not None
    assert len(advisor.contexts) == 1


def test_advisor_failure_falls_back_to_rules(prices):
    advisor = FakeAdvisor(error=AdvisorError("Claude a refuzat cererea"))
    rec = Analyzer(FakeData(prices), advisor=advisor, claude_mode="always").analyze("X")
    assert rec.decision == rec.rule_decision
    assert rec.claude_error == "Claude a refuzat cererea"


def test_short_history_raises():
    with pytest.raises(DataError, match="istoric prea scurt"):
        Analyzer(FakeData(make_prices(n=150))).analyze("NEW")
