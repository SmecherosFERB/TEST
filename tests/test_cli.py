from stockai.__main__ import render
from stockai.advisor import ClaudeVerdict
from stockai.analyzer import Recommendation
from stockai.calibration import HistoricalOdds


def test_render_shows_claude_opinion_and_history():
    rec = Recommendation(
        ticker="AAPL",
        as_of="2026-09-23",
        price=227.35,
        scores={"technical": 12.0, "fundamental": None, "sentiment": -40.0, "earnings": 35.0, "insiders": 0.0,
                "market": 60.0, "composite": -3.2},
        rule_decision="HOLD",
        odds=HistoricalOdds(0.58, 0.56, 0.012, 240, 20),
        ambiguity_reasons=["scor compus neutru (-3, prag ±25)"],
        extras={
            "market": {"score": 60.0, "label": "în creștere", "return_1m": 0.021},
            "earnings": {"score": 35.0, "last_reported": "2026-07-30", "days_since": 56, "last_surprise_pct": 7.4,
                         "beats_last4": "4/4"},
            "insiders": {"score": 0.0, "days": 180, "buys": 0, "buyers": 0, "buy_value": 0, "sells": 12,
                         "sellers": 2, "sell_value": 5300000},
        },
        model={"prob": 0.571, "base_rate": 0.55, "horizon": 20, "trained": {"tickers": 113}},
        claude=ClaudeVerdict(
            decision="SELL", probability_up_pct=41, confidence="medium", reasoning="Știri slabe.", key_risks=["volatilitate"]
        ),
    )
    out = render(rec)
    assert "Decizie: SELL  (decis de Claude, încredere medie)" in out
    assert "fundamental n/a" in out and "compus -3" in out and "rezultate +35" in out
    assert "Model statistic: 57% șanse de creștere în 20 zile (de obicei 55%; antrenat pe 113 acțiuni)" in out
    assert "Piața (S&P 500): în creștere, +2.1% în ultima lună" in out
    assert "surpriză +7.4%" in out and "4/4" in out
    assert "12 vânzări de la 2 persoane ($5,300,000)" in out
    assert "a urcat în 58%" in out and "n=240" in out
    assert "Claude: 41% șanse de creștere" in out
    assert "Riscuri: volatilitate" in out
