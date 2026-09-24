from stockai.__main__ import render
from stockai.advisor import ClaudeVerdict
from stockai.analyzer import Recommendation
from stockai.calibration import HistoricalOdds


def test_render_shows_claude_opinion_and_history():
    rec = Recommendation(
        ticker="AAPL",
        as_of="2026-09-23",
        price=227.35,
        scores={"technical": 12.0, "fundamental": None, "sentiment": -40.0, "composite": -3.2},
        rule_decision="HOLD",
        odds=HistoricalOdds(0.58, 0.56, 0.012, 240, 20),
        ambiguity_reasons=["scor compus neutru (-3, prag ±25)"],
        claude=ClaudeVerdict(
            decision="SELL", probability_up_pct=41, confidence="medium", reasoning="Știri slabe.", key_risks=["volatilitate"]
        ),
    )
    out = render(rec)
    assert "Decizie: SELL  (decis de Claude, încredere medie)" in out
    assert "fundamental n/a" in out and "compus -3" in out
    assert "a urcat în 58%" in out and "n=240" in out
    assert "Claude: 41% șanse de creștere" in out
    assert "Riscuri: volatilitate" in out
