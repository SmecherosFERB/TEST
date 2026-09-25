from types import SimpleNamespace

import pytest

from stockai.decision import base_decision, position_size, stat_verdict, with_claude

EDGE = 0.03
MOVE = 0.08  # o abatere tipică pe 4 săptămâni


def est(p, lo, hi, base=0.55, source="model"):
    return {"p": p, "lo": lo, "hi": hi, "base": base, "edge": p - base,
            "sure": "up" if lo > base else "down" if hi < base else None, "source": source}


SURE_UP = est(0.61, 0.58, 0.64)          # tot intervalul peste 55%
SURE_DOWN = est(0.48, 0.45, 0.51)
LEAN_UP = est(0.59, 0.53, 0.65)          # avantaj, dar intervalul include media
NOTHING = est(0.555, 0.50, 0.61)


def verdict(decision, pct, confidence="medium"):
    return SimpleNamespace(decision=decision, probability_up_pct=pct, confidence=confidence)


def test_stat_verdicts():
    assert [stat_verdict(e, EDGE) for e in (SURE_UP, SURE_DOWN, LEAN_UP, NOTHING, None)] == \
        ["up", "down", "lean_up", "none", "none"]


@pytest.mark.parametrize("e, rule, action, conf, asks", [
    (SURE_UP, "BUY", "BUY", "high", False),      # statistica sigură + regulile de acord
    (SURE_UP, "HOLD", "BUY", "medium", False),   # regulile neutre nu schimbă direcția
    (SURE_UP, "SELL", "BUY", "low", True),       # conflict: direcția statisticii, convingere scăzută, Claude
    (SURE_DOWN, "SELL", "SELL", "high", False),
    (SURE_DOWN, "BUY", "SELL", "low", True),
    (LEAN_UP, "BUY", "BUY", "low", True),        # avantaj nesigur: poziție mică
    (LEAN_UP, "HOLD", "BUY", "low", False),
    (NOTHING, "HOLD", "BUY", "none", False),     # fără avantaj: direcția, dar fără poziție
])
def test_base_decision_is_always_buy_or_sell(e, rule, action, conf, asks):
    d = base_decision(e, rule, 30.0 if rule == "BUY" else -30.0 if rule == "SELL" else 0.0, EDGE, move=MOVE)
    assert (d.action, d.confidence, bool(d.ask)) == (action, conf, asks)
    assert d.action in ("BUY", "SELL") and d.why


def test_position_size_follows_risk_and_conviction():
    assert position_size("high", 0.08) == 0.1  # 1% / 8% = 12,5%, limitat la 10%
    assert position_size("medium", 0.08) == 0.075
    assert position_size("low", 0.08) == 0.0313
    assert position_size("none", 0.08) == 0.0
    assert position_size("high", 0.2) == 0.05  # acțiune mai volatilă → poziție mai mică
    assert position_size("high", None) is None
    assert base_decision(SURE_UP, "BUY", 40, EDGE, move=MOVE).size == 0.1
    assert base_decision(NOTHING, "HOLD", 0, EDGE, move=MOVE).size == 0.0


def test_without_an_edge_the_direction_comes_from_the_next_best_evidence():
    beat_down = est(0.44, 0.40, 0.48, base=0.50)
    flat = est(0.55, 0.50, 0.60)
    d = base_decision(flat, "HOLD", 0, EDGE, beat_est=beat_down, move=MOVE)
    assert d.action == "SELL" and d.confidence == "none" and any("bate S&P 500" in w for w in d.why)
    assert base_decision(flat, "SELL", -10, EDGE, move=MOVE).action == "SELL"  # la final, regulile
    strong = base_decision(NOTHING, "BUY", 55.0, EDGE, move=MOVE)
    assert strong.action == "BUY" and strong.confidence == "none" and any("semnal puternic" in a for a in strong.ask)


def test_gates_lower_conviction():
    poor = base_decision(SURE_UP, "BUY", 40, EDGE, quality_grade="slabă", move=MOVE)
    assert poor.action == "BUY" and poor.confidence == "none" and poor.size == 0.0
    before_report = base_decision(SURE_UP, "BUY", 40, EDGE, earnings_in_days=10, move=MOVE)
    assert before_report.confidence == "medium" and any("raport trimestrial" in w for w in before_report.why)
    assert base_decision(est(0.61, 0.58, 0.64, source="history"), "BUY", 40, EDGE).confidence == "medium"
    assert base_decision(SURE_UP, "BUY", 40, EDGE, earnings_in_days=45).confidence == "high"


def test_trade_model_confirms_or_weakens_the_direction():
    trade_up = est(0.58, 0.55, 0.61, base=0.50)
    trade_down = est(0.42, 0.39, 0.45, base=0.50)
    confirmed = base_decision(SURE_UP, "HOLD", 0, EDGE, trade_est=trade_up)
    assert confirmed.action == "BUY" and confirmed.confidence == "high"
    weakened = base_decision(SURE_UP, "BUY", 40, EDGE, trade_est=trade_down)
    assert weakened.action == "BUY" and weakened.confidence == "medium" and any("contrazice" in w for w in weakened.why)


def test_claude_is_checked_against_its_own_probability_and_sure_statistics():
    base = base_decision(LEAN_UP, "BUY", 30, EDGE, move=MOVE)
    ok = with_claude(base, verdict("BUY", 62), LEAN_UP, EDGE)
    assert ok.action == "BUY" and ok.decided_by == "Claude" and ok.size == 0.075
    # 56% când rata obișnuită e 55%: Claude nu are un avantaj → rămâne statistica
    weak = with_claude(base, verdict("BUY", 56), LEAN_UP, EDGE)
    assert weak.decided_by == "statistică" and weak.action == "BUY" and weak.confidence == "low"
    # direcția contrazice propria probabilitate → contează probabilitatea
    assert with_claude(base, verdict("BUY", 45), LEAN_UP, EDGE).action == "SELL"
    conflict = base_decision(SURE_UP, "SELL", -30, EDGE, move=MOVE)
    flipped = with_claude(conflict, verdict("SELL", 40), SURE_UP, EDGE)
    assert flipped.action == "BUY" and flipped.confidence == "low"


def test_claude_far_outside_the_interval_and_poor_data():
    base = base_decision(LEAN_UP, "BUY", 30, EDGE, move=MOVE)
    far = with_claude(base, verdict("BUY", 80, "high"), LEAN_UP, EDGE)
    assert far.action == "BUY" and far.confidence == "low"
    assert with_claude(base, verdict("BUY", 62), LEAN_UP, EDGE, quality_grade="slabă").size == 0.0


def test_statistics_decide_when_claude_has_a_worse_track_record():
    base = base_decision(LEAN_UP, "BUY", 30, EDGE, move=MOVE)
    d = with_claude(base, verdict("SELL", 40), LEAN_UP, EDGE, claude_worse=True)
    assert d.action == "BUY" and d.decided_by == "statistică"


def test_track_record_gate(tmp_path):
    from stockai import track

    path = tmp_path / "p.csv"
    rows = []
    for k in range(40):
        up = k % 2
        rows.append({"id": f"S{k}|stat", "ticker": f"S{k}", "made_on": "2026-01-05", "prob": 0.6 if up else 0.5,
                     "base": 0.55, "source": "stat", "resolved_on": "2026-02-05", "up": up, "ret": 0.0})
        rows.append({"id": f"S{k}|claude", "ticker": f"S{k}", "made_on": "2026-01-05", "prob": 0.4 if up else 0.7,
                     "base": 0.55, "source": "claude", "resolved_on": "2026-02-05", "up": up, "ret": 0.0})
    track._write(path, rows)
    assert track.claude_worse(path) is True
    assert track.claude_worse(path, min_n=100) is False
