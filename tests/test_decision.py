from types import SimpleNamespace

import pytest

from stockai.decision import base_decision, stat_verdict, with_claude

EDGE = 0.03


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
    (SURE_UP, "HOLD", "BUY", "medium", False),   # regulile neutre nu blochează
    (SURE_UP, "SELL", "HOLD", "low", True),      # conflict: întrebăm pe Claude
    (SURE_DOWN, "SELL", "SELL", "high", False),
    (SURE_DOWN, "BUY", "HOLD", "low", True),
    (LEAN_UP, "BUY", "HOLD", "low", True),       # regulile singure nu pot crea un BUY
    (LEAN_UP, "HOLD", "HOLD", "low", False),
    (NOTHING, "HOLD", "HOLD", "low", False),     # fără avantaj și reguli neutre: HOLD clar, fără Claude
])
def test_base_decision_table(e, rule, action, conf, asks):
    d = base_decision(e, rule, 30.0 if rule == "BUY" else -30.0 if rule == "SELL" else 0.0, EDGE)
    assert (d.action, d.confidence, bool(d.ask)) == (action, conf, asks)
    assert d.why


def test_strong_rules_without_statistical_edge_ask_claude_but_do_not_buy():
    d = base_decision(NOTHING, "BUY", 55.0, EDGE)
    assert d.action == "HOLD" and any("semnal puternic" in a for a in d.ask)


def test_gates_lower_confidence_or_block():
    assert base_decision(SURE_UP, "BUY", 40, EDGE, quality_grade="slabă").action == "HOLD"
    before_report = base_decision(SURE_UP, "BUY", 40, EDGE, earnings_in_days=10)
    assert before_report.action == "BUY" and before_report.confidence == "medium"
    assert any("raport trimestrial" in w for w in before_report.why)
    history_only = base_decision(est(0.61, 0.58, 0.64, source="history"), "BUY", 40, EDGE)
    assert history_only.confidence == "medium"
    assert base_decision(SURE_UP, "BUY", 40, EDGE, earnings_in_days=45).confidence == "high"


def test_claude_must_match_its_own_probability_and_cannot_flip_sure_statistics():
    base = base_decision(LEAN_UP, "BUY", 30, EDGE)
    assert with_claude(base, verdict("BUY", 62), LEAN_UP, EDGE).action == "BUY"
    # BUY cu 56% când rata obișnuită e 55%: nu e un avantaj
    weak = with_claude(base, verdict("BUY", 56), LEAN_UP, EDGE)
    assert weak.action == "HOLD" and weak.decided_by == "Claude"
    conflict = base_decision(SURE_UP, "SELL", -30, EDGE)
    assert with_claude(conflict, verdict("SELL", 40), SURE_UP, EDGE).action == "HOLD"
    assert with_claude(conflict, verdict("BUY", 62), SURE_UP, EDGE).action == "BUY"


def test_claude_far_outside_the_interval_gets_low_confidence_and_poor_data_stays_hold():
    base = base_decision(LEAN_UP, "BUY", 30, EDGE)
    far = with_claude(base, verdict("BUY", 80, "high"), LEAN_UP, EDGE)
    assert far.action == "BUY" and far.confidence == "low"
    assert with_claude(base, verdict("BUY", 62), LEAN_UP, EDGE, quality_grade="slabă").action == "HOLD"


def test_statistics_decide_when_claude_has_a_worse_track_record():
    base = base_decision(LEAN_UP, "BUY", 30, EDGE)
    d = with_claude(base, verdict("BUY", 62), LEAN_UP, EDGE, claude_worse=True)
    assert d.action == "HOLD" and d.decided_by == "statistică"


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
