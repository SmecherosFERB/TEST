from datetime import date

import pandas as pd
import pytest

from stockai import track
from stockai.calibration import HistoricalOdds, honest_estimate, wilson_interval


def test_wilson_interval_narrows_with_more_cases():
    lo_small, hi_small = wilson_interval(0.6, 20)
    lo_big, hi_big = wilson_interval(0.6, 2000)
    assert lo_small < lo_big < 0.6 < hi_big < hi_small
    assert wilson_interval(0.5, 0) == (0.0, 1.0)


def test_honest_estimate_shrinks_few_cases_toward_prior():
    # 68% din 91 de zile care se suprapun (orizont 4) = doar ~23 de cazuri independente.
    odds = HistoricalOdds(probability_up=0.68, base_rate=0.65, avg_return=0.04, samples=91, horizon=4, total=520)
    h = honest_estimate(odds)
    assert h["stock_independent_cases"] == pytest.approx(22.75)
    assert 0.65 < h["p"] < 0.68  # tras spre rata de bază
    assert h["lo"] < h["base"] < h["hi"] and h["sure"] is None

    # Cu un model antrenat pe toată lista (rata de bază comună 55%), rata „de obicei” coboară spre realitate.
    h2 = honest_estimate(odds, prior_prob=0.56, prior_base=0.55, total_samples=520)
    assert h2["base"] < 0.65 and h2["p"] < h["p"]


def test_honest_estimate_can_be_sure_with_many_cases():
    odds = HistoricalOdds(probability_up=0.75, base_rate=0.55, avg_return=0.03, samples=8000, horizon=20, total=20000)
    h = honest_estimate(odds)
    assert h["sure"] == "up" and h["lo"] > h["base"]


def test_prediction_log_resolve_and_report(tmp_path):
    path = tmp_path / "pred.csv"
    assert track.log_prediction("AAPL", date(2026, 8, 3), 0.62, 0.56, "stat", path)
    assert not track.log_prediction("AAPL", date(2026, 8, 5), 0.60, 0.56, "stat", path)  # aceeași săptămână
    assert track.log_prediction("AAPL", date(2026, 8, 5), 0.45, 0.56, "claude", path)
    assert track.log_prediction("MSFT", date(2026, 9, 20), 0.55, 0.56, "stat", path)  # încă nu e scadentă

    idx = pd.bdate_range("2026-07-01", "2026-09-23")
    close = pd.Series(range(100, 100 + len(idx)), index=idx, dtype=float)  # prețul urcă tot timpul
    done = track.resolve(lambda t: pd.DataFrame({"Close": close}), path, today=date(2026, 9, 24))
    assert done == 2

    text = track.report(path)
    assert "Statistica programului: 1 predicții verificate" in text
    assert "Claude: 1 predicții verificate" in text
    assert "Așteaptă verificarea: 1" in text
