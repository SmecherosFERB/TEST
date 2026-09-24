import pandas as pd
import pytest

from stockai.calibration import historical_odds


def test_odds_reflect_what_followed_similar_scores():
    # Scor mare (80) → prețul urcă a doua zi; scor mic (-80) → scade.
    close = pd.Series([100, 101, 100, 101, 100, 101, 100, 101, 100, 101], dtype=float)
    score = pd.Series([80, -80] * 5, dtype=float)
    odds = historical_odds(close, score, current_score=75, horizon=1, bucket_width=10)
    assert odds.probability_up == 1.0
    assert odds.base_rate == pytest.approx(5 / 9)
    assert odds.samples == 5  # ultima zi nu are încă rezultat
    assert odds.edge == pytest.approx(1 - 5 / 9)


def test_odds_none_without_similar_history():
    close = pd.Series([100.0, 101.0, 102.0])
    score = pd.Series([10.0, 10.0, 10.0])
    assert historical_odds(close, score, current_score=90, horizon=1, bucket_width=5) is None
