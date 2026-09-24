import pandas as pd
import pytest

from stockai.indicators import compute_all
from stockai.scoring import (
    composite_score,
    decision_from_score,
    fundamental_score,
    sentiment_score,
    technical_components,
    technical_score,
)

from .conftest import make_prices


def test_uptrend_has_positive_trend_component():
    ind = compute_all(make_prices(drift=0.003, vol=0.005))
    assert technical_components(ind)["trend"].iloc[-1] == 1.0


def test_downtrend_has_negative_trend_component():
    ind = compute_all(make_prices(drift=-0.003, vol=0.005))
    assert technical_components(ind)["trend"].iloc[-1] == -1.0


def test_technical_score_bounded_and_nan_during_warmup(prices):
    score = technical_score(compute_all(prices))
    assert score.iloc[:199].isna().all()
    assert score.dropna().between(-100, 100).all()


def test_fundamental_score_positive_for_healthy_company():
    info = {
        "trailingPE": 30,
        "forwardPE": 24,
        "revenueGrowth": 0.2,
        "profitMargins": 0.25,
        "debtToEquity": 40,
        "recommendationMean": 1.8,
    }
    assert fundamental_score(info) > 50


def test_fundamental_score_negative_for_weak_company():
    info = {"revenueGrowth": -0.1, "profitMargins": -0.05, "debtToEquity": 300, "recommendationMean": 4}
    assert fundamental_score(info) < -50


def test_fundamental_score_none_without_data():
    assert fundamental_score({}) is None


def test_sentiment_weighted_by_relevance():
    news = [
        {"sentiment": 0.35, "relevance": 0.9},
        {"sentiment": -0.35, "relevance": 0.1},
        {"title": "fără scor", "sentiment": None},
    ]
    assert sentiment_score(news) == pytest.approx(80.0)


def test_sentiment_none_without_scored_news():
    assert sentiment_score([{"title": "x", "sentiment": None}]) is None


def test_composite_renormalizes_missing_components():
    weights = {"technical": 0.5, "fundamental": 0.25, "sentiment": 0.25}
    assert composite_score({"technical": 40, "fundamental": None, "sentiment": 80}, weights) == pytest.approx(
        (0.5 * 40 + 0.25 * 80) / 0.75
    )
    assert composite_score({"technical": None, "fundamental": None, "sentiment": None}, weights) == 0.0


@pytest.mark.parametrize("score,expected", [(30, "BUY"), (25, "BUY"), (0, "HOLD"), (-24.9, "HOLD"), (-25, "SELL")])
def test_decision_thresholds(score, expected):
    assert decision_from_score(score, 25) == expected
