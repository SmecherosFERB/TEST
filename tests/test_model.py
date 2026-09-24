import numpy as np
import pandas as pd
import pytest

from stockai.model import FEATURES, ProbabilityModel, build_dataset, design_effect, feature_frame, walk_forward

from .conftest import make_prices


def momentum_world(n_stocks=12, n_days=2600, seed=3):
    """Prețuri sintetice în care trendul persistă: randamentul de mâine depinde de cel din ultima lună."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-23", periods=n_days)
    out = {}
    for k in range(n_stocks):
        r = np.zeros(n_days)
        eps = rng.normal(0.0002, 0.012, n_days)
        for t in range(1, n_days):
            # jumătate din media ultimelor 21 de zile continuă: proces stabil, cu trend persistent
            r[t] = 0.5 * r[max(0, t - 21):t].mean() + eps[t]
        close = 50 * np.exp(np.cumsum(r))
        vol = rng.integers(1_000_000, 2_000_000, n_days).astype(float)
        out[f"S{k}"] = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": vol}, index=idx)
    market = make_prices(n=n_days, drift=0.0004, vol=0.01, seed=99)["Close"]
    market.index = idx
    return out, market


def test_feature_frame_columns_and_clipping():
    px = make_prices()
    f = feature_frame(px, px["Close"], None)
    assert list(f.columns) == FEATURES
    last = f.iloc[-1]
    assert last.notna().all()
    assert -0.8 <= last["high_52w"] <= 0 and np.log(0.05) <= last["vol_60d"] <= np.log(2)
    assert f["mom_bear"].dropna().abs().le(f["mom_12_1"].dropna().abs() + 1e-12).all()


def test_build_dataset_labels_and_step():
    prices, market = momentum_world(n_stocks=2, n_days=900)
    data = build_dataset(prices, market, horizon=20, step=5)
    assert set(data["ticker"]) == {"S0", "S1"}
    assert data["label"].isin([0.0, 1.0]).all()
    assert data[FEATURES].notna().all().all()
    # ultimele 21 de zile nu au încă rezultat (intrare a doua zi + 20 de zile), deci nu intră în setul de date
    assert data["date"].max() == prices["S0"].index[-22]


def test_walk_forward_finds_real_signal_and_model_roundtrip(tmp_path):
    prices, market = momentum_world()
    data = build_dataset(prices, market, horizon=20, step=5)
    report = walk_forward(data, horizon=20, min_train_years=3)
    assert report.n > 1000 and report.years
    assert report.skill > 0 and report.auc > 0.55
    assert report.top_decile_up > report.overall_up
    assert len(report.calibration) == 10 and len(report.groups) == 5
    assert report.groups[-1]["actual"] > report.groups[0]["actual"]
    assert report.recalibration.helps and report.n_independent <= report.n
    text = report.render()
    assert "walk-forward" in text and "AUC" in text and "a contat sigur statistic" in text

    model = ProbabilityModel(20).fit(data)
    model.calibration = report.recalibration
    path = tmp_path / "m.pkl"
    model.save(path)
    loaded = ProbabilityModel.load(path)
    p = loaded.predict(data.iloc[:5])
    assert ((p > 0) & (p < 1)).all()
    est = loaded.estimate(data.iloc[:50])
    assert ((est["lo"] <= est["p"]) & (est["p"] <= est["hi"])).all()
    # un model care a ajutat în test dă probabilități diferite de la o situație la alta
    assert est["p"].std() > 0.01
    top = next(iter(loaded.coefficients()))
    assert top in ("ret_1m", "tech", "mom_12_1", "dist_sma200", "high_52w")


def test_walk_forward_on_pure_noise_does_not_beat_base_rate_by_much():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(end="2026-09-23", periods=2600)
    prices = {}
    for k in range(8):
        close = 40 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx))))
        prices[f"N{k}"] = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                                         "Volume": np.full(len(idx), 1e6)}, index=idx)
    data = build_dataset(prices, None, horizon=20, step=5)
    report = walk_forward(data, horizon=20)
    # pe zgomot, un test corect nu trebuie să „descopere” un avantaj mare
    assert report.skill < 0.02
    assert not report.recalibration.helps
    # iar probabilitățile recalibrate rămân lângă medie, cu intervalul incluzând media
    model = ProbabilityModel(20).fit(data)
    model.calibration = report.recalibration
    est = model.estimate(data.iloc[-200:])
    base = est["base"]
    assert np.abs(est["p"] - base).max() < 0.03
    assert ((est["lo"] <= base) & (base <= est["hi"])).mean() > 0.95


def test_beat_target_labels_compare_with_market():
    prices, market = momentum_world(n_stocks=2, n_days=900)
    data = build_dataset(prices, market, horizon=20, target="beat")
    close, m = prices["S0"]["Close"], market.reindex(prices["S0"].index, method="ffill")
    row = data[data["ticker"] == "S0"].iloc[0]
    # intrarea e la închiderea de a doua zi, ieșirea după încă 20 de zile
    for _, row in data[data["ticker"] == "S0"].iterrows():
        i = close.index.get_loc(row["date"])
        stock_ret = close.iloc[i + 21] / close.iloc[i + 1] - 1
        market_ret = m.iloc[i + 21] / m.iloc[i + 1] - 1
        assert row["label"] == float(stock_ret > market_ret)
    # implicit, ferestrele de 20 de zile nu se suprapun
    dates = pd.DatetimeIndex(data[data["ticker"] == "S0"]["date"])
    assert (np.diff(close.index.get_indexer(dates)) == 20).all()
    with pytest.raises(ValueError, match="S&P 500"):
        build_dataset(prices, None, target="beat")


def test_design_effect_counts_months_that_move_together():
    rng = np.random.default_rng(1)
    months = np.repeat(np.arange(60), 40).astype(str)
    independent = rng.normal(0, 0.5, len(months))
    together = independent + np.repeat(rng.normal(0, 0.5, 60), 40)
    assert design_effect(independent, months) < 1.5
    assert design_effect(together, months) > 10


def test_old_model_file_is_rejected(tmp_path):
    import pickle

    prices, market = momentum_world(n_stocks=2, n_days=900)
    model = ProbabilityModel(20).fit(build_dataset(prices, market, horizon=20))
    model.features = ["tech", "rsi"]
    path = tmp_path / "old.pkl"
    path.write_bytes(pickle.dumps(model))
    with pytest.raises(TypeError, match="versiune mai veche"):
        ProbabilityModel.load(path)


def test_beta_feature_recovers_true_beta_and_share_classes_count_once():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range(end="2026-09-23", periods=1200)
    mkt = rng.normal(0.0003, 0.01, len(idx))
    market = pd.Series(100 * np.exp(np.cumsum(mkt)), index=idx)
    prices = {}
    for name, beta in (("LOW", 0.5), ("HIGH", 1.8)):
        close = 50 * np.exp(np.cumsum(beta * mkt + rng.normal(0, 0.012, len(idx))))
        prices[name] = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                                     "Volume": np.full(len(idx), 1e6)}, index=idx)
    assert abs(feature_frame(prices["LOW"], market)["beta_1y"].iloc[-1] - 0.5) < 0.15
    assert abs(feature_frame(prices["HIGH"], market)["beta_1y"].iloc[-1] - 1.8) < 0.15

    prices["GOOGL"], prices["GOOG"] = prices["LOW"], prices["LOW"] * 1.001
    data = build_dataset(prices, market, horizon=20)
    assert "GOOG" not in set(data["ticker"]) and "GOOGL" in set(data["ticker"])
