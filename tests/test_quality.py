from datetime import date, datetime

import numpy as np
import pandas as pd

from stockai.model import build_dataset
from stockai.quality import NEW_YORK, check_prices, drop_partial_bar, excluded_rows

from .conftest import make_prices

TODAY = date(2026, 9, 24)  # make_prices se termină pe 2026-09-23


def failed(q):
    return [c.label for c in q.failed]


def test_clean_series_passes_every_check():
    q = check_prices(make_prices(n=2600), today=TODAY)
    assert q.grade == "bună" and q.score == 100 and not q.failed and q.tradable
    assert q.adv > 1e8


def test_unadjusted_split_is_flagged_and_its_windows_leave_the_dataset():
    px = make_prices(n=2600)
    split = px.index[-400]
    px.loc[px.index < split, ["Open", "High", "Low", "Close"]] *= 4  # 4:1 neajustat
    q = check_prices(px, today=TODAY)
    assert q.suspect == [split] and "Fără split-uri neajustate" in failed(q) and q.grade == "acceptabilă"

    bad = excluded_rows(px.index, q.suspect, lookback=252, horizon=20)
    k = px.index.get_loc(split)
    assert bad[k - 21] and bad[k + 251] and not bad[k - 22] and not bad[k + 252]
    data = build_dataset({"X": px, "Y": make_prices(n=2600, seed=3)}, None, horizon=20)
    x_dates = pd.DatetimeIndex(data[data["ticker"] == "X"]["date"])
    assert not ((x_dates >= px.index[k - 21]) & (x_dates <= px.index[k + 251])).any()


def test_real_crash_is_not_mistaken_for_a_split():
    px = make_prices(n=2600)
    px.loc[px.index >= px.index[-200], "Close"] *= 0.45  # −55% într-o zi: nu e un raport de split
    q = check_prices(px, today=TODAY)
    assert not q.suspect and q.grade == "bună"
    assert any(c.label == "Mișcări extreme" for c in q.checks)


def test_stale_gaps_frozen_quotes_and_missing_volume():
    px = make_prices(n=2600)
    assert "Date la zi" in failed(check_prices(px, today=date(2026, 10, 9)))

    gappy = px.drop(px.index[-120:-110])
    assert "Fără zile lipsă" in failed(check_prices(gappy, today=TODAY))

    frozen = px.copy()
    frozen.loc[frozen.index[-60:-52], "Close"] = frozen["Close"].iloc[-61]
    assert "Prețuri care se mișcă" in failed(check_prices(frozen, today=TODAY))

    novol = px.copy()
    novol.loc[novol.index[-80::2], "Volume"] = 0
    assert "Volum raportat" in failed(check_prices(novol, today=TODAY))


def test_illiquid_or_cheap_stocks_are_not_tradable_and_poor_series_skip_training():
    px = make_prices(n=2600)
    thin = px.assign(Volume=1000.0)
    q = check_prices(thin, today=TODAY)
    assert not q.liquid and not q.tradable and q.grade == "bună"

    broken = px.copy()
    broken.loc[broken.index < broken.index[-300], "Close"] *= 10
    broken = broken.drop(broken.index[-200:-180]).assign(Volume=0.0)
    poor = check_prices(broken, today=date(2026, 10, 30))
    assert poor.grade == "slabă"
    data = build_dataset({"BAD": broken, "OK": make_prices(n=2600, seed=4)}, None, horizon=20)
    assert set(data["ticker"]) == {"OK"}
    assert data.attrs["excluded"]["BAD"].startswith("date slabe")


def test_cross_check_with_reported_52_week_high():
    px = make_prices(n=2600)
    hi = float(px["Close"].iloc[-252:].max())
    assert check_prices(px, today=TODAY, fifty_two_week_high=hi * 1.02).grade == "bună"
    assert "Sursele se potrivesc" in failed(check_prices(px, today=TODAY, fifty_two_week_high=hi * 3))


def test_partial_bar_is_dropped_until_the_new_york_close():
    px = make_prices(n=300)
    last = px.index[-1].date()
    before = datetime(last.year, last.month, last.day, 11, 30, tzinfo=NEW_YORK)
    after = datetime(last.year, last.month, last.day, 16, 30, tzinfo=NEW_YORK)
    assert len(drop_partial_bar(px, before)) == len(px) - 1
    assert len(drop_partial_bar(px, after)) == len(px)
    assert len(drop_partial_bar(px, datetime(2026, 9, 25, 10, 0, tzinfo=NEW_YORK))) == len(px)
    assert np.isclose(drop_partial_bar(px, before)["Close"].iloc[-1], px["Close"].iloc[-2])
