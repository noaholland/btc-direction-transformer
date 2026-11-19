import numpy as np
import pandas as pd

from btcbot.config import DEFAULTS
from btcbot.data.etf_flows import ALL_COLS
from btcbot.features import build_features


def _bars(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-03-03 00:00", periods=n, freq="30min", tz="UTC")
    close = 80_000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * (1 + rng.uniform(0, 0.002, n))
    low = close * (1 - rng.uniform(0, 0.002, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _funding(bars):
    idx = pd.date_range(bars.index[0].floor("h"), bars.index[-1].ceil("h"), freq="h", tz="UTC")
    return pd.DataFrame({"funding_rate": np.full(len(idx), 1e-5), "premium": 0.0}, index=idx)


def _flows():
    rows = []
    for d, total in (("2025-03-10", 500.0), ("2025-03-11", -200.0)):
        r = {c: 0.0 for c in ALL_COLS}
        r["date"] = pd.Timestamp(d)
        r["IBIT"] = total
        r["total"] = total
        rows.append(r)
    return pd.DataFrame(rows)[ALL_COLS]


def test_labels_match_forward_return():
    bars = _bars()
    f, cols = build_features(bars, _funding(bars), None, DEFAULTS)
    H = DEFAULTS["features"]["horizon_bars"]
    close = bars["close"]
    expected = (close.shift(-H) > close).astype(float)
    lab = f.dropna(subset=["y"])
    assert len(lab) > 500
    assert (lab["y"].to_numpy() == expected.loc[lab.index].to_numpy()).all()
    assert (f["decision_ts"] == f.index + pd.Timedelta(minutes=30)).all()


def test_no_lookahead_in_features():
    bars = _bars()
    cut = bars.index[900]
    bars2 = bars.copy()
    bars2.loc[bars2.index > cut, ["open", "high", "low", "close"]] *= 1.5   # change the future only
    f1, cols = build_features(bars, _funding(bars), None, DEFAULTS)
    f2, _ = build_features(bars2, _funding(bars2), None, DEFAULTS)
    past = f1.index <= cut
    np.testing.assert_allclose(f1.loc[past, cols].to_numpy(), f2.loc[past, cols].to_numpy(), rtol=1e-12, atol=1e-12)


def test_etf_flow_is_visible_only_after_publication():
    bars = _bars()
    cfg = DEFAULTS
    f, _ = build_features(bars, _funding(bars), _flows(), cfg)
    # 2025-03-10 flow is published 16:00 ET + 8h = 2025-03-11 00:00 ET = 04:00 UTC
    known = pd.Timestamp("2025-03-11 04:00", tz="UTC")
    before = f[(f["decision_ts"] < known) & (f["decision_ts"] > known - pd.Timedelta(hours=6))]
    after = f[(f["decision_ts"] >= known) & (f["decision_ts"] < known + pd.Timedelta(hours=6))]
    assert (before["flow_total"] == 0.0).all()
    assert np.allclose(after["flow_total"], 5.0)         # 500 $m / 100
    assert np.allclose(after["flow_ibit"], 5.0)
    # the next day's print (-200) is visible from 2025-03-12 04:00 UTC
    later = f[f["decision_ts"] >= pd.Timestamp("2025-03-12 04:00", tz="UTC")]
    assert np.allclose(later["flow_total"], -2.0)
    assert np.allclose(later["flow_5d"], 3.0)           # 500 - 200
    assert (f["flow_age_days"] >= 0).all()


def test_time_features_are_in_eastern_time():
    bars = _bars()
    f, _ = build_features(bars, _funding(bars), None, DEFAULTS)
    et = f["decision_ts"].dt.tz_convert("America/New_York")
    dow_cols = [f"dow_{i}" for i in range(7)]
    assert (f[dow_cols].sum(axis=1) == 1.0).all()
    assert (f[dow_cols].to_numpy().argmax(axis=1) == et.dt.dayofweek.to_numpy()).all()
    # NYSE-open flag is set at 10:00 ET on a weekday and unset at 03:00 ET
    weekday_10 = f[(et.dt.dayofweek < 5) & (et.dt.hour == 10) & (et.dt.minute == 0)]
    night_3 = f[(et.dt.hour == 3) & (et.dt.minute == 0)]
    assert (weekday_10["nyse_open"] == 1.0).all()
    assert (night_3["nyse_open"] == 0.0).all()
