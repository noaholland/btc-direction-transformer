import math

import numpy as np
import pandas as pd

from btcbot.backtest import simulate
from btcbot.config import DEFAULTS


def _cfg():
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    cfg["backtest"]["initial_equity"] = 1000.0
    return cfg


def _setup(path):
    """Flat bars at 80k, then a scripted path after the decision bar (which ends 09:00 ET on a weekday)."""
    n_pre = 20
    start = pd.Timestamp("2025-06-03 05:00", tz="America/New_York").tz_convert("UTC")  # Tuesday
    idx = pd.date_range(start, periods=n_pre + len(path), freq="30min", tz="UTC")
    close = np.array([80_000.0] * n_pre + [p[0] for p in path])
    high = np.array([80_000.0] * n_pre + [p[1] for p in path])
    low = np.array([80_000.0] * n_pre + [p[2] for p in path])
    bars = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)
    decision_bar = idx[n_pre - 1]                       # last flat bar; ends at 09:00 ET? check
    decision_ts = decision_bar + pd.Timedelta(minutes=30)
    label = decision_ts.tz_convert("America/New_York").strftime("%H:%M")
    funding_idx = pd.date_range(idx[0].floor("h"), idx[-1].ceil("h"), freq="h", tz="UTC")
    funding = pd.DataFrame({"funding_rate": 0.0, "premium": 0.0}, index=funding_idx)
    preds = pd.DataFrame({"decision_ts": [decision_ts], "p_up": [0.90]})
    return bars, funding, preds, label


def test_stop_hit_loses_about_the_risk_budget():
    # price falls 1% in the 3rd bar after entry -> stop (0.8%) hit
    path = [(80_000, 80_050, 79_950)] * 2 + [(79_200, 80_000, 79_100)] + [(79_200, 79_300, 79_100)] * 20
    bars, funding, preds, label = _setup(path)
    cfg = _cfg()
    trades, eq = simulate(preds, bars, funding, cfg, decision_times=[label])
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "stop" and t["side"] == 1
    loss_frac = -t["pnl"] / 1000.0
    assert 0.02 < loss_frac < 0.027            # ~2.5% of equity, fees/slippage included
    assert math.isclose(eq.iloc[-1], 1000.0 + t["pnl"])


def test_take_profit_hit():
    path = [(80_000, 80_050, 79_950)] + [(81_400, 81_500, 80_900)] + [(81_400, 81_450, 81_350)] * 20
    bars, funding, preds, label = _setup(path)
    trades, _ = simulate(preds, bars, funding, _cfg(), decision_times=[label])
    t = trades.iloc[0]
    assert t["exit_reason"] == "take_profit"
    assert t["pnl"] > 0
    # 2:1 payoff -> gain is roughly twice the risk budget
    assert 0.04 < t["pnl"] / 1000.0 < 0.055


def test_horizon_exit_when_nothing_triggers():
    path = [(80_100, 80_200, 79_900)] * 30
    bars, funding, preds, label = _setup(path)
    trades, _ = simulate(preds, bars, funding, _cfg(), decision_times=[label])
    t = trades.iloc[0]
    assert t["exit_reason"] == "horizon"
    assert t["bars_held"] == DEFAULTS["features"]["horizon_bars"]
    assert abs(t["pnl"]) < 20.0


def test_liquidation_loses_full_margin():
    cfg = _cfg()
    cfg["trading"]["stop_loss_pct"] = 0.0099  # stop right at the buffer so a gap can jump through it
    path = [(80_000, 80_050, 79_950)] + [(78_000, 80_000, 77_900)] + [(78_000, 78_100, 77_900)] * 20
    bars, funding, preds, label = _setup(path)
    trades, _ = simulate(preds, bars, funding, cfg, decision_times=[label])
    t = trades.iloc[0]
    assert t["exit_reason"] == "liquidated"
    assert math.isclose(-t["pnl"], t["margin"], rel_tol=1e-9)


def test_no_trade_below_threshold_or_outside_decision_times():
    path = [(80_100, 80_200, 79_900)] * 30
    bars, funding, preds, label = _setup(path)
    trades, eq = simulate(preds.assign(p_up=0.55), bars, funding, _cfg(), decision_times=[label])
    assert len(trades) == 0 and math.isclose(eq.iloc[-1], 1000.0)
    trades, _ = simulate(preds, bars, funding, _cfg(), decision_times=["23:30"])
    assert len(trades) == 0
