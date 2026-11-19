"""Feature engineering on fixed-size bars (default 30 minutes).

Conventions (these matter for look-ahead safety):
  * A row is indexed by its bar's *start* time; the decision it supports happens at the bar's
    *end* (`decision_ts`), when the bar's OHLCV is final. Entry price for that decision is the
    bar's close.
  * Funding and ETF flow values are merged as-of `decision_ts` with backward direction, so a row
    only ever sees prints whose publication time is <= decision_ts.
  * The label `y` is the sign of log(close[t+H] / close[t]); rows whose horizon has not completed
    have y = NaN (they are kept so the live bot can score the newest bar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_et import nyse_open_mask
from .data.etf_flows import flows_known_at

ONE_HOT_COLS = [f"dow_{i}" for i in range(7)] + ["nyse_open"]
META_COLS = ["fwd_ret", "y", "close", "decision_ts"]


def build_features(bars: pd.DataFrame, funding: pd.DataFrame, flows: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    fcfg, dcfg = cfg["features"], cfg["data"]
    bar_min = int(dcfg["bar_minutes"])
    tz = fcfg["tz"]
    H = int(fcfg["horizon_bars"])
    per_hour = 60 // bar_min

    bars = bars.sort_index()
    bars.index = _ns_utc(bars.index)
    o, h, l, c, v = bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"]
    decision_ts = _ns_utc(bars.index + pd.Timedelta(minutes=bar_min))

    f = pd.DataFrame(index=bars.index)
    logc = np.log(c)
    f["ret_1"] = logc.diff()
    for hours in (2, 8, 24, 72):
        f[f"ret_{hours}h"] = logc.diff(hours * per_hour)
    f["vol_24h"] = f["ret_1"].rolling(24 * per_hour).std()
    f["vol_168h"] = f["ret_1"].rolling(168 * per_hour).std()
    f["range"] = (h - l) / c
    f["body"] = (c - o) / c
    lv = np.log1p(v)
    w = 168 * per_hour
    f["vol_z"] = (lv - lv.rolling(w).mean()) / (lv.rolling(w).std() + 1e-9)
    f["dist_hi_24h"] = c / h.rolling(24 * per_hour).max() - 1.0
    f["dist_lo_24h"] = c / l.rolling(24 * per_hour).min() - 1.0

    # --- perp funding (hourly; known at its own timestamp) ---
    fu = funding.sort_index().copy()
    fu.index = _ns_utc(fu.index)
    fu.index.name = "ts"
    fu["funding_24h"] = fu["funding_rate"].rolling(24, min_periods=1).sum()
    left = pd.DataFrame({"ts": decision_ts})
    fu_asof = pd.merge_asof(left, fu.reset_index(), on="ts", direction="backward")
    f["funding"] = fu_asof["funding_rate"].fillna(0.0).to_numpy() * 1e4
    f["funding_24h"] = fu_asof["funding_24h"].fillna(0.0).to_numpy() * 1e4

    # --- clock features, from the decision time in ET ---
    et = decision_ts.tz_convert(tz)
    hour_frac = et.hour + et.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0)
    f["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0)
    dow = et.dayofweek
    for i in range(7):
        f[f"dow_{i}"] = (dow == i).astype(float)
    f["nyse_open"] = nyse_open_mask(et).to_numpy().astype(float)

    # --- ETF flows (daily, known at 16:00 ET + publish lag) ---
    if flows is not None and len(flows):
        fk = flows_known_at(flows, float(dcfg["etf_flow_publish_lag_hours"]), tz)
        fk = fk.assign(ts=_ns_utc(fk["known_at"])).sort_values("ts")
        fk_asof = pd.merge_asof(left, fk, on="ts", direction="backward")
        f["flow_total"] = fk_asof["total"].fillna(0.0).to_numpy() / 100.0
        f["flow_5d"] = fk_asof["total_5d"].fillna(0.0).to_numpy() / 100.0
        f["flow_20d"] = fk_asof["total_20d"].fillna(0.0).to_numpy() / 100.0
        f["flow_ibit"] = fk_asof["IBIT"].fillna(0.0).to_numpy() / 100.0
        f["flow_fbtc"] = fk_asof["FBTC"].fillna(0.0).to_numpy() / 100.0
        f["flow_gbtc"] = fk_asof["GBTC"].fillna(0.0).to_numpy() / 100.0
        f["flow_pos_frac"] = fk_asof["pos_frac"].fillna(0.5).to_numpy()
        age_days = (fk_asof["ts"] - fk_asof["known_at"]).dt.total_seconds().to_numpy() / 86400.0
        f["flow_age_days"] = np.clip(np.nan_to_num(age_days, nan=10.0), 0.0, 10.0)
    else:
        for col in ("flow_total", "flow_5d", "flow_20d", "flow_ibit", "flow_fbtc", "flow_gbtc"):
            f[col] = 0.0
        f["flow_pos_frac"] = 0.5
        f["flow_age_days"] = 10.0

    # --- labels / meta ---
    f["fwd_ret"] = logc.shift(-H) - logc
    f["y"] = np.where(f["fwd_ret"].notna(), (f["fwd_ret"] > 0).astype(float), np.nan)
    f["close"] = c
    f["decision_ts"] = decision_ts

    feature_cols = [col for col in f.columns if col not in META_COLS]
    f = f.dropna(subset=feature_cols)
    return f, feature_cols


def continuous_cols(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c not in ONE_HOT_COLS]


def _ns_utc(x):
    """Coerce a DatetimeIndex/Series to tz-aware UTC nanoseconds (merge_asof needs identical units)."""
    if isinstance(x, pd.Series):
        x = pd.to_datetime(x, utc=True)
        return x.astype("datetime64[ns, UTC]")
    idx = pd.DatetimeIndex(x)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.astype("datetime64[ns, UTC]")
