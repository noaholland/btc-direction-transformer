"""Level-1 order flow from tick data: Bloomberg IntradayTick exports or the Hyperliquid recorder.

Tick CSV columns: time, type (TRADE | BID | ASK), price, size [, side]. Bloomberg's own names
(`value` for price, BEST_BID / BID_BEST style event types) are accepted.

Pipeline:  load_ticks -> book_events -> make_bars -> add_features_and_labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_ALIASES = {"BEST_BID": "BID", "BID_BEST": "BID", "BEST_ASK": "ASK", "ASK_BEST": "ASK"}
FEATURES = ["ofi", "flow", "ret_bps", "spread_bps", "imbalance", "log_volume", "log_trades",
            "log_quotes", "rv_bps", "tod_sin", "tod_cos"]


def load_ticks(path, tz: str = "UTC", max_gap: str = "5min") -> pd.DataFrame:
    """Read a tick CSV, normalise names, sort by time (stable, so same-timestamp order is kept),
    and number trading sessions: a silence longer than `max_gap` (overnight, a recorder drop) starts a new one."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"value": "price"})
    t = pd.to_datetime(df["time"], format="ISO8601")
    df["time"] = t.dt.tz_localize(tz).dt.tz_convert("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    df["type"] = df["type"].str.upper().str.strip().replace(EVENT_ALIASES)
    df = df[df["type"].isin(["TRADE", "BID", "ASK"]) & (df["price"] > 0)]
    # trades before quotes on equal timestamps: Hyperliquid stamps a trade and the book update it causes with the same
    # block time (Bloomberg with the same second), so each trade must be signed against the quote that was there before
    df = df.assign(_q=df["type"].ne("TRADE")).sort_values(["time", "_q"], kind="stable").drop(columns="_q").reset_index(drop=True)
    df["session"] = (df["time"].diff() > pd.Timedelta(max_gap)).cumsum()
    return df


def book_events(ticks: pd.DataFrame) -> pd.DataFrame:
    """Carry the best bid/ask state across every tick and derive the two order-flow quantities.

    ofi_e : Cont, Kukanov & Stoikov (2014) order-flow imbalance of each event. Bid side adds the new
            bid size if the bid held or rose and removes the old size if it held or fell; the ask side
            is the mirror image. Positive = net buying pressure at the top of the book. Trades leave the
            quote state untouched, so they contribute 0 here; their effect shows up in the next quote.
    sign  : trade aggressor by Lee & Ready (1991): above the prevailing mid = buyer, below = seller,
            exactly at the mid = tick rule (direction of the last price change).
    """
    t = ticks.copy()
    s = t["session"]
    is_bid, is_ask, is_trade = t["type"].eq("BID"), t["type"].eq("ASK"), t["type"].eq("TRADE")
    for col, mask, src in (("bid", is_bid, "price"), ("bid_sz", is_bid, "size"), ("ask", is_ask, "price"), ("ask_sz", is_ask, "size")):
        t[col] = t[src].where(mask).groupby(s).ffill()
    pb, pq, pa, paq = (t[c].groupby(s).shift() for c in ("bid", "bid_sz", "ask", "ask_sz"))
    e = (np.where(t["bid"] >= pb, t["bid_sz"], 0) - np.where(t["bid"] <= pb, pq, 0)
         - np.where(t["ask"] <= pa, t["ask_sz"], 0) + np.where(t["ask"] >= pa, paq, 0))
    t["ofi_e"] = np.nan_to_num(e)

    tr = t[is_trade]
    step = tr["price"].groupby(s[is_trade]).diff().replace(0.0, np.nan)
    tick_sign = np.sign(step.groupby(s[is_trade]).ffill()).fillna(0.0)
    quote_sign = np.sign(tr["price"] - (tr["bid"] + tr["ask"]) / 2).fillna(0.0)
    t["sign"] = 0.0
    t.loc[is_trade, "sign"] = quote_sign.where(quote_sign != 0, tick_sign)
    t["trade_vol"] = t["size"].where(is_trade, 0.0)
    t["signed_vol"] = t["sign"] * t["trade_vol"]
    t["is_trade"], t["is_quote"] = is_trade.astype(float), (~is_trade).astype(float)
    return t


def lee_ready_accuracy(events: pd.DataFrame) -> float | None:
    """Share of trades whose inferred sign matches the true aggressor (needs a `side` column, e.g. Hyperliquid)."""
    if "side" not in events:
        return None
    tr = events[events["type"].eq("TRADE") & events["side"].isin(["B", "A"])]
    return float((tr["sign"] == np.where(tr["side"] == "B", 1.0, -1.0)).mean()) if len(tr) else None


def make_bars(events: pd.DataFrame, bar: str = "5s") -> pd.DataFrame:
    """Clock-time bars per session: summed flow inside the bar, book state at the bar's end."""
    out = []
    for sid, g in events.groupby("session"):
        r = g.set_index("time").resample(bar)
        b = pd.DataFrame({
            "ofi_raw": r["ofi_e"].sum(), "signed_vol": r["signed_vol"].sum(),
            "volume": r["trade_vol"].sum(), "n_trades": r["is_trade"].sum(), "n_quotes": r["is_quote"].sum(),
            "bid": r["bid"].last(), "bid_sz": r["bid_sz"].last(), "ask": r["ask"].last(), "ask_sz": r["ask_sz"].last(),
        })
        b[["bid", "bid_sz", "ask", "ask_sz"]] = b[["bid", "bid_sz", "ask", "ask_sz"]].ffill()
        b["session"] = sid
        out.append(b.dropna(subset=["bid", "ask"]))
    return pd.concat(out)


def add_features_and_labels(bars: pd.DataFrame, horizon: int = 12, norm_window: int = 120, tz: str = "America/New_York") -> pd.DataFrame:
    """Per-bar features (only past data) and two targets H bars ahead.

    y_dir : 1 if the mid rises by more than half the current spread over the next H bars, 0 if it falls by
            more, NaN in between (too small to matter, left out of training and AUC).
    y_vol : log(1 + traded volume over the next H bars).
    """
    f = bars.copy()
    g = lambda col: f[col].groupby(f["session"])  # noqa: E731 — everything rolls/shifts within a session
    mid = (f["bid"] + f["ask"]) / 2
    f["mid"] = mid
    depth = g("bid_sz").transform(lambda x: x.rolling(norm_window, min_periods=1).mean()) + \
        g("ask_sz").transform(lambda x: x.rolling(norm_window, min_periods=1).mean())
    avg_vol = g("volume").transform(lambda x: x.rolling(norm_window, min_periods=1).mean())
    f["ofi"] = f["ofi_raw"] / (depth / 2 + 1e-12)
    f["flow"] = f["signed_vol"] / (avg_vol + 1e-12)
    f["ret_bps"] = 1e4 * np.log(mid).groupby(f["session"]).diff().fillna(0.0)
    f["spread_bps"] = 1e4 * (f["ask"] - f["bid"]) / mid
    tot = f["bid_sz"] + f["ask_sz"]
    f["imbalance"] = ((f["bid_sz"] - f["ask_sz"]) / tot.where(tot > 0)).fillna(0.0)
    f["log_volume"], f["log_trades"], f["log_quotes"] = np.log1p(f["volume"]), np.log1p(f["n_trades"]), np.log1p(f["n_quotes"])
    f["rv_bps"] = g("ret_bps").transform(lambda x: x.rolling(60, min_periods=2).std()).fillna(0.0)
    et = f.index.tz_convert(tz)
    frac = (et.hour * 3600 + et.minute * 60 + et.second) / 86400.0
    f["tod_sin"], f["tod_cos"] = np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)

    fut_mid = g("mid").shift(-horizon)
    move = fut_mid - mid
    dead = (f["ask"] - f["bid"]) / 2
    f["y_dir"] = np.where(move > dead, 1.0, np.where(move < -dead, 0.0, np.nan))
    f.loc[fut_mid.isna(), "y_dir"] = np.nan
    cum = g("volume").cumsum()
    fut_vol = cum.groupby(f["session"]).shift(-horizon) - cum          # volume in bars t+1 .. t+H
    f["y_vol"] = np.log1p(fut_vol)
    f["past_vol"] = np.log1p(cum - cum.groupby(f["session"]).shift(horizon).fillna(0.0))  # bars t-H+1 .. t (baseline)
    f["fwd_ret_bps"] = 1e4 * move / mid
    return f


def valid_window_ends(sessions: np.ndarray, seq_len: int) -> np.ndarray:
    """Row positions whose trailing window of seq_len bars is one unbroken run of a single session."""
    run = np.r_[0, np.cumsum(sessions[1:] != sessions[:-1])]
    i = np.arange(seq_len - 1, len(sessions))
    return i[run[i] == run[i - seq_len + 1]]
