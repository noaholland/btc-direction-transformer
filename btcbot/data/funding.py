"""Hyperliquid hourly funding history for the perp (paid hourly, sign: longs pay when positive)."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .prices import HL_INFO_URL, HL_TESTNET_INFO_URL, _request_json, _session, _to_utc

log = logging.getLogger(__name__)


def fetch_hl_funding(coin: str, start: datetime, end: datetime, testnet: bool = False) -> pd.DataFrame:
    """Page through fundingHistory (500 rows per call). Returns [funding_rate, premium] indexed by UTC time."""
    url = HL_TESTNET_INFO_URL if testnet else HL_INFO_URL
    s = _session()
    start_ms = int(_to_utc(start).timestamp() * 1000)
    end_ms = int(_to_utc(end).timestamp() * 1000)
    rows: list[tuple] = []
    while start_ms < end_ms:
        data = _request_json(s, "POST", url, json={"type": "fundingHistory", "coin": coin, "startTime": start_ms, "endTime": end_ms})
        if not data:
            break
        rows.extend((int(d["time"]), float(d["fundingRate"]), float(d.get("premium", 0.0))) for d in data)
        last = int(data[-1]["time"])
        if last <= start_ms or len(data) < 500:
            break
        start_ms = last + 1
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame(columns=["funding_rate", "premium"], index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(rows, columns=["time_ms", "funding_rate", "premium"]).drop_duplicates("time_ms").sort_values("time_ms")
    # funding timestamps are hh:00:00.0xx — snap to the hour
    df["time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True).dt.floor("h").astype("datetime64[ns, UTC]")
    df = df.drop(columns="time_ms").drop_duplicates("time", keep="last").set_index("time")
    df.index.name = "time"
    return df


def load_or_fetch_funding(path: Path, coin: str, start: datetime, end: datetime, update: bool = True) -> pd.DataFrame:
    existing = None
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["time"], index_col="time")
        existing.index = existing.index.tz_localize("UTC") if existing.index.tz is None else existing.index.tz_convert("UTC")
    if existing is not None and len(existing):
        if not update:
            return existing
        new = fetch_hl_funding(coin, existing.index[-1].to_pydatetime(), end)
        df = pd.concat([existing, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = fetch_hl_funding(coin, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="time")
    return df
