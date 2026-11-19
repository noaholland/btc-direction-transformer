"""Record Hyperliquid best bid/ask and trades in the same tick layout as the Bloomberg exporter.

    .venv/bin/python -m orderflow.record_hyperliquid BTC data/ticks/hl_BTC.csv

Columns: time (UTC ISO), type (BID | ASK | TRADE), price, size, side.
`side` is the trade aggressor (B = buyer lifted the ask, A = seller hit the bid). Bloomberg ticks
have no aggressor flag, so the feature code infers it with Lee-Ready; this file lets you check
how often that inference is right.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import websocket

URL = "wss://api.hyperliquid.xyz/ws"


def _ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat(timespec="milliseconds")


def record(coin: str, path: str) -> None:
    last = {"BID": None, "ASK": None}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "type", "price", "size", "side"])

        def on_open(ws):
            for kind in ("bbo", "trades"):
                ws.send(json.dumps({"method": "subscribe", "subscription": {"type": kind, "coin": coin}}))

        def on_message(ws, raw):
            m = json.loads(raw)
            ch, d = m.get("channel"), m.get("data")
            if ch == "bbo":
                for kind, lvl in zip(("BID", "ASK"), d["bbo"]):
                    # like Bloomberg: one BID/ASK tick whenever that side's price or size changes
                    if lvl and (lvl["px"], lvl["sz"]) != last[kind]:
                        last[kind] = (lvl["px"], lvl["sz"])
                        w.writerow([_ts(d["time"]), kind, lvl["px"], lvl["sz"], ""])
            elif ch == "trades":
                for t in d:
                    w.writerow([_ts(t["time"]), "TRADE", t["px"], t["sz"], t["side"]])
            fh.flush()

        while True:  # BTC streams continuously, so the server never idles us out; just reconnect on drops
            websocket.WebSocketApp(URL, on_open=on_open, on_message=on_message).run_forever()
            last.update(BID=None, ASK=None)
            time.sleep(2)


if __name__ == "__main__":
    record(sys.argv[1] if len(sys.argv) > 1 else "BTC", sys.argv[2] if len(sys.argv) > 2 else "data/ticks/hl_BTC.csv")
