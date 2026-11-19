"""Export Bloomberg intraday ticks (trades plus best bid/ask with sizes) into the tick CSV that orderflow/ reads.

Runs only on the Windows PC where the Bloomberg Terminal is logged in: the Desktop API talks to bbcomm on
localhost:8194, and Bloomberg's macOS/Linux libraries do not work with a Terminal. Needs only blpapi + stdlib.

    python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
    python bbg_export.py "IBIT US Equity" 2026-10-27T13:30:00 2026-10-27T20:00:00 ibit_20261027.csv

Times are UTC (the API is always GMT; Excel BDH uses local time). History reaches back about 140 business days.
A request costs securities x event types against the Terminal's daily/monthly limits however many ticks come
back, so pull a whole session per request. Before copying the CSV off the Terminal PC, read DAPI<GO>: many
institutions' terms keep API data on that machine.
"""
import csv
import sys
from datetime import datetime, timezone

EVENTS = ("TRADE", "BID", "ASK")


def tick_rows(ticks):
    """IntradayTickResponse ticks (dicts) -> CSV rows [time, type, price, size, side, cond, exch].

    Bloomberg has no aggressor flag for equities, so `side` stays empty and orderflow infers it (Lee-Ready).
    Some futures (e.g. CME) carry aggressor condition codes AB / AS; those fill `side`.
    """
    for t in ticks:
        ts = t["time"] if t["time"].tzinfo else t["time"].replace(tzinfo=timezone.utc)
        codes = t.get("conditionCodes", "")
        tokens = set(codes.replace(";", ",").split(","))
        side = "B" if "AB" in tokens else "A" if "AS" in tokens else ""
        yield [ts.isoformat(), t["type"], t["value"], t["size"], side, codes, t.get("exchangeCode", "")]


def _ticks(msg):
    arr = msg.getElement("tickData").getElement("tickData")
    for i in range(arr.numValues()):
        e = arr.getValueAsElement(i)
        d = {"time": e.getElementAsDatetime("time"), "type": e.getElementAsString("type"),
             "value": e.getElementAsFloat("value"), "size": e.getElementAsFloat("size")}
        for f in ("conditionCodes", "exchangeCode"):
            if e.hasElement(f):
                d[f] = e.getElementAsString(f)
        yield d


def export(security: str, start: str, end: str, path: str, events=EVENTS, host="localhost", port=8194) -> int:
    import blpapi

    opts = blpapi.SessionOptions()
    opts.setServerHost(host)
    opts.setServerPort(port)
    session = blpapi.Session(opts)
    if not session.start():
        raise SystemExit("cannot reach bbcomm on localhost:8194: is the Terminal running and logged in on this PC?")
    if not session.openService("//blp/refdata"):
        raise SystemExit("cannot open //blp/refdata")
    req = session.getService("//blp/refdata").createRequest("IntradayTickRequest")
    req.set("security", security)
    for ev in events:
        req.append("eventTypes", ev)
    req.set("startDateTime", start)                  # yyyy-mm-ddThh:mm:ss, UTC
    req.set("endDateTime", end)
    req.set("includeConditionCodes", True)
    req.set("includeExchangeCodes", True)
    session.sendRequest(req)
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "type", "price", "size", "side", "cond", "exch"])
        while True:
            ev = session.nextEvent(5000)
            for msg in ev:
                if msg.hasElement("responseError"):    # e.g. category LIMIT, DAILY_LIMIT_REACHED / MONTHLY_LIMIT_REACHED
                    raise SystemExit(f"Bloomberg refused the request: {msg.getElement('responseError')}")
                if msg.hasElement("tickData"):
                    for row in tick_rows(_ticks(msg)):
                        w.writerow(row)
                        n += 1
            if ev.eventType() == blpapi.Event.RESPONSE:  # PARTIAL_RESPONSE events come first, then one RESPONSE
                break
    session.stop()
    return n


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    sec, start, end, out = sys.argv[1:]
    for s in (start, end):
        datetime.fromisoformat(s)                    # fail fast on a typo before spending a request
    print(f"{export(sec, start, end, out)} ticks written to {out}")
