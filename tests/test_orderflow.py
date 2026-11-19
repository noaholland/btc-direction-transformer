import numpy as np
import pandas as pd

from orderflow.features import add_features_and_labels, book_events, load_ticks, make_bars, valid_window_ends


def _ticks(tmp_path, rows):
    p = tmp_path / "t.csv"
    pd.DataFrame(rows, columns=["time", "type", "price", "size"]).to_csv(p, index=False)
    return load_ticks(p)


def test_ofi_matches_cont_kukanov_stoikov(tmp_path):
    t0 = "2026-10-27T14:00:0{}Z"
    ev = book_events(_ticks(tmp_path, [
        (t0.format(0), "BID", 100.0, 5), (t0.format(1), "ASK", 101.0, 5),
        (t0.format(2), "BID", 100.0, 8),     # bid queue grows by 3        -> +3
        (t0.format(3), "ASK", 101.0, 2),     # ask queue shrinks by 3      -> +3
        (t0.format(4), "BID", 100.5, 1),     # bid improves, new size 1    -> +1
        (t0.format(5), "ASK", 100.75, 4),    # ask improves, new size 4    -> -4
        (t0.format(6), "TRADE", 100.75, 1),  # trades do not move the quote state
    ]))
    assert ev["ofi_e"].tolist() == [0, 0, 3, 3, 1, -4, 0]


def test_lee_ready_quote_rule_then_tick_rule(tmp_path):
    t0 = "2026-10-27T14:00:0{}Z"
    ev = book_events(_ticks(tmp_path, [
        (t0.format(0), "BID", 100.0, 5), (t0.format(1), "ASK", 101.0, 5),
        (t0.format(2), "TRADE", 100.5, 1),   # at the mid, no earlier trade -> unknown (0)
        (t0.format(3), "TRADE", 101.0, 1),   # at the ask -> buyer
        (t0.format(4), "TRADE", 100.0, 1),   # at the bid -> seller
        (t0.format(5), "TRADE", 100.5, 1),   # at the mid after a 100 -> uptick -> buyer
    ]))
    assert ev.loc[ev["type"] == "TRADE", "sign"].tolist() == [0, 1, -1, 1]


def test_sessions_split_on_gaps_and_windows_never_cross_them(tmp_path):
    t = _ticks(tmp_path, [("2026-10-27T14:00:00Z", "BID", 1, 1), ("2026-10-27T14:01:00Z", "ASK", 2, 1),
                          ("2026-10-28T14:00:00Z", "BID", 1, 1)])
    assert t["session"].tolist() == [0, 0, 1]
    ends = valid_window_ends(np.array([0, 0, 0, 1, 1, 1, 1]), seq_len=3)
    assert ends.tolist() == [2, 5, 6]


def test_labels_look_exactly_h_bars_ahead_and_respect_the_dead_zone():
    idx = pd.date_range("2026-10-27 14:00", periods=6, freq="5s", tz="UTC")
    bars = pd.DataFrame({"ofi_raw": 0.0, "signed_vol": 0.0, "volume": [1, 2, 3, 4, 5, 6], "n_trades": 1.0, "n_quotes": 1.0,
                         "bid": [100, 100, 101, 100, 100, 99], "bid_sz": 1.0, "ask": [101, 101, 102, 101, 101, 100],
                         "ask_sz": 1.0, "session": 0}, index=idx)
    f = add_features_and_labels(bars, horizon=2)
    assert f["y_vol"].tolist()[:4] == list(np.log1p([2 + 3, 3 + 4, 4 + 5, 5 + 6]))
    assert np.isnan(f["y_vol"].iloc[-1])
    # mid path 100.5, 100.5, 101.5, 100.5, 100.5, 99.5; half spread = 0.5
    y = f["y_dir"].to_numpy()
    assert y[0] == 1.0          # 100.5 -> 101.5 (+1 > 0.5)
    assert np.isnan(y[1])       # 100.5 -> 100.5 (flat)
    assert y[2] == 0.0          # 101.5 -> 100.5
    assert y[3] == 0.0          # 100.5 -> 99.5
    assert np.isnan(y[4]) and np.isnan(y[5])   # horizon runs past the data
    assert f["past_vol"].tolist()[:3] == list(np.log1p([1, 1 + 2, 2 + 3]))


def test_bars_carry_book_state_through_quiet_bars(tmp_path):
    ev = book_events(_ticks(tmp_path, [("2026-10-27T14:00:00Z", "BID", 100.0, 5), ("2026-10-27T14:00:01Z", "ASK", 101.0, 5),
                                       ("2026-10-27T14:00:12Z", "TRADE", 101.0, 2)]))
    b = make_bars(ev, "5s")
    assert len(b) == 3                        # 14:00:00, :05 (no events), :10
    assert b["bid"].tolist() == [100, 100, 100] and b["volume"].tolist() == [0, 0, 2]
    assert b["signed_vol"].tolist() == [0, 0, 2]


def test_bloomberg_tick_rows_and_loader_round_trip(tmp_path):
    from datetime import datetime

    from orderflow.bbg_export import tick_rows

    ticks = [  # what blpapi hands back: naive UTC datetimes, Bloomberg event names, optional codes
        {"time": datetime(2026, 10, 27, 14, 0, 0), "type": "BEST_BID", "value": 60.01, "size": 500.0, "exchangeCode": "Q"},
        {"time": datetime(2026, 10, 27, 14, 0, 0), "type": "BEST_ASK", "value": 60.02, "size": 300.0},
        {"time": datetime(2026, 10, 27, 14, 0, 1), "type": "TRADE", "value": 60.02, "size": 100.0, "conditionCodes": "AB"},
        {"time": datetime(2026, 10, 27, 14, 0, 1), "type": "TRADE", "value": 60.01, "size": 200.0, "conditionCodes": "@,F"},
    ]
    rows = list(tick_rows(ticks))
    assert rows[0][:4] == ["2026-10-27T14:00:00+00:00", "BEST_BID", 60.01, 500.0]
    assert [r[4] for r in rows] == ["", "", "B", ""]
    p = tmp_path / "bbg.csv"
    pd.DataFrame(rows, columns=["time", "type", "price", "size", "side", "cond", "exch"]).to_csv(p, index=False)
    ev = book_events(load_ticks(p))
    assert ev["type"].tolist() == ["BID", "ASK", "TRADE", "TRADE"]
    assert ev.loc[ev["type"] == "TRADE", "sign"].tolist() == [1.0, -1.0]   # at the ask = buy, at the bid = sell


def test_excel_local_time_export_is_converted_to_utc(tmp_path):
    p = tmp_path / "xl.csv"
    pd.DataFrame({"Time": ["2026-10-27 10:00:00"], "Type": ["BID"], "Value": [60.0], "Size": [100]}).to_csv(p, index=False)
    t = load_ticks(p, tz="America/New_York")
    assert str(t["time"].iloc[0]) == "2026-10-27 14:00:00+00:00"


def test_trades_sort_before_quotes_with_the_same_timestamp(tmp_path):
    # Hyperliquid: the trade lifts the ask and the new book arrives first with the same block time
    ev = book_events(_ticks(tmp_path, [("2026-10-27T14:00:00Z", "BID", 100.0, 5), ("2026-10-27T14:00:00Z", "ASK", 101.0, 5),
                                       ("2026-10-27T14:00:01Z", "ASK", 102.0, 5), ("2026-10-27T14:00:01Z", "TRADE", 101.0, 5)]))
    assert ev["type"].tolist() == ["BID", "ASK", "TRADE", "ASK"]
    assert ev.loc[ev["type"] == "TRADE", "sign"].tolist() == [1.0]       # bought at the old ask


def test_windows_reject_interleaved_sessions():
    assert valid_window_ends(np.array([0, 1, 0, 1, 0]), 3).tolist() == []
