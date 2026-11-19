"""Command line entry point: python -m btcbot <command>.

Commands:
  fetch      download/refresh candles, funding and ETF flows into data/
  sessions   day-of-week x session performance tables (no model)
  train      walk-forward train the transformer, write OOS predictions + final model
  backtest   simulate Kelly-sized 40x trades on the OOS predictions
  predict    one-off prediction with the saved final model (what the bot would do now)
  bot        run the live loop (mode from config.yaml: dry_run | testnet | mainnet)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import load_config, resolve


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def cmd_fetch(cfg: dict, args) -> None:
    from .data.etf_flows import update_flows_csv
    from .data.funding import load_or_fetch_funding
    from .data.prices import load_or_fetch_prices

    d = cfg["data"]
    data_dir = resolve(cfg, d["data_dir"])
    start = datetime.fromisoformat(d["start"]).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc) - timedelta(minutes=1)
    if not args.skip_prices:
        px = load_or_fetch_prices(data_dir / f"btc_15m_{d['price_source']}.csv", start, end, d["price_source"], 900)
        print(f"prices: {len(px)} x 15m bars, {px.index[0]} -> {px.index[-1]}")
    if not args.skip_funding:
        fu = load_or_fetch_funding(data_dir / "funding_btc.csv", cfg["trading"]["coin"], start, end)
        print(f"funding: {len(fu)} hourly rows, {fu.index[0]} -> {fu.index[-1]}")
    if not args.skip_flows:
        flows, updated = update_flows_csv(resolve(cfg, d["etf_flows_csv"]), d["farside_url"])
        if flows is None:
            print("ETF flows: none available (Farside blocked and no cached CSV). See README for the CSV fallback.")
        else:
            print(f"ETF flows: {len(flows)} trading days, latest {flows['date'].iloc[-1].date()} (refreshed={updated})")


def cmd_sessions(cfg: dict, args) -> None:
    from .sessions import run_session_analysis

    run_session_analysis(cfg)


def cmd_train(cfg: dict, args) -> None:
    from .model.walkforward import run_walkforward

    run_walkforward(cfg, skip_final=args.skip_final)


def cmd_backtest(cfg: dict, args) -> None:
    from .backtest import run_backtest_report

    run_backtest_report(cfg)


def cmd_predict(cfg: dict, args) -> None:
    from .live.bot import predict_now

    predict_now(cfg)


def cmd_bot(cfg: dict, args) -> None:
    from .live.bot import run_bot

    run_bot(cfg, once=args.once)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="btcbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--skip-prices", action="store_true")
    f.add_argument("--skip-funding", action="store_true")
    f.add_argument("--skip-flows", action="store_true")
    f.set_defaults(fn=cmd_fetch)

    sub.add_parser("sessions").set_defaults(fn=cmd_sessions)

    t = sub.add_parser("train")
    t.add_argument("--skip-final", action="store_true", help="only walk-forward, do not train the deployable final model")
    t.set_defaults(fn=cmd_train)

    sub.add_parser("backtest").set_defaults(fn=cmd_backtest)
    sub.add_parser("predict").set_defaults(fn=cmd_predict)

    b = sub.add_parser("bot")
    b.add_argument("--once", action="store_true", help="run a single poll iteration and exit")
    b.set_defaults(fn=cmd_bot)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
