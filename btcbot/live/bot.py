"""Live loop: score the newest bar at the configured decision times, size with Kelly, trade on Hyperliquid.

Modes (config: live.mode)
  dry_run  - everything except order placement; paper-tracks positions with real prices
  testnet  - real orders on Hyperliquid testnet (free test USDC from the testnet faucet)
  mainnet  - real money

State is persisted to live.state_file after every step so a restart resumes cleanly.
Kill switches: touch live.pause_file, or the daily loss / consecutive-loss guards.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import resolve
from ..data.etf_flows import load_flows_csv, update_flows_csv
from ..data.funding import fetch_hl_funding
from ..data.prices import fetch_coinbase, fetch_hyperliquid_candles, fill_gaps, resample_bars
from ..features import build_features
from ..kelly import TradePlan, plan_trade
from ..model.dataset import FeatureScaler, windows_view
from ..model.archs import make_model

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------
# model bundle
# ---------------------------------------------------------------------------------------------
class ModelBundle:
    def __init__(self, model_dir: Path):
        with open(model_dir / "bundle.json") as fh:
            self.meta = json.load(fh)
        self.feature_cols = self.meta["feature_cols"]
        self.scaler = FeatureScaler.from_dict(self.meta["scaler"])
        self.T = float(self.meta["temperature"])
        self.seq_len = int(self.meta["seq_len"])
        m = self.meta["model"]
        self.models = []
        for k in range(int(self.meta.get("n_models", 1))):
            model = make_model(m.get("arch", "transformer"), len(self.feature_cols), self.seq_len, int(m["d_model"]), int(m["n_heads"]), int(m["n_layers"]), float(m["dropout"]))
            model.load_state_dict(torch.load(model_dir / f"model_{k}.pt", map_location="cpu"))
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def p_up(self, feats: pd.DataFrame) -> float:
        """Calibrated ensemble P(up) for the LAST row of `feats` (which must contain >= seq_len rows)."""
        if len(feats) < self.seq_len:
            raise ValueError(f"need {self.seq_len} rows, got {len(feats)}")
        X = self.scaler.transform(feats.iloc[-self.seq_len :])
        win = windows_view(X, self.seq_len)[-1:].float()
        logit = float(np.mean([float(m(win)[0]) for m in self.models]))
        return float(1.0 / (1.0 + np.exp(-logit / self.T)))


# ---------------------------------------------------------------------------------------------
# live features
# ---------------------------------------------------------------------------------------------
def live_features(cfg: dict, decision_ts: pd.Timestamp, flows: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Rebuild the feature frame from fresh data, using only bars that closed at or before decision_ts."""
    bar_min = int(cfg["data"]["bar_minutes"])
    seq_len = int(cfg["features"]["seq_len"])
    lookback_bars = seq_len + 400  # rolling-window warmup (168h at 30m = 336 bars)
    start = decision_ts - pd.Timedelta(minutes=bar_min * lookback_bars)
    px15 = fetch_coinbase(start.to_pydatetime(), decision_ts.to_pydatetime(), 900)
    bars = fill_gaps(resample_bars(px15, bar_min), bar_min)
    bars = bars[bars.index + pd.Timedelta(minutes=bar_min) <= decision_ts]
    funding = fetch_hl_funding(cfg["trading"]["coin"], (start - pd.Timedelta(days=2)).to_pydatetime(), decision_ts.to_pydatetime())
    feats, cols = build_features(bars, funding, flows, cfg)
    return feats, cols


# ---------------------------------------------------------------------------------------------
# bot
# ---------------------------------------------------------------------------------------------
class Bot:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lcfg, self.tcfg = cfg["live"], cfg["trading"]
        self.tz = cfg["features"]["tz"]
        self.bar_min = int(cfg["data"]["bar_minutes"])
        self.H = int(cfg["features"]["horizon_bars"])
        self.mode = str(self.lcfg["mode"])
        self.coin = self.tcfg["coin"]
        self.state_path = resolve(cfg, self.lcfg["state_file"])
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.bundle = ModelBundle(resolve(cfg, cfg["walkforward"]["model_dir"]) / "final")
        self.flows_path = resolve(cfg, cfg["data"]["etf_flows_csv"])
        self.flows = load_flows_csv(self.flows_path)
        self.client = None
        if self.mode in ("testnet", "mainnet"):
            from dotenv import load_dotenv

            from .hl_client import HLClient

            load_dotenv(Path(cfg["_root"]) / ".env")
            key, addr = os.environ.get("HL_PRIVATE_KEY"), os.environ.get("HL_ACCOUNT_ADDRESS")
            if not key or not addr:
                raise SystemExit("HL_PRIVATE_KEY / HL_ACCOUNT_ADDRESS missing from environment (.env)")
            self.client = HLClient(key, addr, testnet=(self.mode == "testnet"))
            self.max_leverage = self.client.max_leverage.get(self.coin, 40.0)
            self.sz_decimals = self.client.sz_decimals.get(self.coin, 5)
        else:
            self.max_leverage, self.sz_decimals = 40.0, 5
        log.info("bot mode=%s coin=%s leverage=%sx sizing=%s thr=%s decision_times=%s", self.mode, self.coin, self.tcfg["leverage"],
                 self.tcfg["sizing_mode"], self.tcfg["confidence_threshold"], self.tcfg["decision_times_et"])

    # ---- state -----------------------------------------------------------------------------
    def _load_state(self) -> dict:
        if self.state_path.exists():
            with open(self.state_path) as fh:
                return json.load(fh)
        return {"position": None, "last_decision_key": None, "day": None, "day_start_equity": None,
                "consecutive_losses": 0, "paused_until": None, "paper_equity": float(self.cfg["backtest"]["initial_equity"]), "trades": []}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=2, default=str)
        tmp.replace(self.state_path)

    # ---- helpers ---------------------------------------------------------------------------
    def equity(self) -> float:
        return self.client.equity() if self.client else float(self.state["paper_equity"])

    def price(self) -> float:
        if self.client:
            return self.client.mid(self.coin)
        c = fetch_hyperliquid_candles(self.coin, "1m", start_ms=int((time.time() - 600) * 1000))
        return float(c["close"].iloc[-1])

    def _paused(self, now: datetime) -> str | None:
        if resolve(self.cfg, self.lcfg["pause_file"]).exists():
            return "pause file present"
        pu = self.state.get("paused_until")
        if pu and datetime.fromisoformat(pu) > now:
            return f"paused until {pu}"
        day = pd.Timestamp(now).tz_convert(self.tz).date().isoformat()
        if self.state.get("day") != day:
            self.state["day"], self.state["day_start_equity"] = day, self.equity()
        dse = self.state.get("day_start_equity") or 0
        if dse and self.equity() < dse * (1 - float(self.lcfg["daily_loss_limit"])):
            return f"daily loss limit hit (equity {self.equity():.2f} < {dse * (1 - float(self.lcfg['daily_loss_limit'])):.2f})"
        if self.flows is not None and len(self.flows):
            age = (now.date() - pd.Timestamp(self.flows["date"].iloc[-1]).date()).days
            if age > int(self.lcfg["max_flow_age_days"]):
                return f"ETF flow data is {age} days old (> {self.lcfg['max_flow_age_days']})"
        return None

    def maybe_refresh_flows(self, now_et: pd.Timestamp) -> None:
        key = now_et.date().isoformat()
        if now_et.hour >= int(self.lcfg["etf_refresh_hour_et"]) and self.state.get("flows_refreshed_day") != key:
            flows, ok = update_flows_csv(self.flows_path, self.cfg["data"]["farside_url"])
            if flows is not None:
                self.flows = flows
            self.state["flows_refreshed_day"] = key
            log.info("ETF flow refresh %s (latest %s)", "ok" if ok else "FAILED, using cache",
                     None if self.flows is None else self.flows["date"].iloc[-1].date())

    # ---- main step -----------------------------------------------------------------------------
    def step(self) -> None:
        now = datetime.now(timezone.utc)
        now_et = pd.Timestamp(now).tz_convert(self.tz)
        self.maybe_refresh_flows(now_et)
        if self.client and not self.state.get("position"):
            self.adopt_unknown_position(now)
        if self.state.get("position"):
            self.reconcile_position(now)
        if not self.state.get("position"):
            self.maybe_enter(now, now_et)
        self._save_state()

    def adopt_unknown_position(self, now: datetime) -> None:
        """If the exchange shows a position the state file does not know about (lost state, manual trade),
        adopt it and give it the normal horizon instead of opening a second one."""
        live = self.client.position(self.coin)
        if live is None:
            return
        side = 1 if live["size"] > 0 else -1
        size, entry = abs(live["size"]), live["entry_px"]
        sl, tp = float(self.tcfg["stop_loss_pct"]), self.tcfg.get("take_profit_pct")
        tp = None if tp in (None, "null", "") else float(tp)
        pos = {"side": side, "size": size, "entry_px": entry, "notional": size * entry, "margin": live["margin_used"],
               "stop_px": entry * (1 - sl * side), "tp_px": None if tp is None else entry * (1 + tp * side),
               "liq_px": live["liq_px"] or entry * (1 - 0.0125 * side), "risk_frac": float("nan"), "p_up": float("nan"),
               "p_conf": float("nan"), "opened_at": now.isoformat(),
               "exit_at": (now + timedelta(minutes=self.bar_min * self.H)).isoformat(), "oids": {}, "mode": self.mode, "adopted": True}
        log.warning("adopting unknown %s position of %.5f %s @ %.1f from the exchange", "LONG" if side > 0 else "SHORT", size, self.coin, entry)
        if not self.client.open_orders(self.coin):
            pos["oids"] = self.client.place_stop_and_tp(self.coin, side > 0, size, pos["stop_px"], pos["tp_px"])
        self.state["position"] = pos

    def maybe_enter(self, now: datetime, now_et: pd.Timestamp) -> None:
        decision_ts = pd.Timestamp(now).floor(f"{self.bar_min}min")
        key = decision_ts.isoformat()
        if decision_ts.tz_convert(self.tz).strftime("%H:%M") not in set(self.tcfg["decision_times_et"]):
            return
        if self.state.get("last_decision_key") == key:
            return
        if (pd.Timestamp(now) - decision_ts) > pd.Timedelta(minutes=10):
            self.state["last_decision_key"] = key  # too late for this bar; do not chase
            return
        self.state["last_decision_key"] = key
        why = self._paused(now)
        if why:
            log.info("decision %s skipped: %s", key, why)
            return
        # Coinbase needs a few seconds to finalise the 15m bar that just closed
        time.sleep(5)
        feats, _ = live_features(self.cfg, decision_ts, self.flows)
        if len(feats) == 0 or feats["decision_ts"].iloc[-1] != decision_ts:
            log.warning("decision %s: newest bar not available yet (have %s)", key, None if len(feats) == 0 else feats["decision_ts"].iloc[-1])
            self.state["last_decision_key"] = None  # retry on next poll
            return
        p_up = self.bundle.p_up(feats)
        equity = self.equity()
        price = self.price()
        plan, reason = plan_trade(equity, p_up, price, self.tcfg, self.H, self.max_leverage, self.sz_decimals)
        conf = max(p_up, 1 - p_up)
        log.info("decision %s: p_up=%.3f conf=%.3f equity=%.2f px=%.1f -> %s", key, p_up, conf, equity, price, reason)
        self.state.setdefault("decisions", []).append({"ts": key, "p_up": p_up, "equity": equity, "price": price, "reason": reason})
        self.state["decisions"] = self.state["decisions"][-500:]
        if plan is None:
            return
        self.open_position(plan, decision_ts, p_up)

    def open_position(self, plan: TradePlan, decision_ts: pd.Timestamp, p_up: float) -> None:
        exit_at = decision_ts + pd.Timedelta(minutes=self.bar_min * self.H)
        pos = {"side": plan.side, "size": plan.size, "entry_px": plan.entry_px, "notional": plan.notional, "margin": plan.margin,
               "stop_px": plan.stop_px, "tp_px": plan.tp_px, "liq_px": plan.liq_px, "risk_frac": plan.risk_frac, "p_up": p_up,
               "p_conf": plan.p_conf, "opened_at": decision_ts.isoformat(), "exit_at": exit_at.isoformat(), "oids": {}, "mode": self.mode}
        if self.client:
            self.client.set_isolated_leverage(self.coin, int(self.tcfg["leverage"]))
            px, sz = self.client.market_open(self.coin, plan.side > 0, plan.size)
            pos["entry_px"], pos["size"], pos["notional"] = px, sz, px * sz
            # re-anchor stop/TP/liq to the actual fill
            ratio = px / plan.entry_px
            pos["stop_px"], pos["liq_px"] = plan.stop_px * ratio, plan.liq_px * ratio
            pos["tp_px"] = None if plan.tp_px is None else plan.tp_px * ratio
            try:
                pos["oids"] = self.client.place_stop_and_tp(self.coin, plan.side > 0, sz, pos["stop_px"], pos["tp_px"])
            except Exception as e:  # noqa: BLE001 — never leave a naked 40x position
                log.error("stop/TP placement failed (%s); flattening the position immediately", e)
                self.client.cancel_all(self.coin)
                fill = self.client.market_close(self.coin)
                exit_px = fill[0] if fill else self.client.mid(self.coin)
                pnl = plan.side * (exit_px - px) * sz - 2 * float(self.tcfg["taker_fee"]) * px * sz
                self.state["position"] = pos
                self._close_bookkeeping(pos, exit_px, pnl, "aborted_no_stop", datetime.now(timezone.utc))
                return
            log.info("LIVE %s %s %.5f %s @ %.1f (notional $%.0f, margin $%.2f, stop %.1f, tp %s, liq %.1f) oids=%s", self.mode.upper(),
                     "LONG" if plan.side > 0 else "SHORT", sz, self.coin, px, pos["notional"], pos["margin"], pos["stop_px"], pos["tp_px"], pos["liq_px"], pos["oids"])
        else:
            log.info("DRY-RUN %s %.5f %s @ %.1f (notional $%.0f, margin $%.2f, risk %.2f%% of equity, stop %.1f, tp %s, liq %.1f, exit by %s)",
                     "LONG" if plan.side > 0 else "SHORT", plan.size, self.coin, plan.entry_px, plan.notional, plan.margin, 100 * plan.risk_frac,
                     plan.stop_px, plan.tp_px, plan.liq_px, exit_at)
        self.state["position"] = pos

    def reconcile_position(self, now: datetime) -> None:
        pos = self.state["position"]
        exit_at = datetime.fromisoformat(pos["exit_at"])
        if self.client:
            live = self.client.position(self.coin)
            if live is None:  # closed by stop / TP / liquidation
                self.client.cancel_all(self.coin)
                fills = self.client.fills_since(int(datetime.fromisoformat(pos["opened_at"]).timestamp() * 1000))
                closing = [f for f in fills if f["coin"] == self.coin and (float(f["sz"]) > 0) and ((f["side"] == "A") == (pos["side"] > 0))]
                exit_px = float(np.average([float(f["px"]) for f in closing], weights=[float(f["sz"]) for f in closing])) if closing else self.client.mid(self.coin)
                pnl = sum(float(f.get("closedPnl", 0)) for f in closing) - sum(float(f.get("fee", 0)) for f in fills if f["coin"] == self.coin)
                self._close_bookkeeping(pos, exit_px, pnl, "triggered", now)
            elif now >= exit_at:
                fill = self.client.market_close(self.coin)
                self.client.cancel_all(self.coin)
                exit_px = fill[0] if fill else self.client.mid(self.coin)
                pnl = pos["side"] * (exit_px - pos["entry_px"]) * pos["size"] - 2 * float(self.tcfg["taker_fee"]) * pos["notional"]
                self._close_bookkeeping(pos, exit_px, pnl, "horizon", now)
            return
        # paper mode: replay bars since entry
        opened = pd.Timestamp(pos["opened_at"])
        c = fetch_hyperliquid_candles(self.coin, "1m", start_ms=int(opened.timestamp() * 1000))
        c = c[c.index >= opened]
        if len(c) == 0:
            return
        side, fee = pos["side"], float(self.tcfg["taker_fee"])
        exit_px, reason = None, None
        for ts, row in c.iterrows():
            if side > 0:
                if row["low"] <= pos["liq_px"]:
                    exit_px, reason = pos["liq_px"], "liquidated"
                elif row["low"] <= pos["stop_px"]:
                    exit_px, reason = pos["stop_px"], "stop"
                elif pos["tp_px"] is not None and row["high"] >= pos["tp_px"]:
                    exit_px, reason = pos["tp_px"], "take_profit"
            else:
                if row["high"] >= pos["liq_px"]:
                    exit_px, reason = pos["liq_px"], "liquidated"
                elif row["high"] >= pos["stop_px"]:
                    exit_px, reason = pos["stop_px"], "stop"
                elif pos["tp_px"] is not None and row["low"] <= pos["tp_px"]:
                    exit_px, reason = pos["tp_px"], "take_profit"
            if exit_px is not None:
                break
        if exit_px is None and now >= exit_at:
            exit_px, reason = float(c["close"].iloc[-1]), "horizon"
        if exit_px is None:
            return
        pnl = -pos["margin"] if reason == "liquidated" else max(side * (exit_px - pos["entry_px"]) * pos["size"] - 2 * fee * pos["notional"], -pos["margin"])
        self.state["paper_equity"] = float(self.state["paper_equity"]) + pnl
        self._close_bookkeeping(pos, exit_px, pnl, reason, now)

    def _close_bookkeeping(self, pos: dict, exit_px: float, pnl: float, reason: str, now: datetime) -> None:
        rec = {**pos, "exit_px": exit_px, "pnl": pnl, "exit_reason": reason, "closed_at": now.isoformat(), "equity_after": self.equity()}
        self.state["trades"].append(rec)
        self.state["position"] = None
        self.state["consecutive_losses"] = 0 if pnl > 0 else int(self.state.get("consecutive_losses", 0)) + 1
        if self.state["consecutive_losses"] >= int(self.lcfg["max_consecutive_losses"]):
            self.state["paused_until"] = (now + timedelta(hours=24)).isoformat()
            log.warning("%d consecutive losses -> pausing new entries for 24h", self.state["consecutive_losses"])
        log.info("CLOSED %s %s @ %.1f pnl %+.2f (%s) equity now %.2f", "LONG" if pos["side"] > 0 else "SHORT", self.coin, exit_px, pnl, reason, self.equity())
        _append_csv(resolve(self.cfg, self.lcfg["log_dir"]) / "trades.csv", rec)

    def run(self, once: bool = False) -> None:
        poll = int(self.lcfg["poll_seconds"])
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 — keep the loop alive, log everything
                log.exception("step failed: %s", e)
            if once:
                return
            time.sleep(poll)


def _append_csv(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in rec.items()}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_bot(cfg: dict, once: bool = False) -> None:
    log_dir = resolve(cfg, cfg["live"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "bot.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    if cfg["live"]["mode"] == "mainnet":
        log.warning("MAINNET MODE — real orders will be placed with real money")
    Bot(cfg).run(once=once)


def predict_now(cfg: dict) -> None:
    """Score the most recently closed bar and print what the bot would do (no orders, no state)."""
    bar_min = int(cfg["data"]["bar_minutes"])
    decision_ts = pd.Timestamp.now(tz="UTC").floor(f"{bar_min}min")
    bundle = ModelBundle(resolve(cfg, cfg["walkforward"]["model_dir"]) / "final")
    flows = load_flows_csv(resolve(cfg, cfg["data"]["etf_flows_csv"]))
    feats, _ = live_features(cfg, decision_ts, flows)
    if feats["decision_ts"].iloc[-1] != decision_ts:
        decision_ts = feats["decision_ts"].iloc[-1]
    p_up = bundle.p_up(feats)
    price = float(feats["close"].iloc[-1])
    equity = float(cfg["backtest"]["initial_equity"])
    plan, reason = plan_trade(equity, p_up, price, cfg["trading"], int(cfg["features"]["horizon_bars"]))
    et = decision_ts.tz_convert(cfg["features"]["tz"])
    print(f"bar closed {decision_ts} ({et.strftime('%a %H:%M')} ET)  BTC {price:,.0f}")
    print(f"P(up) = {p_up:.3f}   confidence = {max(p_up, 1 - p_up):.3f}   threshold = {cfg['trading']['confidence_threshold']}")
    print(f"latest ETF flow print: {None if flows is None else flows['date'].iloc[-1].date()}  total ${None if flows is None else flows['total'].iloc[-1]:.1f}m")
    if plan is None:
        print(f"no trade: {reason}")
    else:
        d = plan.to_dict()
        print(f"would go {'LONG' if plan.side > 0 else 'SHORT'} with ${equity:.0f} equity: " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}))
