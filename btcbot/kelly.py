"""Kelly sizing and Hyperliquid isolated-margin liquidation math.

Terminology
  p            model confidence in the chosen direction (>= 0.5)
  b            payoff ratio: net gain if take-profit hits / net loss if stop hits (fractions of notional)
  f*           full Kelly fraction of equity to put *at risk* = p - (1 - p) / b
  risk_frac    fraction of equity actually lost if the stop is hit (after fees and slippage)
  notional     position size in USD = risk_frac * equity / loss_per_notional
  margin       isolated margin posted = notional / leverage

Leverage does not change Kelly's answer. It only changes how much collateral you post for a
given notional, and how far away liquidation sits. With 40x on Hyperliquid the maintenance
margin is 1.25% of notional, so an isolated position is liquidated after a ~1.23-1.27% adverse
move; the stop must therefore sit well inside that.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def kelly_fraction(p: float, b: float) -> float:
    """Full Kelly fraction for a binary bet with win prob p and payoff ratio b. Never negative."""
    if b <= 0:
        return 0.0
    return max(0.0, p - (1.0 - p) / b)


def maintenance_margin_frac(max_leverage: float) -> float:
    """Hyperliquid: maintenance margin is half the initial margin at max leverage."""
    return 0.5 / max_leverage


def liquidation_distance(leverage: float, max_leverage: float, side: int) -> float:
    """Adverse fractional price move at which an isolated position is liquidated (fees ignored).

    Derived from Hyperliquid's liq_price = px - side * margin_available / size / (1 - l * side),
    with margin_available = notional/leverage - l * notional and l = maintenance margin fraction.
    """
    m = maintenance_margin_frac(max_leverage)
    base = 1.0 / leverage - m
    if base <= 0:
        return 0.0
    return base / (1.0 - m) if side > 0 else base / (1.0 + m)


def liquidation_price(entry: float, leverage: float, max_leverage: float, side: int) -> float:
    d = liquidation_distance(leverage, max_leverage, side)
    return entry * (1.0 - d) if side > 0 else entry * (1.0 + d)


def payoff_ratio(take_profit_pct: float | None, stop_loss_pct: float, fee_round_trip: float, slippage: float) -> float:
    """Net win / net loss per unit notional. Without a take-profit assume a symmetric 1:1 gross payoff."""
    tp = stop_loss_pct if take_profit_pct is None else take_profit_pct
    win = tp - fee_round_trip - slippage
    loss = stop_loss_pct + fee_round_trip + slippage
    return max(1e-6, win) / loss


@dataclass
class TradePlan:
    side: int                 # +1 long, -1 short
    p_conf: float             # confidence in `side`
    payoff_b: float
    f_full: float             # full Kelly fraction
    risk_frac: float          # fraction of equity lost if the stop is hit
    notional: float
    margin: float
    size: float               # coin size
    entry_px: float
    stop_px: float
    tp_px: float | None
    liq_px: float
    liq_dist: float
    sizing_mode: str
    exit_after_bars: int

    def to_dict(self) -> dict:
        return asdict(self)


def floor_to_decimals(x: float, decimals: int) -> float:
    q = 10 ** decimals
    return math.floor(x * q + 1e-9) / q


def plan_trade(equity: float, p_up: float, price: float, tcfg: dict, horizon_bars: int,
               max_leverage: float = 40.0, sz_decimals: int = 5) -> tuple[TradePlan | None, str]:
    """Turn a calibrated P(up) into a fully specified order, or (None, reason)."""
    thr = float(tcfg["confidence_threshold"])
    side = 1 if p_up >= 0.5 else -1
    p = p_up if side > 0 else 1.0 - p_up
    if p < thr:
        return None, f"below_threshold (conf={p:.3f} < {thr})"

    leverage = float(tcfg["leverage"])
    sl = float(tcfg["stop_loss_pct"])
    tp = tcfg.get("take_profit_pct")
    tp = None if tp in (None, "null", "") else float(tp)
    fee_rt = 2.0 * float(tcfg["taker_fee"])
    slip = float(tcfg["slippage_bps"]) / 1e4 * 2.0

    liq_dist = liquidation_distance(leverage, max_leverage, side)
    if sl > float(tcfg["stop_liq_buffer"]) * liq_dist:
        return None, f"stop_outside_liquidation (stop {sl:.4f} > {tcfg['stop_liq_buffer']} x liq {liq_dist:.4f})"

    b = payoff_ratio(tp, sl, fee_rt, slip)
    f_full = kelly_fraction(p, b)
    if f_full <= 0:
        return None, f"kelly_nonpositive (p={p:.3f}, b={b:.2f})"

    loss_per_notional = sl + fee_rt + slip
    max_risk = float(tcfg["max_risk_per_trade"])
    mode = str(tcfg.get("sizing_mode", "risk"))
    if mode == "margin":
        margin = max_risk * equity
        notional = margin * leverage
        risk_frac = notional * loss_per_notional / equity
    else:
        risk_frac = min(float(tcfg["kelly_multiplier"]) * f_full, max_risk)
        notional = risk_frac * equity / loss_per_notional
        margin = notional / leverage
        cap = float(tcfg["max_margin_fraction"]) * equity
        if margin > cap:
            margin = cap
            notional = margin * leverage
            risk_frac = notional * loss_per_notional / equity

    size = floor_to_decimals(notional / price, sz_decimals)
    notional = size * price
    margin = notional / leverage
    risk_frac = notional * loss_per_notional / equity  # what the rounded size actually risks
    if notional < float(tcfg["min_notional_usd"]):
        return None, f"below_min_notional (${notional:.2f})"

    stop_px = price * (1.0 - sl) if side > 0 else price * (1.0 + sl)
    tp_px = None if tp is None else (price * (1.0 + tp) if side > 0 else price * (1.0 - tp))
    plan = TradePlan(
        side=side, p_conf=p, payoff_b=b, f_full=f_full, risk_frac=risk_frac, notional=notional, margin=margin,
        size=size, entry_px=price, stop_px=stop_px, tp_px=tp_px,
        liq_px=liquidation_price(price, leverage, max_leverage, side), liq_dist=liq_dist,
        sizing_mode=mode, exit_after_bars=horizon_bars,
    )
    return plan, "ok"
