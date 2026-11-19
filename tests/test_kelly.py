import math

import pytest

from btcbot.kelly import kelly_fraction, liquidation_distance, liquidation_price, payoff_ratio, plan_trade

TCFG = {
    "leverage": 40, "sizing_mode": "risk", "kelly_multiplier": 0.25, "max_risk_per_trade": 0.025,
    "stop_loss_pct": 0.008, "take_profit_pct": 0.016, "confidence_threshold": 0.65, "max_margin_fraction": 0.5,
    "stop_liq_buffer": 0.8, "taker_fee": 0.00045, "maker_fee": 0.00015, "slippage_bps": 1.0, "min_notional_usd": 10.0,
}


def test_kelly_fraction_basics():
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert math.isclose(kelly_fraction(0.6, 1.0), 0.2)
    assert math.isclose(kelly_fraction(0.6, 2.0), 0.4)
    assert kelly_fraction(0.4, 1.0) == 0.0  # never negative
    assert kelly_fraction(0.9, 0.0) == 0.0


def test_liquidation_distance_40x():
    # 40x max leverage -> maintenance margin 1.25%; long liquidates ~1.27% down, short ~1.23% up
    assert math.isclose(liquidation_distance(40, 40, +1), 0.0125 / 0.9875, rel_tol=1e-9)
    assert math.isclose(liquidation_distance(40, 40, -1), 0.0125 / 1.0125, rel_tol=1e-9)
    assert liquidation_distance(10, 40, +1) > liquidation_distance(40, 40, +1)
    assert liquidation_price(100_000, 40, 40, +1) < 100_000 < liquidation_price(100_000, 40, 40, -1)


def test_payoff_ratio_nets_out_costs():
    b = payoff_ratio(0.016, 0.008, 0.0009, 0.0002)
    assert math.isclose(b, (0.016 - 0.0011) / (0.008 + 0.0011))
    assert payoff_ratio(None, 0.008, 0.0, 0.0) == 1.0


def test_plan_trade_risk_mode_caps_at_max_risk():
    plan, reason = plan_trade(1000.0, 0.70, 80_000.0, TCFG, horizon_bars=16)
    assert plan is not None, reason
    assert plan.side == 1
    loss_per_notional = 0.008 + 0.0009 + 0.0002
    assert plan.risk_frac <= 0.025 + 1e-9
    assert math.isclose(plan.notional * loss_per_notional / 1000.0, plan.risk_frac, rel_tol=1e-6)
    assert math.isclose(plan.margin, plan.notional / 40, rel_tol=1e-9)
    assert plan.margin <= 0.5 * 1000.0
    # stop is inside liquidation
    assert plan.stop_px > plan.liq_px
    assert plan.tp_px is not None and plan.tp_px > plan.entry_px
    assert 0 < plan.f_full < 1


def test_plan_trade_short_side():
    plan, _ = plan_trade(1000.0, 0.25, 80_000.0, TCFG, horizon_bars=16)
    assert plan is not None and plan.side == -1
    assert plan.stop_px > plan.entry_px > plan.tp_px
    assert plan.liq_px > plan.stop_px


def test_plan_trade_below_threshold():
    plan, reason = plan_trade(1000.0, 0.60, 80_000.0, TCFG, horizon_bars=16)
    assert plan is None and reason.startswith("below_threshold")


def test_plan_trade_rejects_stop_outside_liquidation():
    cfg = dict(TCFG, stop_loss_pct=0.012)
    plan, reason = plan_trade(1000.0, 0.9, 80_000.0, cfg, horizon_bars=16)
    assert plan is None and reason.startswith("stop_outside_liquidation")


def test_plan_trade_margin_mode_is_literal():
    cfg = dict(TCFG, sizing_mode="margin")
    plan, _ = plan_trade(1000.0, 0.70, 80_000.0, cfg, horizon_bars=16)
    assert plan is not None
    assert math.isclose(plan.margin, 25.0, rel_tol=0.01)          # 2.5% of equity posted
    assert math.isclose(plan.notional, 1000.0, rel_tol=0.01)      # 40x -> full equity notional
    assert plan.risk_frac < 0.01                                  # but the stop only risks ~0.9%


def test_plan_trade_margin_cap():
    cfg = dict(TCFG, kelly_multiplier=1.0, max_risk_per_trade=0.5)
    plan, _ = plan_trade(1000.0, 0.95, 80_000.0, cfg, horizon_bars=16)
    assert plan is not None
    assert plan.margin <= 500.0 + 1e-6


def test_plan_trade_min_notional():
    plan, reason = plan_trade(2.0, 0.9, 80_000.0, TCFG, horizon_bars=16)
    assert plan is None and reason.startswith("below_min_notional")
