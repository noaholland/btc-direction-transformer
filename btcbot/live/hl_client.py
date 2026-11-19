"""Thin wrapper over hyperliquid-python-sdk for the few calls the bot needs.

Security: pass an *API wallet* key (Hyperliquid -> Settings -> API). API wallets can trade
but cannot withdraw. The key is read from the environment by the bot and never logged.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

log = logging.getLogger(__name__)


class HLClient:
    def __init__(self, private_key: str, account_address: str, testnet: bool = False):
        import eth_account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        base = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.testnet = testnet
        self.address = account_address
        self.info = Info(base, skip_ws=True)
        wallet = eth_account.Account.from_key(private_key)
        self.exchange = Exchange(wallet, base, account_address=account_address)
        meta = self.info.meta()
        self.sz_decimals = {a["name"]: int(a["szDecimals"]) for a in meta["universe"]}
        self.max_leverage = {a["name"]: float(a["maxLeverage"]) for a in meta["universe"]}
        log.info("HLClient ready (%s), signer %s", "testnet" if testnet else "MAINNET", wallet.address[:8] + "…")

    # ---- formatting -------------------------------------------------------------------------
    def round_px(self, coin: str, px: float) -> float:
        """Hyperliquid: max 5 significant figures and at most (6 - szDecimals) decimals."""
        return round(float(f"{px:.5g}"), 6 - self.sz_decimals[coin])

    def floor_sz(self, coin: str, sz: float) -> float:
        q = 10 ** self.sz_decimals[coin]
        return math.floor(sz * q + 1e-9) / q

    # ---- account ------------------------------------------------------------------------------
    def user_state(self) -> dict:
        return self.info.user_state(self.address)

    def equity(self) -> float:
        return float(self.user_state()["marginSummary"]["accountValue"])

    def position(self, coin: str) -> dict | None:
        for ap in self.user_state().get("assetPositions", []):
            p = ap["position"]
            if p["coin"] == coin and float(p["szi"]) != 0.0:
                return {"size": float(p["szi"]), "entry_px": float(p["entryPx"]), "liq_px": float(p["liquidationPx"] or 0),
                        "margin_used": float(p["marginUsed"]), "unrealized_pnl": float(p["unrealizedPnl"]),
                        "leverage": p["leverage"]}
        return None

    def mid(self, coin: str) -> float:
        return float(self.info.all_mids()[coin])

    def open_orders(self, coin: str) -> list[dict]:
        return [o for o in self.info.open_orders(self.address) if o["coin"] == coin]

    def fills_since(self, start_ms: int) -> list[dict]:
        return self.info.user_fills_by_time(self.address, start_ms)

    # ---- trading -------------------------------------------------------------------------------
    def set_isolated_leverage(self, coin: str, leverage: int) -> Any:
        res = self.exchange.update_leverage(leverage, coin, is_cross=False)
        _raise_on_error(res, "update_leverage")
        return res

    def market_open(self, coin: str, is_buy: bool, size: float, slippage: float = 0.005) -> tuple[float, float]:
        """Returns (avg fill px, filled size)."""
        res = self.exchange.market_open(coin, is_buy, size, None, slippage)
        _raise_on_error(res, "market_open")
        return _parse_fill(res)

    def market_close(self, coin: str, slippage: float = 0.005) -> tuple[float, float] | None:
        res = self.exchange.market_close(coin, None, None, slippage)
        if res is None:
            return None
        _raise_on_error(res, "market_close")
        return _parse_fill(res)

    def place_stop_and_tp(self, coin: str, position_is_long: bool, size: float, stop_px: float, tp_px: float | None) -> dict:
        """Reduce-only trigger orders that close the position. Returns {'sl': oid, 'tp': oid|None}."""
        is_buy = not position_is_long
        tol = 0.03  # worst acceptable fill once triggered
        oids: dict[str, int | None] = {"sl": None, "tp": None}
        sl_trig = self.round_px(coin, stop_px)
        sl_limit = self.round_px(coin, stop_px * (1 + tol if is_buy else 1 - tol))
        res = self.exchange.order(coin, is_buy, size, sl_limit, {"trigger": {"triggerPx": sl_trig, "isMarket": True, "tpsl": "sl"}}, reduce_only=True)
        _raise_on_error(res, "stop order")
        oids["sl"] = _parse_oid(res)
        if tp_px is not None:
            tp_trig = self.round_px(coin, tp_px)
            tp_limit = self.round_px(coin, tp_px * (1 + tol if is_buy else 1 - tol))
            res = self.exchange.order(coin, is_buy, size, tp_limit, {"trigger": {"triggerPx": tp_trig, "isMarket": True, "tpsl": "tp"}}, reduce_only=True)
            _raise_on_error(res, "take-profit order")
            oids["tp"] = _parse_oid(res)
        return oids

    def cancel_all(self, coin: str) -> int:
        n = 0
        for o in self.open_orders(coin):
            try:
                self.exchange.cancel(coin, o["oid"])
                n += 1
            except Exception as e:  # noqa: BLE001
                log.warning("cancel %s failed: %s", o["oid"], e)
        return n


def _raise_on_error(res: Any, what: str) -> None:
    if not isinstance(res, dict):
        return
    if res.get("status") != "ok":
        raise RuntimeError(f"{what} failed: {res}")
    statuses = res.get("response", {}).get("data", {}).get("statuses", [])
    for s in statuses:
        if isinstance(s, dict) and "error" in s:
            raise RuntimeError(f"{what} rejected: {s['error']}")


def _parse_fill(res: dict) -> tuple[float, float]:
    px, sz = 0.0, 0.0
    for s in res.get("response", {}).get("data", {}).get("statuses", []):
        f = s.get("filled") if isinstance(s, dict) else None
        if f:
            fs = float(f["totalSz"])
            px = (px * sz + float(f["avgPx"]) * fs) / (sz + fs) if sz + fs > 0 else float(f["avgPx"])
            sz += fs
    if sz == 0.0:
        raise RuntimeError(f"order not filled: {res}")
    return px, sz


def _parse_oid(res: dict) -> int | None:
    for s in res.get("response", {}).get("data", {}).get("statuses", []):
        if isinstance(s, dict):
            for key in ("resting", "filled"):
                if key in s and "oid" in s[key]:
                    return int(s[key]["oid"])
    return None


def now_ms() -> int:
    return int(time.time() * 1000)
