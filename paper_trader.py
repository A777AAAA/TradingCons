"""
paper_trader.py — Виртуальные сделки на реальных данных.
Интегрируется с существующим outcome_tracker и live_signal.
"""

import json
import os
import logging
import ccxt
from datetime import datetime, timezone

PAPER_FILE   = "paper_trades.json"
BALANCE_FILE = "paper_balance.json"

INITIAL_BALANCE = 600.0
TRADE_PCT       = 0.10   # 10% от баланса
TP_PCT          = 0.03   # +3% тейк-профит
SL_PCT          = 0.015  # -1.5% стоп-лосс

# Единая конфигурация exchange  ✅ ИСПРАВЛЕНО
OKX_CONFIG = {
    'options': {'defaultType': 'spot'},
    'timeout': 30000
}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_balance() -> dict:
    if not os.path.exists(BALANCE_FILE):
        data = {
            "balance":    INITIAL_BALANCE,
            "total_pnl":  0.0,
            "trades":     0,
            "wins":       0,
            "losses":     0,
            "created_at": _now()
        }
        save_balance(data)
        return data
    with open(BALANCE_FILE, "r") as f:
        return json.load(f)


def save_balance(data: dict):
    with open(BALANCE_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_trades() -> list:
    if not os.path.exists(PAPER_FILE):
        return []
    with open(PAPER_FILE, "r") as f:
        return json.load(f)


def save_trades(trades: list):
    with open(PAPER_FILE, "w") as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# Текущая цена
# ─────────────────────────────────────────────
def get_current_price(symbol="TON/USDT") -> float:
    try:
        exchange = ccxt.okx(OKX_CONFIG)  # ✅ ИСПРАВЛЕНО
        ticker   = exchange.fetch_ticker(symbol)
        return float(ticker['last'])
    except Exception as e:
        logger.error(f"[Paper] ❌ Ошибка цены: {e}")
        return 0.0


# ─────────────────────────────────────────────
# Открыть виртуальную сделку
# ─────────────────────────────────────────────
def open_trade(signal: str, price: float,
               confidence: float, symbol="TON/USDT") -> dict | None:
    if signal not in ("BUY", "SELL"):
        return None

    trades = load_trades()

    # Только одна открытая сделка одновременно
    if any(t["status"] == "OPEN" for t in trades):
        logger.info("[Paper] ⚠️ Уже есть открытая сделка")
        return None

    balance    = load_balance()
    amount_usd = round(balance["balance"] * TRADE_PCT, 2)
    qty        = round(amount_usd / price, 4)

    if signal == "BUY":
        tp = round(price * (1 + TP_PCT), 6)
        sl = round(price * (1 - SL_PCT), 6)
    else:
        tp = round(price * (1 - TP_PCT), 6)
        sl = round(price * (1 + SL_PCT), 6)

    trade = {
        "id":          len(trades) + 1,
        "symbol":      symbol,
        "signal":      signal,
        "status":      "OPEN",
        "price_open":  price,
        "qty":         qty,
        "amount_usd":  amount_usd,
        "tp":          tp,
        "sl":          sl,
        "confidence":  round(confidence * 100, 1),
        "opened_at":   _now(),
        "closed_at":   None,
        "price_close": None,
        "pnl_usd":     None,
        "pnl_pct":     None,
        "result":      None,
        "closed_by":   None,
    }

    trades.append(trade)
    save_trades(trades)
    logger.info(
        f"[Paper] 📝 Открыта #{trade['id']}: "
        f"{signal} {symbol} @ {price} | "
        f"TP={tp} SL={sl}"
    )
    return trade


# ─────────────────────────────────────────────
# Мониторинг открытых сделок
# ─────────────────────────────────────────────
def monitor_trades(symbol="TON/USDT") -> list:
    """Проверяет TP/SL. Возвращает список закрытых сделок."""
    trades = load_trades()
    if not any(t["status"] == "OPEN" for t in trades):
        return []

    price = get_current_price(symbol)
    if price == 0.0:
        return []

    balance    = load_balance()
    closed_now = []

    for i, trade in enumerate(trades):
        if trade["status"] != "OPEN":
            continue

        signal = trade["signal"]
        tp     = trade["tp"]
        sl     = trade["sl"]
        hit    = None

        if signal == "BUY":
            if price >= tp:
                hit = "TP"
            elif price <= sl:
                hit = "SL"
        else:
            if price <= tp:
                hit = "TP"
            elif price >= sl:
                hit = "SL"

        if not hit:
            continue

        close_price = tp if hit == "TP" else sl

        if signal == "BUY":
            pnl_pct = (close_price - trade["price_open"]) / trade["price_open"] * 100
        else:
            pnl_pct = (trade["price_open"] - close_price) / trade["price_open"] * 100

        pnl_usd = round(trade["amount_usd"] * pnl_pct / 100, 2)
        result  = "WIN" if hit == "TP" else "LOSS"

        trades[i].update({
            "status":      "CLOSED",
            "closed_at":   _now(),
            "price_close": close_price,
            "pnl_usd":     pnl_usd,
            "pnl_pct":     round(pnl_pct, 2),
            "result":      result,
            "closed_by":   hit,
        })

        balance["balance"]   = round(balance["balance"] + pnl_usd, 2)
        balance["total_pnl"] = round(balance["total_pnl"] + pnl_usd, 2)
        balance["trades"]   += 1
        balance["wins"]     += (1 if result == "WIN" else 0)
        balance["losses"]   += (0 if result == "WIN" else 1)

        closed_now.append(trades[i])
        logger.info(
            f"[Paper] {'✅' if result == 'WIN' else '❌'} "
            f"#{trade['id']} закрыта по {hit}: "
            f"{pnl_pct:+.2f}% | ${pnl_usd:+.2f} | "
            f"Баланс: ${balance['balance']}"
        )

    save_trades(trades)
    save_balance(balance)
    return closed_now


# ─────────────────────────────────────────────
# Статистика
# ─────────────────────────────────────────────
def get_stats() -> dict:
    balance = load_balance()
    trades  = load_trades()
    closed  = [t for t in trades if t["status"] == "CLOSED"]

    total   = balance["trades"]
    wins    = balance["wins"]
    winrate = round(wins / total * 100, 1) if total > 0 else 0

    pnl_list    = [t["pnl_pct"] for t in closed if t.get("pnl_pct") is not None]
    avg_pnl     = round(sum(pnl_list) / len(pnl_list), 2) if pnl_list else 0
    best_trade  = round(max(pnl_list), 2) if pnl_list else 0
    worst_trade = round(min(pnl_list), 2) if pnl_list else 0
    growth_pct  = round(
        (balance["balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2
    )

    return {
        "balance":       balance["balance"],
        "start_balance": INITIAL_BALANCE,
        "growth_pct":    growth_pct,
        "total_pnl":     balance["total_pnl"],
        "total_trades":  total,
        "wins":          wins,
        "losses":        balance["losses"],
        "winrate":       winrate,
        "avg_pnl":       avg_pnl,
        "best_trade":    best_trade,
        "worst_trade":   worst_trade,
        "open_trades":   len([t for t in trades if t["status"] == "OPEN"]),
    }


def format_stats_message(stats: dict) -> str:
    emoji = "📈" if stats["growth_pct"] >= 0 else "📉"
    return (
        f"📊 <b>Paper Trading — Статистика</b>\n\n"
        f"💰 Баланс:       <b>${stats['balance']:.2f}</b> {emoji}\n"
        f"📈 Рост:         <b>{stats['growth_pct']:+.2f}%</b>\n"
        f"💵 P&L всего:    <b>${stats['total_pnl']:+.2f}</b>\n\n"
        f"📋 Сделок:       <b>{stats['total_trades']}</b>\n"
        f"✅ Побед:        <b>{stats['wins']}</b>\n"
        f"❌ Поражений:    <b>{stats['losses']}</b>\n"
        f"🎯 Winrate:      <b>{stats['winrate']}%</b>\n\n"
        f"📊 Средний P&L:  <b>{stats['avg_pnl']:+.2f}%</b>\n"
        f"🏆 Лучшая:       <b>{stats['best_trade']:+.2f}%</b>\n"
        f"💀 Худшая:       <b>{stats['worst_trade']:+.2f}%</b>\n"
        f"⏳ Открыто:      <b>{stats['open_trades']}</b>"
    )