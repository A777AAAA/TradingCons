"""
Трекер результатов сделок
Отслеживает открытые позиции и фиксирует результаты
"""

import os
import json
import time
from datetime import datetime
from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT
from trade_archive import log_signal, update_result

# Файл для хранения текущей открытой позиции (в памяти)
_current_position = {
    "active": False,
    "symbol": "",
    "signal": "",
    "price_entry": 0.0,
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "confidence": 0.0,
    "sentiment": "neutral",
    "opened_at": ""
}


def open_position(
    symbol: str,
    signal: str,
    price: float,
    confidence: float,
    sentiment: str = "neutral",
    note: str = ""
) -> bool:
    """
    Открывает новую позицию и записывает в архив.
    signal: "BUY" или "SELL"
    """
    global _current_position

    if _current_position["active"]:
        print(f"[Tracker] ⚠️ Позиция уже открыта! Закрой текущую сначала.")
        return False

    # Считаем уровни
    if signal == "BUY":
        stop_loss   = round(price * (1 - STOP_LOSS_PCT), 6)
        take_profit = round(price * (1 + TAKE_PROFIT_PCT), 6)
    elif signal == "SELL":
        stop_loss   = round(price * (1 + STOP_LOSS_PCT), 6)
        take_profit = round(price * (1 - TAKE_PROFIT_PCT), 6)
    else:
        print(f"[Tracker] ⚠️ Неизвестный сигнал: {signal}")
        return False

    # Сохраняем позицию в памяти
    _current_position.update({
        "active":      True,
        "symbol":      symbol,
        "signal":      signal,
        "price_entry": price,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "confidence":  confidence,
        "sentiment":   sentiment,
        "opened_at":   datetime.utcnow().isoformat()
    })

    # Записываем в Google Sheets
    log_signal(
        symbol=symbol,
        signal=signal,
        price=price,
        confidence=confidence,
        sentiment=sentiment,
        stop_loss=stop_loss,
        take_profit=take_profit,
        note=note
    )

    print(f"[Tracker] ✅ Позиция открыта: {signal} {symbol} @ {price}")
    print(f"[Tracker]    SL: {stop_loss} | TP: {take_profit}")
    return True


def check_position(current_price: float) -> dict:
    """
    Проверяет текущую позицию по текущей цене.
    Возвращает статус: "TP" / "SL" / "OPEN" / "NONE"
    """
    global _current_position

    if not _current_position["active"]:
        return {"status": "NONE"}

    signal      = _current_position["signal"]
    price_entry = _current_position["price_entry"]
    stop_loss   = _current_position["stop_loss"]
    take_profit = _current_position["take_profit"]

    hit_tp = hit_sl = False

    if signal == "BUY":
        hit_tp = current_price >= take_profit
        hit_sl = current_price <= stop_loss
    elif signal == "SELL":
        hit_tp = current_price <= take_profit
        hit_sl = current_price >= stop_loss

    if hit_tp:
        pnl = TAKE_PROFIT_PCT * 100
        _close_position(current_price, "ПРИБЫЛЬ", pnl, "TP")
        return {"status": "TP", "pnl": pnl}

    elif hit_sl:
        pnl = -STOP_LOSS_PCT * 100
        _close_position(current_price, "УБЫТОК", pnl, "SL")
        return {"status": "SL", "pnl": pnl}

    # Позиция ещё открыта — считаем текущий PnL
    if signal == "BUY":
        current_pnl = (current_price - price_entry) / price_entry * 100
    else:
        current_pnl = (price_entry - current_price) / price_entry * 100

    return {
        "status":      "OPEN",
        "pnl":         round(current_pnl, 2),
        "price_entry": price_entry,
        "current":     current_price,
        "signal":      signal
    }


def close_position_manual(current_price: float, reason: str = "MANUAL") -> dict:
    """
    Принудительно закрывает позицию (например при новом сигнале).
    """
    global _current_position

    if not _current_position["active"]:
        return {"status": "NONE"}

    signal      = _current_position["signal"]
    price_entry = _current_position["price_entry"]

    if signal == "BUY":
        pnl = (current_price - price_entry) / price_entry * 100
    else:
        pnl = (price_entry - current_price) / price_entry * 100

    result = "ПРИБЫЛЬ" if pnl > 0 else ("УБЫТОК" if pnl < 0 else "БЕЗУБЫТОК")
    _close_position(current_price, result, pnl, reason)

    return {"status": "CLOSED", "pnl": round(pnl, 2), "result": result}


def _close_position(price: float, result: str, pnl: float, closed_by: str):
    """Внутренняя функция закрытия позиции"""
    global _current_position

    price_entry = _current_position["price_entry"]

    # Обновляем Google Sheets
    update_result(
        price_entry=price_entry,
        result=result,
        pnl_pct=round(pnl, 2),
        closed_by=closed_by
    )

    print(f"[Tracker] 🔒 Позиция закрыта: {result} {pnl:+.2f}% по {closed_by}")

    # Сбрасываем позицию
    _current_position.update({
        "active":      False,
        "symbol":      "",
        "signal":      "",
        "price_entry": 0.0,
        "stop_loss":   0.0,
        "take_profit": 0.0,
        "confidence":  0.0,
        "sentiment":   "neutral",
        "opened_at":   ""
    })


def get_position_status() -> dict:
    """Возвращает текущий статус позиции"""
    return _current_position.copy()


def has_open_position() -> bool:
    """Проверяет есть ли открытая позиция"""
    return _current_position["active"]


if __name__ == "__main__":
    # Тест
    print("Тест трекера позиций...")
    print(f"Открытая позиция: {has_open_position()}")

    # Симуляция открытия
    open_position(
        symbol="TON/USDT",
        signal="BUY",
        price=5.234,
        confidence=0.72,
        sentiment="bullish",
        note="Тест"
    )

    # Симуляция проверки
    result = check_position(current_price=5.50)
    print(f"Результат проверки: {result}")