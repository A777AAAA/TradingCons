"""
Архив сделок — локальное хранение в JSON файле
(Замена Google Sheets — не нужен gspread и google-auth)
"""

import os
import json
from datetime import datetime

ARCHIVE_FILE = "trade_archive.json"


def _load_archive() -> list:
    """Загружает архив из файла"""
    if not os.path.exists(ARCHIVE_FILE):
        return []
    try:
        with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Archive] ⚠️ Ошибка загрузки архива: {e}")
        return []


def _save_archive(data: list) -> bool:
    """Сохраняет архив в файл"""
    try:
        with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[Archive] ❌ Ошибка сохранения архива: {e}")
        return False


def log_signal(
    symbol: str,
    signal: str,
    price: float,
    confidence: float,
    sentiment: str = "neutral",
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    note: str = ""
) -> bool:
    """Записывает новый сигнал в архив."""
    try:
        archive = _load_archive()
        now     = datetime.utcnow()

        entry = {
            "date":        now.strftime("%Y-%m-%d"),
            "time":        now.strftime("%H:%M:%S") + " UTC",
            "symbol":      symbol,
            "signal":      signal,
            "price_entry": round(price, 6),
            "confidence":  round(confidence * 100, 1),
            "sentiment":   sentiment,
            "stop_loss":   round(stop_loss, 6) if stop_loss else 0.0,
            "take_profit": round(take_profit, 6) if take_profit else 0.0,
            "result":      "ОТКРЫТА",
            "pnl_pct":     None,
            "closed_by":   None,
            "note":        note
        }

        archive.append(entry)
        _save_archive(archive)
        print(f"[Archive] ✅ Записан сигнал: {signal} {symbol} @ {price}")
        return True

    except Exception as e:
        print(f"[Archive] ❌ Ошибка записи сигнала: {e}")
        return False


def update_result(
    price_entry: float,
    result: str,
    pnl_pct: float,
    closed_by: str = ""
) -> bool:
    """Обновляет результат последней открытой сделки."""
    try:
        archive = _load_archive()

        # Ищем последнюю открытую сделку с нужной ценой входа
        for i in range(len(archive) - 1, -1, -1):
            entry = archive[i]
            if (entry.get("result") == "ОТКРЫТА" and
                    abs(entry.get("price_entry", 0) - price_entry) < 0.000001):
                archive[i]["result"]    = result
                archive[i]["pnl_pct"]   = round(pnl_pct, 2)
                archive[i]["closed_by"] = closed_by
                _save_archive(archive)
                print(f"[Archive] ✅ Обновлён результат: {result} {pnl_pct:+.2f}%")
                return True

        print("[Archive] ⚠️ Открытая сделка не найдена")
        return False

    except Exception as e:
        print(f"[Archive] ❌ Ошибка обновления: {e}")
        return False


def get_statistics() -> dict:
    """Считает статистику из архива."""
    try:
        archive  = _load_archive()
        closed   = [e for e in archive if e.get("result") in ("ПРИБЫЛЬ", "УБЫТОК", "БЕЗУБЫТОК")]

        total    = len(closed)
        wins     = sum(1 for e in closed if e.get("result") == "ПРИБЫЛЬ")
        losses   = sum(1 for e in closed if e.get("result") == "УБЫТОК")
        pnl_list = [e["pnl_pct"] for e in closed if e.get("pnl_pct") is not None]

        winrate  = round(wins / total * 100, 1) if total > 0 else 0
        avg_pnl  = round(sum(pnl_list) / len(pnl_list), 2) if pnl_list else 0

        return {
            "total":   total,
            "wins":    wins,
            "losses":  losses,
            "winrate": winrate,
            "avg_pnl": avg_pnl
        }

    except Exception as e:
        print(f"[Archive] ❌ Ошибка статистики: {e}")
        return {"total": 0, "wins": 0, "losses": 0, "winrate": 0, "avg_pnl": 0}


if __name__ == "__main__":
    print("Тест архива...")
    stats = get_statistics()
    print(f"Статистика: {stats}")