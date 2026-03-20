"""
Архив сделок в Google Sheets
Сохраняет все сигналы и результаты для анализа
"""

import os
import json
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials

# Области доступа Google API
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

# Заголовки таблицы
HEADERS = [
    "Дата",
    "Время",
    "Символ",
    "Сигнал",
    "Цена входа",
    "Уверенность %",
    "Настроение",
    "Стоп-лосс",
    "Тейк-профит",
    "Результат",
    "Прибыль/Убыток %",
    "Закрыто по",
    "Примечание"
]


def _get_client():
    """Создаёт подключение к Google Sheets"""
    creds_json = os.getenv("GOOGLE_CREDS_JSON", "")
    if not creds_json:
        raise ValueError("GOOGLE_CREDS_JSON не задан в переменных окружения")

    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client


def _get_sheet():
    """Открывает таблицу и возвращает первый лист"""
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID не задан в переменных окружения")

    client = _get_client()
    spreadsheet = client.open_by_key(sheet_id)

    # Берём первый лист или создаём "Архив"
    try:
        sheet = spreadsheet.worksheet("Архив")
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title="Архив", rows=10000, cols=20)
        # Добавляем заголовки
        sheet.append_row(HEADERS)
        # Форматируем заголовок (жирный)
        sheet.format("A1:M1", {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.2, "green": 0.4, "blue": 0.8}
        })

    return sheet


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
    """
    Записывает новый сигнал в таблицу.
    Возвращает True если успешно.
    """
    try:
        sheet = _get_sheet()
        now = datetime.utcnow()

        row = [
            now.strftime("%Y-%m-%d"),          # Дата
            now.strftime("%H:%M:%S") + " UTC", # Время
            symbol,                             # Символ
            signal,                             # Сигнал (BUY/SELL/HOLD)
            round(price, 6),                    # Цена входа
            round(confidence * 100, 1),         # Уверенность %
            sentiment,                          # Настроение
            round(stop_loss, 6) if stop_loss else "",    # Стоп-лосс
            round(take_profit, 6) if take_profit else "", # Тейк-профит
            "ОТКРЫТА",                          # Результат (обновим позже)
            "",                                 # Прибыль/Убыток
            "",                                 # Закрыто по
            note                                # Примечание
        ]

        sheet.append_row(row)
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
    """
    Обновляет результат последней открытой сделки.
    result: "ПРИБЫЛЬ" / "УБЫТОК" / "БЕЗУБЫТОК"
    closed_by: "TP" / "SL" / "MANUAL" / "SIGNAL"
    """
    try:
        sheet = _get_sheet()
        all_rows = sheet.get_all_values()

        # Ищем последнюю строку с ценой входа и статусом "ОТКРЫТА"
        for i in range(len(all_rows) - 1, 0, -1):
            row = all_rows[i]
            if len(row) >= 10 and row[4] == str(price_entry) and row[9] == "ОТКРЫТА":
                row_num = i + 1  # gspread считает с 1
                sheet.update_cell(row_num, 10, result)
                sheet.update_cell(row_num, 11, round(pnl_pct, 2))
                sheet.update_cell(row_num, 12, closed_by)
                print(f"[Archive] ✅ Обновлён результат: {result} {pnl_pct:+.2f}%")
                return True

        print("[Archive] ⚠️ Открытая сделка не найдена для обновления")
        return False

    except Exception as e:
        print(f"[Archive] ❌ Ошибка обновления результата: {e}")
        return False


def get_statistics() -> dict:
    """
    Считает статистику из таблицы.
    Возвращает словарь с метриками.
    """
    try:
        sheet = _get_sheet()
        all_rows = sheet.get_all_values()

        if len(all_rows) <= 1:
            return {"total": 0, "wins": 0, "losses": 0, "winrate": 0, "avg_pnl": 0}

        total = wins = losses = 0
        pnl_list = []

        for row in all_rows[1:]:  # Пропускаем заголовок
            if len(row) >= 11 and row[9] in ("ПРИБЫЛЬ", "УБЫТОК", "БЕЗУБЫТОК"):
                total += 1
                if row[9] == "ПРИБЫЛЬ":
                    wins += 1
                elif row[9] == "УБЫТОК":
                    losses += 1
                try:
                    pnl_list.append(float(row[10]))
                except:
                    pass

        winrate = round(wins / total * 100, 1) if total > 0 else 0
        avg_pnl = round(sum(pnl_list) / len(pnl_list), 2) if pnl_list else 0

        return {
            "total":   total,
            "wins":    wins,
            "losses":  losses,
            "winrate": winrate,
            "avg_pnl": avg_pnl
        }

    except Exception as e:
        print(f"[Archive] ❌ Ошибка получения статистики: {e}")
        return {"total": 0, "wins": 0, "losses": 0, "winrate": 0, "avg_pnl": 0}


if __name__ == "__main__":
    # Тест подключения
    print("Тестирование подключения к Google Sheets...")
    stats = get_statistics()
    print(f"Статистика: {stats}")