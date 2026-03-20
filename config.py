"""
Центральный файл конфигурации TradeBot
Все настройки берутся из переменных окружения Render
"""

import os

# ═══════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════
# БИРЖА OKX
# ═══════════════════════════════════════════
OKX_API_KEY    = os.getenv("OKX_API_KEY", "")
OKX_SECRET     = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# ═══════════════════════════════════════════
# HUGGING FACE (хранение модели)
# ═══════════════════════════════════════════
HF_TOKEN   = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# ═══════════════════════════════════════════
# GOOGLE SHEETS (архив сделок)
# ═══════════════════════════════════════════
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON", "")
GOOGLE_SHEET_ID   = os.getenv("GOOGLE_SHEET_ID", "")

# ═══════════════════════════════════════════
# OPENROUTER (AI анализ настроений)
# ═══════════════════════════════════════════
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ═══════════════════════════════════════════
# ТОРГОВЫЕ ПАРАМЕТРЫ
# ═══════════════════════════════════════════
SYMBOL         = "TON/USDT"
TIMEFRAME      = "1h"
TRADE_AMOUNT   = float(os.getenv("TRADE_AMOUNT", "10.0"))  # USDT на сделку

# Риск менеджмент
STOP_LOSS_PCT  = 0.025   # 2.5% стоп-лосс
TAKE_PROFIT_PCT = 0.05   # 5.0% тейк-профит
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "30.0"))  # макс. убыток в день

# ═══════════════════════════════════════════
# РАСПИСАНИЕ
# ═══════════════════════════════════════════
SIGNAL_INTERVAL_MINUTES = 60   # проверка сигнала каждый час
RETRAIN_DAY = "sunday"         # день переобучения модели
RETRAIN_HOUR = 2               # час переобучения (02:00 UTC)

# ═══════════════════════════════════════════
# ПОРОГИ СИГНАЛОВ
# ═══════════════════════════════════════════
MIN_CONFIDENCE = 0.62          # минимальная уверенность для входа
STRONG_SIGNAL  = 0.75          # уверенность "сильного" сигнала


def validate_config() -> list:
    """
    Проверяет наличие обязательных переменных.
    Возвращает список отсутствующих переменных.
    """
    required = {
        "TELEGRAM_TOKEN":  TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
        "OKX_API_KEY":     OKX_API_KEY,
        "OKX_SECRET":      OKX_SECRET,
        "OKX_PASSPHRASE":  OKX_PASSPHRASE,
        "HF_TOKEN":        HF_TOKEN,
        "HF_REPO_ID":      HF_REPO_ID,
    }

    missing = [key for key, val in required.items() if not val]
    return missing


if __name__ == "__main__":
    missing = validate_config()
    if missing:
        print(f"⚠️  Отсутствуют переменные: {missing}")
    else:
        print("✅ Все переменные окружения на месте!")
    
    print(f"\n📊 Торговые параметры:")
    print(f"   Символ:      {SYMBOL}")
    print(f"   Таймфрейм:   {TIMEFRAME}")
    print(f"   Сумма:       {TRADE_AMOUNT} USDT")
    print(f"   Стоп-лосс:   {STOP_LOSS_PCT*100}%")
    print(f"   Тейк-профит: {TAKE_PROFIT_PCT*100}%")