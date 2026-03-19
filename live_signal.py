import ccxt
import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
from datetime import datetime
from hf_storage import load_model_from_hub
from telegram_notify import send_telegram_message

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------
# Функция для получения 4-часовых признаков (без изменений)
# ----------------------------------------------------------------------
def get_4h_features(symbol='TON/USDT', limit=100):
    logging.info("Начинаем получение 4h данных...")
    try:
        exchange = ccxt.okx({'timeout': 30000})
        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=limit)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_4h['Timestamp'] = pd.to_datetime(df_4h['Timestamp'], unit='ms')
        df_4h.set_index('Timestamp', inplace=True)

        df_4h['EMA50_4h'] = ta.ema(df_4h['Close'], length=50)
        df_4h['RSI_4h'] = ta.rsi(df_4h['Close'], length=14)
        df_4h['ATR_4h'] = ta.atr(df_4h['High'], df_4h['Low'], df_4h['Close'], length=14)

        macd_4h = ta.macd(df_4h['Close'])
        macdh_col = None
        for col in macd_4h.columns:
            if 'MACDh' in col.upper():
                macdh_col = col
                break
        if macdh_col is None:
            df_4h['MACD_Hist_4h'] = 0.0
            logging.warning("Колонка MACDh не найдена, используется 0")
        else:
            df_4h['MACD_Hist_4h'] = macd_4h[macdh_col]

        df_4h.dropna(inplace=True)
        return df_4h[['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h']]
    except Exception as e:
        logging.error(f"Ошибка получения 4h данных: {e}")
        return pd.DataFrame(columns=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])


# ----------------------------------------------------------------------
# Функция для получения свежих данных BTC (процентные изменения)
# ----------------------------------------------------------------------
def get_btc_context_live(limit=100):
    """Получает свежие данные по BTC для расчёта BTC_pct_1h и BTC_pct_4h"""
    try:
        exchange = ccxt.okx()
        btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=limit)
        df_btc = pd.DataFrame(btc_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_btc['Timestamp'] = pd.to_datetime(df_btc['Timestamp'], unit='ms')
        df_btc.set_index('Timestamp', inplace=True)

        df_btc['BTC_pct_1h'] = df_btc['Close'].pct_change() * 100
        df_btc['BTC_pct_4h'] = df_btc['Close'].pct_change(4) * 100
        return df_btc[['BTC_pct_1h', 'BTC_pct_4h']].dropna()
    except Exception as e:
        logging.error(f"Ошибка получения BTC данных: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# Главная функция подготовки признаков для предсказания
# ----------------------------------------------------------------------
def prepare_realtime_features(df_1h_raw, df_4h_features):
    """
    Принимает:
      df_1h_raw - DataFrame с сырыми 1-часовыми свечами (колонки: Open, High, Low, Close, Volume)
      df_4h_features - DataFrame с 4-часовыми признаками (EMA50_4h, RSI_4h, ATR_4h, MACD_Hist_4h)
    Возвращает DataFrame с одной строкой, содержащей все 18 признаков в нужном порядке.
    """
    # Создаём копию, чтобы не портить исходные данные
    df_1h = df_1h_raw.copy()

    # --- 1. Расчёт индикаторов на 1h ---
    df_1h['RSI'] = ta.rsi(df_1h['Close'], length=14)
    df_1h['ATR'] = ta.atr(df_1h['High'], df_1h['Low'], df_1h['Close'], length=14)

    bb = ta.bbands(df_1h['Close'], length=20, std=2)
    bbl_col = [col for col in bb.columns if 'BBL' in col.upper()][0]
    df_1h['BB_Dist_Lower'] = (bb[bbl_col] - df_1h['Close']) / df_1h['Close'] * 100

    macd = ta.macd(df_1h['Close'])
    macd_col = [col for col in macd.columns if 'MACDH' in col.upper()][0]
    df_1h['MACD_Hist'] = macd[macd_col]

    df_1h['Vol_Change'] = df_1h['Volume'].pct_change() * 100
    df_1h['Price_Change_3h'] = df_1h['Close'].pct_change(3) * 100

    # --- 2. Дополнительные индикаторы 1h (новые) ---
    df_1h['EMA20'] = ta.ema(df_1h['Close'], length=20)
    df_1h['EMA50'] = ta.ema(df_1h['Close'], length=50)
    df_1h['RSI7'] = ta.rsi(df_1h['Close'], length=7)
    df_1h['Volume_SMA5'] = df_1h['Volume'].rolling(window=5).mean()
    df_1h['High_Low_pct'] = (df_1h['High'] - df_1h['Low']) / df_1h['Close'] * 100
    df_1h['Close_shift_1'] = df_1h['Close'].shift(1)

    # --- 3. Получение BTC-контекста ---
    df_btc = get_btc_context_live(limit=len(df_1h) + 10)

    # --- 4. Синхронизация временных индексов ---
    df_1h.index = pd.to_datetime(df_1h.index).as_unit('ns')
    df_4h_features.index = pd.to_datetime(df_4h_features.index).as_unit('ns')
    if not df_btc.empty:
        df_btc.index = pd.to_datetime(df_btc.index).as_unit('ns')

    # --- 5. Объединение всех признаков ---
    # Сначала мержим 1h и 4h
    df_merged = pd.merge_asof(
        df_1h.sort_index(),
        df_4h_features.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )
    # Затем добавляем BTC
    if not df_btc.empty:
        df_merged = pd.merge_asof(
            df_merged.sort_index(),
            df_btc.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
    else:
        # Если BTC недоступен, ставим нули
        df_merged['BTC_pct_1h'] = 0.0
        df_merged['BTC_pct_4h'] = 0.0

    # --- 6. Удаляем строки с NaN (первые из-за pct_change и shift) ---
    df_clean = df_merged.dropna().copy()

    # --- 7. Формируем финальный набор признаков в нужном порядке ---
    feature_names = [
        'RSI', 'ATR', 'BB_Dist_Lower', 'MACD_Hist', 'Vol_Change', 'Price_Change_3h',
        'EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h',
        'EMA20', 'EMA50', 'RSI7', 'Volume_SMA5', 'High_Low_pct', 'Close_shift_1',
        'BTC_pct_1h', 'BTC_pct_4h'
    ]

    # Берём самую свежую (последнюю) запись
    X = df_clean[feature_names].iloc[-1:]

    logging.info(f"✅ Признаки сформированы, всего признаков: {len(feature_names)}")
    return X


# ----------------------------------------------------------------------
# Основная функция, вызываемая из app.py (или напрямую)
# ----------------------------------------------------------------------
def get_signal():
    logging.info("=" * 50)
    logging.info("Начало работы get_signal()")

    # 1. Загружаем модель и метаданные
    try:
        model, metadata = load_model_from_hub()
        logging.info(f"✅ Модель загружена. Точность: {metadata.get('accuracy', 'N/A')}")
        atr_mean = metadata.get('atr_mean')
        if atr_mean:
            logging.info(f"📊 ATR из метаданных: {atr_mean:.4f}")
    except Exception as e:
        logging.error(f"❌ Не удалось загрузить модель: {e}")
        return None, None

    # 2. Получаем свежие 1-часовые свечи (сырые)
    try:
        exchange = ccxt.okx()
        ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=150)  # берём с запасом
        df_1h = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_1h['Timestamp'] = pd.to_datetime(df_1h['Timestamp'], unit='ms')
        df_1h.set_index('Timestamp', inplace=True)
        logging.info(f"✅ 1h данные получены, {len(df_1h)} свечей")
    except Exception as e:
        logging.error(f"❌ Ошибка получения 1h данных: {e}")
        return None, None

    # 3. Получаем 4-часовые признаки
    df_4h = get_4h_features()

    # 4. Формируем признаки для предсказания
    X = prepare_realtime_features(df_1h, df_4h)

    # 5. Выполняем предсказание
    try:
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        logging.info(f"📊 Предсказание: {pred} (1 – сигнал), вероятность: {prob:.4f}")
    except Exception as e:
        logging.error(f"❌ Ошибка при предсказании: {e}")
        return None, None

    # 6. Отправка сигнала в Telegram (если условия выполнены)
    if pred == 1 and prob > 0.6:
        current_price = df_1h['Close'].iloc[-1]
        msg = (f"🚀 <b>СИГНАЛ НА ПОКУПКУ TON/USDT</b>\n"
               f"Цена входа: {current_price:.4f}\n"
               f"Вероятность: {prob:.2%}\n"
               f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if atr_mean:
            sl = current_price - 2 * atr_mean
            tp = current_price + 3 * atr_mean
            msg += f"\n\n📉 Стоп-лосс: {sl:.4f} (-{2*atr_mean/current_price*100:.2f}%)\n"
            msg += f"📈 Тейк-профит: {tp:.4f} (+{3*atr_mean/current_price*100:.2f}%)"
        send_result = send_telegram_message(msg)
        if send_result:
            logging.info("✅ Сигнал отправлен в Telegram")
        else:
            logging.error("❌ Не удалось отправить сигнал в Telegram")
    else:
        logging.info("⏺️ Условия не выполнены (нет сигнала)")

    logging.info("🏁 Завершение get_signal()")
    logging.info("=" * 50)
    return pred, prob


# ----------------------------------------------------------------------
# Для тестового запуска из командной строки
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p, pr = get_signal()
    if p is not None:
        signal = "📈 ПОКУПКА" if p == 1 else "📉 НЕТ СИГНАЛА"
        print(f"{signal} | Вероятность: {pr:.2%}")