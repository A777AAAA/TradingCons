import pandas as pd
import numpy as np
import ccxt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

MODEL_PATH = "ai_brain.pkl"


# ----------------------------------------------------------------------
# Индикаторы вручную (как в live_signal.py)
# ----------------------------------------------------------------------
def calc_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast    = calc_ema(series, fast)
    ema_slow    = calc_ema(series, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line - signal_line


# ----------------------------------------------------------------------
# Сбор данных
# ----------------------------------------------------------------------
def fetch_mega_data(symbol="TON/USDT:USDT", timeframe="1h", limit=2000):
    print(f"📡 Сбор данных для {symbol} (2000 свечей)...")
    exchange = ccxt.okx({'options': {'defaultType': 'swap'}})

    # 1. TON данные
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    # 2. BTC корреляция
    btc_ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe=timeframe, limit=limit)
    df_btc = pd.DataFrame(btc_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_btc['Timestamp'] = pd.to_datetime(df_btc['Timestamp'], unit='ms')
    df_btc.set_index('Timestamp', inplace=True)
    df['BTC_pct_1h'] = df_btc['Close'].pct_change() * 100

    # 3. Funding Rate
    try:
        funding = exchange.fetch_funding_rate_history(symbol, limit=limit)
        df_f = pd.DataFrame(funding, columns=['timestamp', 'fundingRate'])
        df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], unit='ms')
        df_f.set_index('timestamp', inplace=True)
        df = df.join(df_f[['fundingRate']], how='left').ffill().fillna(0)
    except Exception:
        df['fundingRate'] = 0.0

    # 4. Индикаторы (вручную — без pandas_ta)
    df['RSI']        = calc_rsi(df['Close'], 14)
    df['ATR']        = calc_atr(df['High'], df['Low'], df['Close'], 14)
    df['EMA20']      = calc_ema(df['Close'], 20)
    df['MACD_Hist']  = calc_macd_hist(df['Close'])
    df['Vol_Change'] = df['Volume'].pct_change() * 100
    df['High_Low_pct'] = (df['High'] - df['Low']) / df['Close'] * 100

    # 5. Сигнал учителя (RSI < 40)
    df['Primary_Signal'] = (df['RSI'] < 40).astype(int)

    # 6. Таргет: цена через 8 часов выше на 1.5%
    df['Target'] = 0
    df.loc[(df['Close'].shift(-8) > df['Close'] * 1.015), 'Target'] = 1

    final_df = df.dropna()
    print(f"📊 Датасет готов! Строк: {len(final_df)}")
    return final_df


# ----------------------------------------------------------------------
# Обучение модели
# ----------------------------------------------------------------------
def train_model():
    """
    Обучает модель и сохраняет локально.
    Возвращает dict с результатами для weekly_retrainer.
    """
    try:
        df = fetch_mega_data()

        if len(df) < 200:
            return {"success": False, "error": "Мало данных"}

        # Фильтруем по сигналу учителя
        train_df = df[df['Primary_Signal'] == 1].copy()
        if len(train_df) < 50:
            print("⚠️ Мало сигналов RSI — обучаем на всём датасете")
            train_df = df

        features = [
            'RSI', 'ATR', 'MACD_Hist', 'Vol_Change',
            'High_Low_pct', 'BTC_pct_1h', 'fundingRate'
        ]
        X = train_df[features]
        y = train_df['Target']

        # Разбивка train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Балансировка классов
        pos  = sum(y_train)
        neg  = len(y_train) - pos
        scale = neg / pos if pos > 0 else 1

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=scale,
            eval_metric='logloss'
        )

        print(f"🚀 Обучение на {len(X_train)} примерах...")
        model.fit(X_train, y_train)

        # Метрики
        y_pred    = model.predict(X_test)
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)

        print(f"✅ Точность:  {accuracy:.1%}")
        print(f"✅ Precision: {precision:.1%}")
        print(f"✅ Recall:    {recall:.1%}")

        # Сохраняем локально
        joblib.dump(model, MODEL_PATH)
        print(f"💾 Модель сохранена: {MODEL_PATH}")

        return {
            "success":   True,
            "model":     model,
            "scaler":    None,
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "n_samples": len(X_train),
        }

    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    train_model()