import pandas as pd
import pandas_ta as ta
import ccxt
import time
from hf_storage import save_model_to_hub
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBClassifier

def fetch_mega_data(symbol="TON/USDT:USDT", timeframe="1h", limit=2000):
    print(f"📡 Сбор МЕГА-датасета для {symbol} (2000 свечей)...")
    exchange = ccxt.okx({'options': {'defaultType': 'swap'}})
    
    # 1. Загрузка TON (2000 свечей для глубокого обучения)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    
    # 2. Загрузка BTC
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
    except:
        df['fundingRate'] = 0.0

    # 4. Индикаторы (Упростили, чтобы не терять строки)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['MACD_Hist'] = ta.macd(df['Close']).iloc[:, 1]
    df['Vol_Change'] = df['Volume'].pct_change() * 100
    df['High_Low_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    
    # --- МЕТА-ЛЕЙБЛИНГ ЛОГИКА ---
    # 5. Определяем "Сигналы Учителя" (простая стратегия: RSI < 40)
    df['Primary_Signal'] = (df['RSI'] < 40).astype(int)
    
    # 6. Определяем ТАРГЕТ (успех сделки: +1.5% тейк при 1% стопе)
    # Ищем, что случится раньше в следующие 12 часов
    future_close = df['Close'].shift(-12)
    df['Target'] = 0
    # Упрощенный таргет для обучения: если цена через 8ч выше на 1.5%
    df.loc[(df['Close'].shift(-8) > df['Close'] * 1.015), 'Target'] = 1
    
    final_df = df.dropna()
    print(f"📊 Датасет готов! Строк: {len(final_df)}")
    return final_df

def train_meta_brain():
    df = fetch_mega_data()
    
    if len(df) < 200:
        print("❌ Данных всё еще мало. Попробуй позже.")
        return

    # Обучаем модель ТОЛЬКО на тех свечах, где Учитель дал сигнал
    # Модель учится фильтровать плохие входы Учителя
    train_df = df[df['Primary_Signal'] == 1].copy()
    
    if len(train_df) < 50:
        print("⚠️ Слишком мало сигналов от стратегии RSI. Обучаем на всём датасете.")
        train_df = df

    features = ['RSI', 'ATR', 'MACD_Hist', 'Vol_Change', 'High_Low_pct', 'BTC_pct_1h', 'fundingRate']
    X = train_df[features]
    y = train_df['Target']

    # Веса для балансировки
    scale = (len(y) - sum(y)) / sum(y) if sum(y) > 0 else 1

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric='logloss'
    )

    print(f"🚀 Обучение Критика на {len(train_df)} потенциальных сделках...")
    model.fit(X, y)
    
    # Сохранение
    save_model_to_hub(model, {"type": "meta_labeler", "features": features})
    print("✅ Мета-мозг готов и загружен!")

if __name__ == "__main__":
    train_meta_brain()