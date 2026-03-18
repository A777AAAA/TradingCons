from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
from datetime import datetime
from sklearn.metrics import accuracy_score
from hf_storage import save_model_to_hub, update_historical_data, load_model_from_hub
import pandas as pd
import ccxt
import pandas_ta as ta
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

def tune_hyperparameters_xgb_pipeline(X, y):
    print("🔄 Запуск подбора гиперпараметров XGBoost с TimeSeriesSplit...")
    # Расчет веса классов для борьбы с дисбалансом
    scale_pos_weight = len(y[y==0]) / len(y[y==1]) if len(y[y==1]) > 0 else 1
    
    param_dist = {
        'xgb__n_estimators': randint(100, 500),
        'xgb__max_depth': randint(3, 10),
        'xgb__learning_rate': uniform(0.01, 0.3),
        'xgb__subsample': uniform(0.6, 0.4),
        'xgb__colsample_bytree': uniform(0.6, 0.4),
        'xgb__min_child_weight': randint(1, 10),
        'xgb__gamma': uniform(0, 0.5)
    }
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            n_jobs=-1
        ))
    ])
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=25,
        cv=tscv,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X, y)
    print(f"✅ Лучшие параметры XGBoost: {random_search.best_params_}")
    return random_search.best_estimator_, random_search.best_params_

def get_btc_context(limit=1000):
    print("🟠 Получаем контекст BTC/USDT...")
    try:
        exchange = ccxt.okx()
        btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=limit)
        df_btc = pd.DataFrame(btc_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_btc['Timestamp'] = pd.to_datetime(df_btc['Timestamp'], unit='ms')
        df_btc.set_index('Timestamp', inplace=True)
        
        df_btc['BTC_pct_1h'] = df_btc['Close'].pct_change() * 100
        df_btc['BTC_pct_4h'] = df_btc['Close'].pct_change(4) * 100
        return df_btc[['BTC_pct_1h', 'BTC_pct_4h']]
    except Exception as e:
        print(f"⚠️ Ошибка получения BTC данных: {e}")
        return pd.DataFrame()

def get_4h_features(symbol='TON/USDT', limit=500):
    exchange = ccxt.okx()
    ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=limit)
    df_4h = pd.DataFrame(ohlcv_4h, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_4h['Timestamp'] = pd.to_datetime(df_4h['Timestamp'], unit='ms')
    df_4h.set_index('Timestamp', inplace=True)
    
    df_4h['EMA50_4h'] = ta.ema(df_4h['Close'], length=50)
    df_4h['RSI_4h'] = ta.rsi(df_4h['Close'], length=14)
    df_4h['ATR_4h'] = ta.atr(df_4h['High'], df_4h['Low'], df_4h['Close'], length=14)
    
    macd_4h = ta.macd(df_4h['Close'])
    macdh_col = next((col for col in macd_4h.columns if 'MACDH' in col.upper()), None)
    df_4h['MACD_Hist_4h'] = macd_4h[macdh_col] if macdh_col is not None else 0.0
    
    df_4h.dropna(inplace=True)
    return df_4h[['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h']]

def update_and_train():
    print("🔄 Запуск модуля самообучения (Data Loop)...")
    file_name = "ml_ready_ton_data_v2.csv"
    
    # 1. Загрузка старых данных
    if os.path.exists(file_name):
        df_old = pd.read_csv(file_name, index_col='Timestamp', parse_dates=True)
        # Очищаем колонки, которые будем пересчитывать
        cols_to_drop = ['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h', 'BTC_pct_1h', 'BTC_pct_4h']
        df_old = df_old.drop(columns=[c for c in cols_to_drop if c in df_old.columns])
    else:
        df_old = pd.DataFrame()

    # 2. Получение новых данных TON
    exchange = ccxt.okx()
    ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=100)
    df_new = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'], unit='ms')
    df_new.set_index('Timestamp', inplace=True)

    update_historical_data(df_new) 
    
    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()

    # 3. Синхронизация времени и Мерж (Фикс ошибки Dtype)
    df_4h = get_4h_features()
    df_btc = get_btc_context(limit=len(df_combined)+100)

    # Принудительно переводим все индексы в наносекунды (ns)
    df_combined.index = pd.to_datetime(df_combined.index).as_unit('ns')
    
    if not df_4h.empty:
        df_4h.index = pd.to_datetime(df_4h.index).as_unit('ns')
        df_combined = pd.merge_asof(df_combined.sort_index(), df_4h.sort_index(), left_index=True, right_index=True, direction='backward')
        
    if not df_btc.empty:
        df_btc.index = pd.to_datetime(df_btc.index).as_unit('ns')
        df_combined = pd.merge_asof(df_combined.sort_index(), df_btc.sort_index(), left_index=True, right_index=True, direction='backward')

    # 4. Расчет технических индикаторов
    df_combined['RSI'] = ta.rsi(df_combined['Close'], length=14)
    df_combined['ATR'] = ta.atr(df_combined['High'], df_combined['Low'], df_combined['Close'], length=14)
    
    bb = ta.bbands(df_combined['Close'], length=20, std=2)
    bbl_col = [col for col in bb.columns if 'BBL' in col.upper()][0]
    df_combined['BB_Dist_Lower'] = (bb[bbl_col] - df_combined['Close']) / df_combined['Close'] * 100
    
    macd = ta.macd(df_combined['Close'])
    macd_col = [col for col in macd.columns if 'MACDH' in col.upper()][0]
    df_combined['MACD_Hist'] = macd[macd_col]
    
    df_combined['Vol_Change'] = df_combined['Volume'].pct_change() * 100
    df_combined['Price_Change_3h'] = df_combined['Close'].pct_change(3) * 100
    df_combined['EMA20'] = ta.ema(df_combined['Close'], length=20)
    df_combined['EMA50'] = ta.ema(df_combined['Close'], length=50)
    df_combined['RSI7'] = ta.rsi(df_combined['Close'], length=7)
    df_combined['Volume_SMA5'] = df_combined['Volume'].rolling(window=5).mean()
    df_combined['High_Low_pct'] = (df_combined['High'] - df_combined['Low']) / df_combined['Close'] * 100
    df_combined['Close_shift_1'] = df_combined['Close'].shift(1)

    # Таргет: рост > 0.8% через 8 часов
    df_combined['Future_Close'] = df_combined['Close'].shift(-8)
    df_combined['Target'] = ((df_combined['Future_Close'] - df_combined['Close']) / df_combined['Close'] > 0.008).astype(int)

    df_clean = df_combined.dropna().copy()
    df_clean.to_csv(file_name)

    features = [
        'RSI', 'ATR', 'BB_Dist_Lower', 'MACD_Hist', 'Vol_Change', 'Price_Change_3h',
        'EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h',
        'EMA20', 'EMA50', 'RSI7', 'Volume_SMA5', 'High_Low_pct', 'Close_shift_1',
        'BTC_pct_1h', 'BTC_pct_4h'
    ]
    
    X = df_clean[features]
    y = df_clean['Target']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 5. Проверка необходимости тюнинга
    try:
        _, metadata = load_model_from_hub()
    except Exception:
        metadata = {'last_tuning_date': None, 'best_params': None}

    today = datetime.now()
    need_tuning = False
    
    # Проверка на наличие старых параметров от RandomForest
    current_params = metadata.get('best_params', {})
    rf_keys = ['max_features', 'min_samples_leaf', 'min_samples_split']
    
    if metadata.get('last_tuning_date') is None or any(k in current_params for k in rf_keys):
        print("🧹 Обнаружены устаревшие параметры или их отсутствие. Запуск тюнинга XGBoost...")
        need_tuning = True
    else:
        days_since_tune = (today - datetime.fromisoformat(metadata['last_tuning_date'])).days
        if days_since_tune >= 7:
            need_tuning = True

    if need_tuning:
        model, best_params = tune_hyperparameters_xgb_pipeline(X_train, y_train)
        metadata['best_params'] = best_params
        metadata['last_tuning_date'] = today.isoformat()
    else:
        print(f"♻️ Используем сохраненные параметры. Тюнинг был {days_since_tune} дн. назад.")
        # Чистим ключи от префикса пайплайна
        params = {k.replace('xgb__', ''): v for k, v in metadata['best_params'].items()}
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
        
        model = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('xgb', xgb.XGBClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                n_jobs=-1,
                **params
            ))
        ])
        model.fit(X_train, y_train)

    # 6. Оценка и сохранение
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Настоящая точность XGBoost на тесте: {acc:.4f}")

    metadata['last_training'] = today.isoformat()
    metadata['accuracy'] = f"{acc:.4f}"
    metadata['atr_mean'] = float(df_clean['ATR'].mean())

    save_model_to_hub(model, metadata)
    joblib.dump(df_clean['ATR'].mean(), 'atr_mean.pkl')
    print("🚀 Модель успешно обучена и сохранена в облако!")

if __name__ == "__main__":
    update_and_train()