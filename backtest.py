import pandas as pd
import matplotlib.pyplot as plt
from hf_storage import load_model_from_hub

def run_advanced_backtest():
    print("📈 Запуск продвинутого бэктеста: Динамический риск-менеджмент...")
    
    # 1. Загрузка данных и модели
    try:
        model, metadata = load_model_from_hub()
        df = pd.read_csv("ml_ready_ton_data_v2.csv", index_col='Timestamp', parse_dates=True)
        print(f"✅ Данные загружены. Записей: {len(df)}")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    # 2. Список признаков
    features = [
        'RSI', 'ATR', 'BB_Dist_Lower', 'MACD_Hist', 'Vol_Change', 'Price_Change_3h',
        'EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h',
        'EMA20', 'EMA50', 'RSI7', 'Volume_SMA5', 'High_Low_pct', 'Close_shift_1',
        'BTC_pct_1h', 'BTC_pct_4h'
    ]
    
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"❌ В данных не хватает колонок: {missing}")
        return

    test_size = int(len(df) * 0.3) # Берем 30% данных для более длинного теста
    if test_size < 1: test_size = len(df)
    
    test_df = df.tail(test_size).copy()
    X_test = test_df[features]
    
    # 3. Предсказания модели
    print("🧠 Нейросеть генерирует вероятности...")
    probs = model.predict_proba(X_test)[:, 1]
    test_df['Signal_Prob'] = probs

    # 4. Параметры торговой симуляции
    initial_balance = 100.0
    balance = initial_balance
    commission = 0.001 # 0.1% комиссия тейкера на OKX
    
    position = 0 # 0 - нет позиции, 1 - в лонге
    entry_price = 0
    position_size_usdt = 0
    take_profit = 0
    stop_loss = 0
    
    history = [initial_balance]
    winning_trades = 0
    losing_trades = 0

    print(f"💰 Старт симуляции. Баланс: {initial_balance} USDT")

    for i in range(len(test_df)):
        current_close = test_df['Close'].iloc[i]
        current_high = test_df['High'].iloc[i]
        current_low = test_df['Low'].iloc[i]
        current_atr = test_df['ATR'].iloc[i]
        prob = test_df['Signal_Prob'].iloc[i]
        
        # --- ЛОГИКА ВЫХОДА (Если уже в позиции) ---
        if position == 1:
            # Проверяем, зацепила ли цена Stop Loss внутри свечи (Low <= SL)
            if current_low <= stop_loss:
                loss_pct = (stop_loss - entry_price) / entry_price
                balance_change = position_size_usdt * loss_pct
                balance += balance_change - (position_size_usdt * commission)
                position = 0
                history.append(balance)
                losing_trades += 1
                print(f"🔴 STOP LOSS: Убыток. Баланс: {balance:.2f} USDT")
                continue # Переходим к следующей свече
                
            # Проверяем, зацепила ли цена Take Profit внутри свечи (High >= TP)
            elif current_high >= take_profit:
                profit_pct = (take_profit - entry_price) / entry_price
                balance_change = position_size_usdt * profit_pct
                balance += balance_change - (position_size_usdt * commission)
                position = 0
                history.append(balance)
                winning_trades += 1
                print(f"🟢 TAKE PROFIT: Профит! Баланс: {balance:.2f} USDT")
                continue

        # --- ЛОГИКА ВХОДА (Если нет позиции) ---
        if position == 0:
            if prob > 0.75:
                risk_allocation = 0.30 # Ставим 30% от банка при высокой уверенности
            elif prob > 0.60:
                risk_allocation = 0.15 # Ставим 15% при средней уверенности
            else:
                risk_allocation = 0 # Игнорируем слабые сигналы
                
            if risk_allocation > 0:
                entry_price = current_close
                position_size_usdt = balance * risk_allocation
                
                # Умные цели на основе ATR (множители можно менять)
                take_profit = entry_price + (current_atr * 2.5) # Тейк = 2.5 средних размаха свечи
                stop_loss = entry_price - (current_atr * 1.5)   # Стоп = 1.5 размаха
                
                # Вычитаем комиссию за вход
                balance -= (position_size_usdt * commission)
                position = 1
                print(f"⚡ ВХОД: Цена {entry_price:.4f} | Риск {risk_allocation*100}% | TP: {take_profit:.4f} | SL: {stop_loss:.4f}")

    # Закрываем позицию принудительно в конце теста, если она осталась
    if position == 1:
        profit_pct = (test_df['Close'].iloc[-1] - entry_price) / entry_price
        balance += (position_size_usdt * profit_pct) - (position_size_usdt * commission)
        history.append(balance)

    # 5. Итоги
    total_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((balance - initial_balance) / initial_balance) * 100
    
    print(f"\n=== ИТОГИ БЭКТЕСТА ===")
    print(f"🏁 Финальный баланс: {balance:.2f} USDT ({total_return:.2f}%)")
    print(f"📊 Всего сделок: {total_trades}")
    print(f"🏆 Win Rate (Успешные): {win_rate:.1f}%")
    print(f"✅ Плюсовых: {winning_trades} | ❌ Минусовых: {losing_trades}")
    
    if len(history) > 1:
        plt.figure(figsize=(10, 5))
        plt.plot(history, color='blue', linewidth=2)
        plt.title("Эквити (Кривая капитала) с Динамическим Риском")
        plt.xlabel("Сделки")
        plt.ylabel("USDT")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    run_advanced_backtest()