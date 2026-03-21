"""
backtest_engine.py — Бэктест стратегии на исторических данных.
Запускается каждые 12 часов из app.py
"""

import ccxt
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def fetch_history(symbol="TON/USDT",
                  timeframe="1h", limit=3000) -> pd.DataFrame:
    try:
        exchange = ccxt.okx({'options': {'defaultType': 'swap'}})
        ohlcv    = exchange.fetch_ohlcv(
            symbol + ":USDT", timeframe=timeframe, limit=limit
        )
        df = pd.DataFrame(
            ohlcv,
            columns=['ts', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        logger.info(f"[Backtest] ✅ Загружено {len(df)} свечей")
        return df
    except Exception as e:
        logger.error(f"[Backtest] ❌ Ошибка: {e}")
        return pd.DataFrame()


def _calc_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calc_atr(df, period=14):
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def run_backtest(
    symbol        = "TON/USDT",
    timeframe     = "1h",
    limit         = 3000,
    tp_pct        = 0.03,
    sl_pct        = 0.015,
    trade_pct     = 0.10,
    start_balance = 600.0
) -> dict:

    logger.info(f"[Backtest] 🔍 Старт: {limit} свечей...")

    df = fetch_history(symbol, timeframe, limit)
    if df.empty:
        return {"success": False, "error": "Нет данных"}

    # Индикаторы
    df['RSI']   = _calc_rsi(df['Close'])
    df['ATR']   = _calc_atr(df)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df = df.dropna()

    balance      = start_balance
    trades       = []
    in_trade     = False
    trade_signal = ""
    trade_open   = 0.0
    tp_price     = 0.0
    sl_price     = 0.0
    amount_usd   = 0.0

    for i in range(1, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # Закрытие
        if in_trade:
            hit = None
            if trade_signal == "BUY":
                if row['High'] >= tp_price:
                    hit = "TP"
                elif row['Low'] <= sl_price:
                    hit = "SL"
            else:
                if row['Low'] <= tp_price:
                    hit = "TP"
                elif row['High'] >= sl_price:
                    hit = "SL"

            if hit:
                cp = tp_price if hit == "TP" else sl_price
                if trade_signal == "BUY":
                    pnl_pct = (cp - trade_open) / trade_open * 100
                else:
                    pnl_pct = (trade_open - cp) / trade_open * 100

                pnl_usd  = amount_usd * pnl_pct / 100
                balance += pnl_usd

                trades.append({
                    "signal": trade_signal,
                    "result": "WIN" if hit == "TP" else "LOSS",
                    "pnl_pct": round(pnl_pct, 2),
                    "pnl_usd": round(pnl_usd, 2),
                })
                in_trade = False

        # Открытие
        if not in_trade:
            signal = None

            # BUY: RSI выходит из зоны перепроданности + EMA20 > EMA50
            if (prev['RSI'] < 35 and row['RSI'] >= 35
                    and row['EMA20'] > row['EMA50']):
                signal = "BUY"

            # SELL: RSI выходит из зоны перекупленности + EMA20 < EMA50
            elif (prev['RSI'] > 65 and row['RSI'] <= 65
                    and row['EMA20'] < row['EMA50']):
                signal = "SELL"

            if signal:
                amount_usd   = balance * trade_pct
                trade_open   = row['Close']
                trade_signal = signal
                in_trade     = True

                if signal == "BUY":
                    tp_price = trade_open * (1 + tp_pct)
                    sl_price = trade_open * (1 - sl_pct)
                else:
                    tp_price = trade_open * (1 - tp_pct)
                    sl_price = trade_open * (1 + sl_pct)

    # Расчёт результатов
    total    = len(trades)
    wins     = sum(1 for t in trades if t["result"] == "WIN")
    losses   = total - wins
    winrate  = round(wins / total * 100, 1) if total > 0 else 0
    pnl_list = [t["pnl_pct"] for t in trades]
    avg_pnl  = round(sum(pnl_list) / len(pnl_list), 2) if pnl_list else 0
    total_pnl   = round(balance - start_balance, 2)
    growth_pct  = round(total_pnl / start_balance * 100, 2)

    # Максимальная просадка
    peak      = start_balance
    max_dd    = 0.0
    running_b = start_balance
    for t in trades:
        running_b += t["pnl_usd"]
        if running_b > peak:
            peak = running_b
        dd = (peak - running_b) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    result = {
        "success":       True,
        "symbol":        symbol,
        "candles":       limit,
        "total_trades":  total,
        "wins":          wins,
        "losses":        losses,
        "winrate":       winrate,
        "avg_pnl":       avg_pnl,
        "total_pnl":     total_pnl,
        "growth_pct":    growth_pct,
        "max_drawdown":  round(max_dd, 2),
        "final_balance": round(balance, 2),
        "start_balance": start_balance,
    }

    logger.info(
        f"[Backtest] ✅ {total} сделок | "
        f"Winrate: {winrate}% | Рост: {growth_pct:+.2f}%"
    )
    return result


def format_backtest_message(r: dict) -> str:
    if not r.get("success"):
        return f"❌ Бэктест не удался: {r.get('error')}"

    emoji = "📈" if r["growth_pct"] >= 0 else "📉"
    return (
        f"🔬 <b>Бэктест завершён</b> {emoji}\n\n"
        f"📊 Свечей:        <b>{r['candles']}</b>\n"
        f"📋 Сделок:        <b>{r['total_trades']}</b>\n"
        f"✅ Побед:         <b>{r['wins']}</b>\n"
        f"❌ Поражений:     <b>{r['losses']}</b>\n"
        f"🎯 Winrate:       <b>{r['winrate']}%</b>\n\n"
        f"💰 Старт:         <b>${r['start_balance']:.2f}</b>\n"
        f"💰 Финиш:         <b>${r['final_balance']:.2f}</b>\n"
        f"📈 Рост:          <b>{r['growth_pct']:+.2f}%</b>\n"
        f"💵 P&L:           <b>${r['total_pnl']:+.2f}</b>\n\n"
        f"📊 Средний P&L:   <b>{r['avg_pnl']:+.2f}%</b>\n"
        f"📉 Макс просадка: <b>{r['max_drawdown']:.2f}%</b>"
    )