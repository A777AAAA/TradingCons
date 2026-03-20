"""
TradeBot - Главный файл запуска
Связывает все модули в единую систему
"""

import threading
import time
import traceback
from datetime import datetime
from flask import Flask

from config import (
    SYMBOL, TIMEFRAME, MIN_CONFIDENCE, STRONG_SIGNAL,
    SIGNAL_INTERVAL_MINUTES, validate_config
)
from live_signal import get_live_signal
from outcome_tracker import (
    open_position, check_position,
    close_position_manual, has_open_position,
    get_position_status
)
from sentiment_analyzer import get_market_sentiment, sentiment_to_signal_boost
from weekly_retrainer import run_retrainer_loop
from telegram_notify import send_message
from trade_archive import get_statistics


# ═══════════════════════════════════════════
# HEALTHCHECK СЕРВЕР (для Render)
# ═══════════════════════════════════════════
health_app = Flask(__name__)

@health_app.route("/health")
def health():
    return {"status": "ok", "bot": "TradeBot v2.0"}, 200

@health_app.route("/")
def index():
    stats = get_statistics()
    pos   = get_position_status()
    return {
        "bot":      "TradeBot v2.0",
        "symbol":   SYMBOL,
        "position": pos,
        "stats":    stats
    }, 200

def run_health_server():
    health_app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)


# ═══════════════════════════════════════════
# ПРОВЕРКА КОНФИГУРАЦИИ
# ═══════════════════════════════════════════
def startup_check():
    """Проверяет конфигурацию при запуске"""
    missing = validate_config()
    if missing:
        msg = "⚠️ Отсутствуют переменные окружения:\n" + "\n".join(missing)
        print(f"[App] {msg}")
        send_message(msg)
        return False

    print("[App] ✅ Конфигурация проверена")
    return True


# ═══════════════════════════════════════════
# ОСНОВНОЙ ТОРГОВЫЙ ЦИКЛ
# ═══════════════════════════════════════════
def trading_loop():
    """
    Основной цикл бота:
    1. Получает сигнал от ML модели
    2. Проверяет настроение рынка
    3. Открывает/закрывает позиции
    4. Отправляет уведомления в Telegram
    """
    print(f"[App] 🚀 Торговый цикл запущен | {SYMBOL} | {TIMEFRAME}")

    while True:
        try:
            now = datetime.utcnow().strftime("%H:%M UTC")
            print(f"\n[App] ⏰ Цикл: {now}")

            # ── Шаг 1: Получаем ML сигнал ──────────────────
            print("[App] Получение сигнала...")
            signal_data = get_live_signal()

            if not signal_data:
                print("[App] ⚠️ Сигнал не получен, пропускаем цикл")
                time.sleep(SIGNAL_INTERVAL_MINUTES * 60)
                continue

            signal     = signal_data.get("signal", "HOLD")
            confidence = signal_data.get("confidence", 0.0)
            price      = signal_data.get("price", 0.0)

            print(f"[App] 📊 Сигнал: {signal} | Уверенность: {confidence:.1%} | Цена: {price}")

            # ── Шаг 2: Проверяем открытую позицию ──────────
            if has_open_position():
                position_check = check_position(current_price=price)
                status = position_check.get("status")

                if status == "TP":
                    pnl = position_check.get("pnl", 0)
                    send_message(
                        f"✅ <b>ТЕЙК-ПРОФИТ!</b>\n"
                        f"💰 Прибыль: +{pnl:.2f}%\n"
                        f"💵 Цена закрытия: {price}"
                    )
                    print(f"[App] ✅ TP сработал: +{pnl:.2f}%")

                elif status == "SL":
                    pnl = position_check.get("pnl", 0)
                    send_message(
                        f"🛑 <b>СТОП-ЛОСС!</b>\n"
                        f"📉 Убыток: {pnl:.2f}%\n"
                        f"💵 Цена закрытия: {price}"
                    )
                    print(f"[App] 🛑 SL сработал: {pnl:.2f}%")

                elif status == "OPEN":
                    current_pnl = position_check.get("pnl", 0)
                    pos = get_position_status()
                    print(f"[App] 📈 Позиция открыта: {pos['signal']} | PnL: {current_pnl:+.2f}%")

                    # Если новый сигнал противоположный — закрываем
                    if signal == "SELL" and pos["signal"] == "BUY" and confidence >= MIN_CONFIDENCE:
                        close_result = close_position_manual(price, reason="SIGNAL")
                        send_message(
                            f"🔄 <b>Позиция закрыта по сигналу</b>\n"
                            f"📊 Результат: {close_result.get('result')}\n"
                            f"💰 PnL: {close_result.get('pnl', 0):+.2f}%"
                        )

                    elif signal == "BUY" and pos["signal"] == "SELL" and confidence >= MIN_CONFIDENCE:
                        close_result = close_position_manual(price, reason="SIGNAL")
                        send_message(
                            f"🔄 <b>Позиция закрыта по сигналу</b>\n"
                            f"📊 Результат: {close_result.get('result')}\n"
                            f"💰 PnL: {close_result.get('pnl', 0):+.2f}%"
                        )

            # ── Шаг 3: Открываем новую позицию ─────────────
            if not has_open_position() and signal in ("BUY", "SELL"):

                if confidence >= MIN_CONFIDENCE:

                    # Получаем настроение рынка
                    change_24h = signal_data.get("change_24h", 0.0)
                    volume     = signal_data.get("volume", 0.0)
                    sentiment  = get_market_sentiment(price, change_24h, volume)
                    sentiment_str = sentiment.get("sentiment", "neutral")

                    # Корректируем уверенность
                    boost = sentiment_to_signal_boost(sentiment, signal)
                    adjusted_confidence = min(confidence * boost, 0.99)

                    print(f"[App] 🧠 Настроение: {sentiment_str} | Буст: {boost:.2f}x")
                    print(f"[App] 📊 Скорректированная уверенность: {adjusted_confidence:.1%}")

                    if adjusted_confidence >= MIN_CONFIDENCE:
                        # Определяем силу сигнала
                        strength = "🔥 СИЛЬНЫЙ" if adjusted_confidence >= STRONG_SIGNAL else "📊 Обычный"

                        # Открываем позицию
                        opened = open_position(
                            symbol=SYMBOL,
                            signal=signal,
                            price=price,
                            confidence=adjusted_confidence,
                            sentiment=sentiment_str,
                            note=f"Буст: {boost:.2f}x"
                        )

                        if opened:
                            pos = get_position_status()
                            emoji = "🟢" if signal == "BUY" else "🔴"
                            send_message(
                                f"{emoji} <b>{signal} {SYMBOL}</b> {strength}\n\n"
                                f"💵 Цена входа:  {price}\n"
                                f"🛑 Стоп-лосс:  {pos['stop_loss']}\n"
                                f"✅ Тейк-профит: {pos['take_profit']}\n"
                                f"🎯 Уверенность: {adjusted_confidence:.1%}\n"
                                f"🧠 Настроение:  {sentiment_str}\n"
                                f"⏰ Время:       {now}"
                            )
                    else:
                        print(f"[App] ⏭️ Сигнал после корректировки слабый: {adjusted_confidence:.1%}")

                else:
                    print(f"[App] ⏭️ Уверенность ниже порога: {confidence:.1%} < {MIN_CONFIDENCE:.1%}")

            # ── Шаг 4: Пауза до следующего цикла ───────────
            print(f"[App] 💤 Следующий цикл через {SIGNAL_INTERVAL_MINUTES} минут")
            time.sleep(SIGNAL_INTERVAL_MINUTES * 60)

        except Exception as e:
            print(f"[App] ❌ Ошибка в торговом цикле: {e}")
            traceback.print_exc()
            time.sleep(60)


# ═══════════════════════════════════════════
# ЕЖЕДНЕВНАЯ СТАТИСТИКА
# ═══════════════════════════════════════════
def daily_stats_loop():
    """Отправляет статистику каждые 24 часа"""
    while True:
        try:
            time.sleep(24 * 60 * 60)
            stats = get_statistics()
            send_message(
                f"📊 <b>Ежедневная статистика</b>\n\n"
                f"📈 Всего сделок: {stats['total']}\n"
                f"✅ Прибыльных:  {stats['wins']}\n"
                f"❌ Убыточных:   {stats['losses']}\n"
                f"🎯 Винрейт:     {stats['winrate']}%\n"
                f"💰 Средний PnL: {stats['avg_pnl']:+.2f}%"
            )
        except Exception as e:
            print(f"[App] Ошибка статистики: {e}")
            time.sleep(60 * 60)


# ═══════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  TradeBot v2.0 — Self-Learning Edition")
    print("=" * 50)

    # Проверка конфигурации
    if not startup_check():
        print("[App] ❌ Конфигурация неполная. Проверь переменные окружения на Render.")
        exit(1)

    # Уведомление о запуске
    send_message(
        f"🤖 <b>TradeBot запущен!</b>\n\n"
        f"📊 Пара:       {SYMBOL}\n"
        f"⏱️ Таймфрейм: {TIMEFRAME}\n"
        f"🎯 Мин. уверенность: {MIN_CONFIDENCE:.0%}\n"
        f"🔄 Переобучение: по воскресеньям в 02:00 UTC"
    )

    # Запускаем потоки
    threads = [
        threading.Thread(target=trading_loop,      daemon=True, name="TradingLoop"),
        threading.Thread(target=run_retrainer_loop, daemon=True, name="Retrainer"),
        threading.Thread(target=daily_stats_loop,   daemon=True, name="DailyStats"),
        threading.Thread(target=run_health_server,  daemon=True, name="HealthCheck"),
    ]

    for t in threads:
        t.start()
        print(f"[App] ✅ Поток запущен: {t.name}")

    print("\n[App] 🚀 Все системы запущены!")

    # Держим главный поток живым
    while True:
        time.sleep(60)