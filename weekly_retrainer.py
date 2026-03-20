"""
Еженедельное переобучение модели
Запускается автоматически по расписанию (воскресенье 02:00 UTC)
"""

import time
import schedule
import traceback
from datetime import datetime

from config import RETRAIN_DAY, RETRAIN_HOUR, SYMBOL, TIMEFRAME
from hf_storage import save_model, load_model
from auto_trainer import train_model
from telegram_notify import send_message


def retrain_job():
    """
    Основная задача переобучения.
    Скачивает свежие данные → обучает модель → сохраняет на HuggingFace
    """
    started_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n[Retrainer] 🔄 Начало переобучения: {started_at}")

    try:
        # Уведомляем в Telegram
        send_message(
            f"🔄 <b>Начало еженедельного переобучения</b>\n"
            f"📅 {started_at}\n"
            f"📊 {SYMBOL} | {TIMEFRAME}"
        )

        # Запускаем обучение
        print("[Retrainer] Загрузка данных и обучение...")
        result = train_model()

        if result and result.get("success"):
            accuracy  = result.get("accuracy", 0)
            precision = result.get("precision", 0)
            recall    = result.get("recall", 0)
            n_samples = result.get("n_samples", 0)

            # Сохраняем модель на HuggingFace
            print("[Retrainer] Сохранение модели на HuggingFace...")
            saved = save_model(
                model=result.get("model"),
                scaler=result.get("scaler"),
                metadata={
                    "trained_at": started_at,
                    "accuracy":   accuracy,
                    "precision":  precision,
                    "recall":     recall,
                    "n_samples":  n_samples,
                    "symbol":     SYMBOL,
                    "timeframe":  TIMEFRAME
                }
            )

            if saved:
                msg = (
                    f"✅ <b>Переобучение завершено!</b>\n\n"
                    f"📊 Результаты:\n"
                    f"   Точность:  {accuracy:.1%}\n"
                    f"   Precision: {precision:.1%}\n"
                    f"   Recall:    {recall:.1%}\n"
                    f"   Образцов:  {n_samples}\n\n"
                    f"💾 Модель сохранена на HuggingFace"
                )
            else:
                msg = (
                    f"⚠️ <b>Переобучение завершено, но модель не сохранена</b>\n"
                    f"Точность: {accuracy:.1%}"
                )

        else:
            error = result.get("error", "Неизвестная ошибка") if result else "Нет результата"
            msg = (
                f"❌ <b>Ошибка переобучения</b>\n"
                f"Причина: {error}"
            )

        send_message(msg)
        print(f"[Retrainer] {msg}")

    except Exception as e:
        error_msg = f"❌ <b>Критическая ошибка переобучения</b>\n{str(e)}"
        print(f"[Retrainer] ОШИБКА: {e}")
        traceback.print_exc()
        send_message(error_msg)


def schedule_retraining():
    """
    Настраивает расписание переобучения.
    По умолчанию: воскресенье в 02:00 UTC
    """
    time_str = f"{RETRAIN_HOUR:02d}:00"

    if RETRAIN_DAY == "sunday":
        schedule.every().sunday.at(time_str).do(retrain_job)
    elif RETRAIN_DAY == "monday":
        schedule.every().monday.at(time_str).do(retrain_job)
    elif RETRAIN_DAY == "saturday":
        schedule.every().saturday.at(time_str).do(retrain_job)
    else:
        schedule.every().sunday.at(time_str).do(retrain_job)

    print(f"[Retrainer] ✅ Переобучение запланировано: {RETRAIN_DAY} в {time_str} UTC")


def run_retrainer_loop():
    """
    Запускает бесконечный цикл планировщика.
    Вызывается из app.py в отдельном потоке.
    """
    schedule_retraining()

    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Проверяем каждую минуту
        except Exception as e:
            print(f"[Retrainer] Ошибка в цикле: {e}")
            time.sleep(60)


def force_retrain():
    """
    Принудительный запуск переобучения (для тестирования).
    """
    print("[Retrainer] 🚀 Принудительный запуск переобучения...")
    retrain_job()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "force":
        # python weekly_retrainer.py force
        force_retrain()
    else:
        # Обычный запуск планировщика
        print("[Retrainer] Запуск планировщика...")
        run_retrainer_loop()