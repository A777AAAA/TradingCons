"""
Анализатор настроений рынка для TON/USDT
Использует OpenRouter API (бесплатные модели)
"""

import os
import json
import requests
from datetime import datetime

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

def get_market_sentiment(price: float, change_24h: float, volume: float) -> dict:
    """
    Получает анализ настроений рынка через AI
    Возвращает: {"sentiment": "bullish/bearish/neutral", "confidence": 0.0-1.0, "reason": "..."}
    """
    if not OPENROUTER_API_KEY:
        return {"sentiment": "neutral", "confidence": 0.5, "reason": "API key not set"}

    prompt = f"""Analyze TON/USDT market:
Price: ${price:.4f}
24h Change: {change_24h:+.2f}%
Volume: ${volume:,.0f}

Reply ONLY with JSON:
{{"sentiment": "bullish" or "bearish" or "neutral", "confidence": 0.0-1.0, "reason": "one sentence"}}"""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.3
            },
            timeout=15
        )

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            # Извлекаем JSON из ответа
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
                return result

    except Exception as e:
        print(f"[Sentiment] Ошибка: {e}")

    return {"sentiment": "neutral", "confidence": 0.5, "reason": "Analysis unavailable"}


def sentiment_to_signal_boost(sentiment: dict, base_signal: str) -> float:
    """
    Преобразует настроение в корректировку уверенности сигнала.
    Возвращает множитель: 1.2 (усилить), 1.0 (нейтрально), 0.8 (ослабить)
    """
    s = sentiment.get("sentiment", "neutral")
    conf = sentiment.get("confidence", 0.5)

    if base_signal == "BUY" and s == "bullish":
        return 1.0 + (0.2 * conf)
    elif base_signal == "BUY" and s == "bearish":
        return 1.0 - (0.2 * conf)
    elif base_signal == "SELL" and s == "bearish":
        return 1.0 + (0.2 * conf)
    elif base_signal == "SELL" and s == "bullish":
        return 1.0 - (0.2 * conf)

    return 1.0


if __name__ == "__main__":
    # Тест
    result = get_market_sentiment(price=5.234, change_24h=2.5, volume=1500000)
    print(f"Результат: {result}")