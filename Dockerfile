# Базовый образ Python
FROM python:3.11-slim

# Рабочая директория
WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости первыми (кэширование слоёв)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Порт для healthcheck (Render требует открытый порт)
EXPOSE 8080

# Переменная окружения для Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Healthcheck — Render проверяет что контейнер живой
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Запуск бота
CMD ["python", "app.py"]