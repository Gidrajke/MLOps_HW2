# MLOps Project: ML Service + MinIO

## Описание проекта

Этот проект демонстрирует реализацию базового MLOps пайплайна с использованием собственного ML-сервиса и MinIO для хранения данных. Проект включает:

1. **ML-сервис** на FastAPI, который:

   * Обрабатывает запросы на обучение модели.
   * Позволяет делать предсказания.
   * Поддерживает управление моделями.

2. **MinIO** для хранения данных и моделей. Используется как локальный аналог S3.

3. **Docker Compose** для поднятия всех сервисов.

> Пункт 4 (ClearML/MLFlow) не реализован.

## Структура проекта

```
MLOps/
├─ app/                  # FastAPI приложение
├─ docker-compose.yml    # Compose файл для ML сервиса и MinIO
├─ requirements.txt      # Зависимости Python
├─ README.md             # Этот файл
```

## Установка и запуск

### Требования

* Docker
* Docker Compose
* Python 3.10+
* Виртуальное окружение (рекомендуется)

### Шаги запуска

1. **Создание виртуального окружения (опционально)**

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
```

2. **Установка зависимостей**

```bash
pip install -r requirements.txt
```

3. **Запуск Docker Compose**

```bash
docker-compose up -d
```

После запуска будут подняты следующие сервисы:

* ML-сервис: `http://localhost:8000`
* MinIO: `http://localhost:9000` (Web UI)

  * Access Key: `minioadmin`
  * Secret Key: `minioadmin`

### Проверка работы ML-сервиса

* Swagger документация: `http://localhost:8000/docs`
* Примеры запросов на обучение модели и предсказания доступны через Swagger.

### Управление сервисами

* Остановка сервисов:

```bash
docker-compose down
```

* Просмотр логов ML-сервиса:

```bash
docker-compose logs -f ml_service
```

* Просмотр логов MinIO:

```bash
docker-compose logs -f minio
```

## Использование MinIO

* Подключение через Web UI: `http://localhost:9000`
* Логин: `minioadmin`
* Пароль: `minioadmin`
* Можно создавать бакеты для хранения данных и моделей.

## Замечания

* ClearML/MLFlow интеграция не реализована.
* Проект готов для локальной демонстрации и базового обучения моделей через API.
