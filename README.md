# Bikeshare

Сервис почасового прогноза спроса на велосипеды для Capital Bikeshare (Вашингтон, 2011-2012). Архитектура состоит из трёх частей:

1. Исследовательская часть (4 ноута): EDA, feature engineering, baseline-модель, основной CatBoost
2. Production-сервис (FastAPI + PostgreSQL): HTTP-эндпоинт `/predict`,  который принимает наблюдение и отдаёт прогноз `cnt` (велосипеды в час)
3. Инфраструктура: PostgreSQL (история наблюдений), MLflow (трекинг экспериментов), Alembic (миграции и сид-скрипт).


## Структура проекта

```
bikeshare/
  alembic/                  Миграции БД + сид-скрипт из raw.csv
  config/                   Конфиги сервиса
  datasets/
    raw.csv                 Исходные 17379 часов UCI Bike Sharing
    eda.csv                 Промежуточный датасет после EDA
    clean/
      train.csv             8645 строк, 2011 год, с cold-start masking
      validation.csv        4358 строк, 1H 2012 (январь-июнь)
      test.csv              4376 строк, 2H 2012 (июль-декабрь), полная история
      test_cold.csv         тот же test, но с masking 80/15/5
      test_immediate.csv    тот же test, но все 12 temporal = NaN
  notebooks/
    01_eda.ipynb            исследование, гипотезы, графики
    02_feature_engineering.ipynb  фичи, log1p, cold-start masking, сплит
    03_baseline.ipynb       DecisionTreeRegressor (нижняя планка)
    04_catboost.ipynb       CatBoost baseline + Optuna + SHAP
  src/
    api/v1/                FastAPI-роутер /predict, /health
    domain/
      base/                BaseService, BaseRepository (generic CRUD)
      common/models.py     SQLAlchemy ORM-модель BikeReading
      forecaster/
        schemas.py         PredictionRequest, PredictionResponse,
                           FeatureVector, TemporalFeatures
        repositories.py    BikeReadingRepository.get_temporal_features
        services.py        ForecastService.predict
    low/
      app.py               FastAPI App-обёртка, lifespan, middleware
      container.py         dependency-injector контейнер
      config/              чтение config.json
      creds/               чтение creds.json
      database/            async SQLAlchemy engine, UoW
      logging.py           логгер с file + console handler
      middleware/          ErrorHandling, DatabaseSession
  main.py                  точка входа: грузит конфиги, поднимает uvicorn
  docker-compose.yml       postgres
  docker-compose-mlflow.yml  mlflow
  alembic.ini              конфигурация миграций
  pyproject.toml           зависимости (uv)
```


## Технологии

- Python 3.13 + uv (pyproject.toml, uv.lock)
- FastAPI + uvicorn (HTTP API)
- PostgreSQL (хранилище истории наблюдений)
- SQLAlchemy + asyncpg (доступ к БД)
- Alembic (миграции + сид)
- CatBoost (модель)
- pandas, numpy, scikit-learn (обработка)
- Optuna (подбор гиперпараметров)
- SHAP (интерпретация)
- MLflow (трекинг экспериментов)
- dependency-injector (DI-контейнер)
- pydantic v2 (валидация request/response)


## Этапы работы

### 01_eda.ipynb. Разведочный анализ

Загрузка 17379 почасовых записей UCI Bike Sharing, анализ, виуализации. Чистка ошибок (hum)

### 02_feature_engineering.ipynb. Подготовка фичей

Из 14 исходных колонок получается 34 признака + таргет:

- Статичные (22): yr, mnth, hr, holiday, weekday, workingday, temp, hum,
  windspeed, weather_1..3 (one-hot, weather_4 склеен с weather_3),
  season_1..4 (one-hot), hr_sin, hr_cos, mnth_sin, mnth_cos, is_rush_hour,
  is_night.
- Temporal (12): cnt_lag_1/3/6/12/24, cnt_rolling_mean_3/6/12/24,
  cnt_rolling_std_6/12, cnt_ewm_6h. Все считаются на log1p(cnt) и сдвинуты
  на 1 строку чтобы не было утечки

Преобразование таргета: `cnt = log1p(cnt)` (правый скос 1.28 уходит в log-пространство). При инференсе делается обратное `expm1`.

Хронологический сплит:
- train: yr=0 (2011), 8645 строк
- val:   yr=1 и mnth <= 6 (январь-июнь 2012), 4358 строк
- test:  yr=1 и mnth >  6 (июль-декабрь 2012), 4376 строк

Cold-start masking на train: 80% normal (вся история), 15% partial
(остаются только cnt_lag_1, cnt_lag_3, cnt_rolling_mean_3, cnt_ewm_6h),
5% cold (все 12 temporal = NaN). Модель учится работать без истории.

Из test готовятся два дополнительных файла:
- test_cold.csv: тот же mix 80/15/5 что в train (моделирует production)
- test_immediate.csv: все 12 temporal = NaN (стресс-тест на zero history)

### 03_baseline.ipynb. Дерево решений

Два DecisionTreeRegressor с глубиной 4 как нижняя планка:
- Дерево 1 (только 22 статичных признака): RMSE на test = 181.6, R2 = 0.686. Метрики идентичны на test, test_cold, test_immediate (cold-start-иммунитет: temporal не используется)
- Дерево 2 (все 34 признака, NaN заполнен медианой train): RMSE на test = 134.9, R2 = 0.867. На test_immediate деградация катастрофическая (RMSE 238.6, R2 = 0.367) - медиана `cnt_lag_1` около 109 неприменима для ночных часов с реальным спросом 0-10.

Вывод: ручное заполнение NaN медианой работает плохо при холодном старте. Нужна модель, которая умеет с NaN нативно (CatBoost с nan_mode='Min').

### 04_catboost.ipynb. Основная модель

Baseline CatBoost с параметрами по умолчанию + `nan_mode='Min'` + early stopping. Затем Optuna TPE, 50 trials, поиск:
learning_rate в [0.01, 0.3] log, depth [4, 10], l2_leaf_reg [1, 50] log, min_data_in_leaf [1, 100], subsample [0.5, 1], colsample_bylevel [0.5, 1].

Финальные метрики лучшей модели:

```
            rmsle     rmse      mae     r2
test       0.3289  102.23    61.35  0.9441
test_cold  0.3509  106.18    64.16  0.9363
immediate  0.5779  148.47   104.53  0.8273
```

- test_cold почти равен test: маскирование на обучении сработало.
- test_immediate хуже, но R2 = 0.83 это работоспособный сервис.
- Деградация на test_immediate концентрируется в часы-пик: там lag несут
  максимум информации, ночью lag почти бесполезны.

SHAP, важности признаков, predicted-vs-actual для трёх режимов содержатся в ноутбуке

### Конфигурация

`config/prod/config.json`:

```json
{
  "general": {
    "creds": "config/prod/creds.json",
    "api_v1_str": "/api/v1",
    "project_name": "ORAG - Org Mode RAG Service",
    "version": "0.1.0",
    "debug": false
  },
  "networking": {
    "server_host": "127.0.0.1",
    "server_port": 8000,
    "frontend": { "origin": ["http://localhost:3000"] }
  },
  "logging": {
    "log_path": "./log",
    "console_level": "INFO",
    "file_level": "DEBUG",
    "max_log_size_mb": 5
  },
  "modules": {
    "database": {
      "server": "192.168.1.10",
      "port": 5432,
      "db_name": "orag_dev",
      "pool_size": 10
    },
    "model": {
      "path": "./models/catboost_bikeshare.cbm",
      "immediate_only": false
    }
  }
}
```

`config/prod/creds.json`:

```json
{
  "modules": {
    "database": { "user": "admin", "password": "***" }
  }
}
```

Ключевое поле для нашей задачи: `modules.model.immediate_only`:

- `false`: production-режим. Сервис при каждом запросе идёт в БД за  историей, считает lag/rolling/EWM, использует полный feature vector.
- `true`: zero-history-режим. Все 12 temporal-признаков подставляются как NaN, CatBoost обрабатывает их через `nan_mode='Min'`. Используется если БД недоступна или сервис только запущен и истории ещё нет.


## Как данные попадают в базу

Сейчас источник истории - сид-миграция Alembic `alembic/versions/7d46af7c7a4c_seed_bike_readings.py`. Он:
1. Читает `datasets/raw.csv` (17379 строк, формат UCI Bike Sharing)
2. Для каждой строки собирает `reading_dt = datetime.combine(dteday, hr)`
3. Применяет `np.log1p(cnt)` (БД хранит cnt в log1p-шкале, как и фичи)
4. Делает один `bulk_insert` в таблицу `bike_readings`

Схема таблицы (`alembic/versions/a6b284fed409_initial_schema.py`):

```sql
CREATE TABLE bike_readings (
    id          UUID PRIMARY KEY,
    reading_dt  TIMESTAMP NOT NULL,
    weekday     INTEGER   NOT NULL,
    season      INTEGER   NOT NULL,
    cnt         DOUBLE PRECISION NOT NULL,
    created_at  TIMESTAMPTZ,
    updated_at  TIMESTAMPTZ
);
CREATE UNIQUE INDEX ix_bike_readings_reading_dt
    ON bike_readings (reading_dt);
```

Уникальный индекс по `reading_dt` гарантирует одну запись на час - также помогает поиску, так как в запросе есть только дата


### Развитие
В дальнейшем расширении сервиса может быть добавлена отдельная ручка для записи новых наблюдений в БД (как POST /readings), или сразу добавление всех предсказаний в таблицу


## Запуск

### 1. Зависимости

```bash
uv sync
```

### 2. Поднять PostgreSQL и MLflow

```bash
docker compose -f docker-compose.yml         up -d
docker compose -f docker-compose-mlflow.yml  up -d
```

### 3. Применить миграции + сид

```bash
uv run alembic upgrade head
```

Это создаст таблицу `bike_readings` и зальёт 17379 строк из `raw.csv`. Путь к датасету можно переопределить переменной `DATASET_PATH`

### 4. Запустить сервис

```bash
uv run python main.py
```

По умолчанию слушает `http://127.0.0.1:8000`. Swagger по адресу `http://127.0.0.1:8000/docs`.


## HTTP API

### GET /api/v1/health

Ответ:

```json
{ "status": "ok" }
```

### POST /api/v1/predict

Запрос (все поля обязательны):

```json
{
  "dteday":     "2012-07-05",
  "hr":         17,
  "holiday":    0,
  "weathersit": 1,
  "temp":       0.72,
  "hum":        0.58,
  "windspeed":  0.0
}
```

Поля:

| Поле       | Тип     | Допустимые значения              | Описание                                |
|------------|---------|----------------------------------|-----------------------------------------|
| dteday     | string  | YYYY-MM-DD, +-365 дней от 2011-2012 | Дата наблюдения                      |
| hr         | int     | 0..23                            | Час дня                                 |
| holiday    | int     | 0 или 1                          | 1 если государственный праздник         |
| weathersit | int     | 1, 2, 3, 4                       | 1 ясно, 2 туман, 3 лёгкий дождь, 4 шторм |
| temp       | float   | 0.0..1.0                         | Нормализованная температура             |
| hum        | float   | 0.0..1.0                         | Нормализованная влажность               |
| windspeed  | float   | 0.0..1.0                         | Нормализованная скорость ветра          |

Ответ:

```json
{
  "predicted_cnt": 348.51,
  "cold_start":    false
}
```

- `predicted_cnt`: прогноз количества арендованных велосипедов в данный
  час (целое количество, но возвращается как float). Получается из
  `expm1(model.predict(...))` с обрезанием отрицательных значений.
- `cold_start`: `true`, если хотя бы один из 12 temporal-признаков не
  удалось получить (либо потому что `immediate_only=true`, либо потому
  что в БД не хватило истории). Это сигнал клиенту, что предсказание
  сделано на ограниченной информации и точность будет ниже (на
  test_immediate против test модель проседает с RMSE 102 до RMSE 148).


### Примеры запуска

#### Режим immediate_only = false (production, с историей)

В БД должна быть история минимум на 24 часа назад относительно
запрашиваемого `reading_dt`.

Запрос (хорошо известная точка - вторник 17:00 в июле, час-пик):

```
POST http://127.0.0.1:8000/api/v1/predict
Content-Type: application/json

{
  "dteday":     "2012-07-05",
  "hr":         17,
  "holiday":    0,
  "weathersit": 1,
  "temp":       0.72,
  "hum":        0.58,
  "windspeed":  0.0
}
```

Ответ (приблизительный, точные числа зависят от содержимого БД):

```json
{
  "predicted_cnt": 348.51,
  "cold_start":    false
}
```

Что произошло внутри:
1. `reading_dt = 2012-07-05 17:00:00`.
2. Репозиторий делает один SQL-запрос:
   ```sql
   SELECT * FROM bike_readings
   WHERE reading_dt < '2012-07-05 17:00:00'
   ORDER BY reading_dt DESC
   LIMIT 48;
   ```
   и разворачивает результат в хронологическом порядке.
3. Считаются 12 temporal-признаков по позициям (lag_1 = последняя
   строка, lag_24 = 24-я с конца и т.п.).
4. `FeatureVector.from_request` собирает 34 признака.
5. `model.predict(...)` отдаёт log1p-прогноз.
6. `expm1(clip(.))` переводит обратно в велосипеды/час.

#### Режим immediate_only = true (cold-start, без истории)

Тот же запрос, но в конфиге `immediate_only=true`:

```
POST http://127.0.0.1:8000/api/v1/predict
Content-Type: application/json

{
  "dteday":     "2012-07-05",
  "hr":         17,
  "holiday":    0,
  "weathersit": 1,
  "temp":       0.72,
  "hum":        0.58,
  "windspeed":  0.0
}
```

Ответ:

```json
{
  "predicted_cnt": 295.87,
  "cold_start":    true
}
```

Что произошло внутри:
1. Сервис вообще не ходит в БД
2. `TemporalFeatures()` создаётся с дефолтными None, и все 12 фичей подставляются в `FeatureVector` как `float('nan')`.
3. CatBoost при инференсе использует ветки `nan_mode='Min'`: NaN всегда уходят в одну сторону сплитов, модель опирается только на статичные признаки (час, день недели, погода, температура, is_rush_hour и т.д.)
4. `cold_start=true` в ответе сигнализирует клиенту о работе в этом режиме

#### Пример с холодной зимней ночью

```
POST http://127.0.0.1:8000/api/v1/predict
{
  "dteday":     "2012-12-23",
  "hr":         3,
  "holiday":    0,
  "weathersit": 2,
  "temp":       0.18,
  "hum":        0.74,
  "windspeed":  0.15
}
```

Ответ в warm-режиме (immediate_only=false):

```json
{ "predicted_cnt": 1.83, "cold_start": false }
```

Ответ в cold-режиме (immediate_only=true):

```json
{ "predicted_cnt": 2.41, "cold_start": true }
```

Ночью lag-признаки почти не помогают (спрос стабильно низкий), поэтому разница между режимами маленькая.

### Ошибки

Если запрос не прошёл валидацию pydantic:

```json
HTTP 400
{
  "message": "Input should be less than or equal to 23",
  "status_code": 400
}
```

(обработчик `validation_error_handler` в `src/low/app.py`).


## Что такое cold_start подробнее

Поле `cold_start` в ответе означает "истории не было / была неполная, прогноз сделан без temporal-признаков". Семантически это флаг качества:
- `cold_start=false`: все 12 temporal-фичей получены, модель работает в ожидаемом режиме. RMSE на тестовом периоде около 102 велосипедов/час,   R2 около 0.94. Это нормальная production-точность
- `cold_start=true`: хотя бы одна temporal-фича отсутствует. Модель опирается только на статичные признаки. RMSE поднимается до 148 велосипедов/час, R2 падает до 0.83. Точность ощутимо ниже, но сервис всё ещё функционален - что подтверждено в ноутбуке 04 на test_immediate

Когда `cold_start=true` может появиться:

1. `immediate_only=true` в конфиге (принудительный режим без БД)
2. БД не содержит ни одной записи раньше `reading_dt` (запрос за
   точку до начала истории)
3. БД содержит меньше 24 записей раньше `reading_dt`: тогда
   `cnt_lag_24` вернётся None, `is_cold` отработает True

Клиент может использовать флаг для принятия решений (например, показать на UI "приблизительный прогноз")