# Handoff Phase 3: Интеграция с Dynamic Pricing Engine

> **Автор:** оркестратор  
> **Дата:** 2026-08-09  
> **Предыдущий handoff:** [handoff.md](file:///Z:/pet-project/causal-pricing-engine/handoff.md) (Фазы 1–2 — DONE)  
> **Аудит:** [audit_report.md](file:///Z:/pet-project/causal-pricing-engine/audit_report.md) — 19/19 тестов, 12/12 задач  
> **Цель:** связать `causal-pricing-engine` с работающим `dynamic-pricing-engine`, замкнуть цикл данных и оценки эффектов.

---

## 1. Контекст: два проекта

| Проект | Путь | Роль | Порт |
|:---|:---|:---|:---|
| **Dynamic Pricing Engine (DPE)** | `Z:\pet-project\dynamic-pricing-engine` | Расчёт цен поездок на графе дорог, Switchback A/B | `:8000` |
| **Causal Pricing Engine (CPE)** | `Z:\pet-project\causal-pricing-engine` | Uplift-моделирование, оценка персонального эффекта | `:8100` (новый) |

CPE сейчас работает автономно на синтетических данных. Задача — обучить его на **реальных данных симуляции DPE** и встроить в pricing pipeline DPE.

---

## 2. Схема БД DPE (источник данных)

### Таблица `simulation_analytics` (770 строк, SQLite)

Файл: `Z:\pet-project\dynamic-pricing-engine\dpe_database.db`

| Колонка | Тип | Описание | Маппинг для CPE |
|:---|:---|:---|:---|
| `id` | INTEGER PK | Автоинкремент | — |
| `timestamp` | REAL | Unix timestamp поездки | Можно извлечь `hour_of_day` |
| `test_group` | TEXT | `"ADDITIVE"` или `"MULTIPLICATIVE"` | **→ treatment** (ADDITIVE=1, MULTIPLICATIVE=0) |
| `trip_id` | TEXT | `"search_XXXXXX"` | → `user_id` (или использовать `driver_id`) |
| `distance_km` | REAL | Расстояние поездки | → feature |
| `duration_sec` | REAL | Время поездки | → feature |
| `price` | REAL | Итоговая цена | → feature |
| `surge_bonus` | REAL | Аддитивная надбавка | → feature |
| `accepted` | INTEGER | 0/1 — принял ли водитель заказ | **→ conversion (y)** |
| `driver_utility` | REAL | Полезность поездки для водителя | → feature или → revenue |
| `driver_id` | TEXT | `"driver_XXX"` | Группировка по водителю для `past_trips` |

### Таблица `prices` (8988 строк)

| Колонка | Тип | Описание |
|:---|:---|:---|
| `h3_index` | TEXT | H3-ячейка |
| `price`, `base_price` | REAL | Цены |
| `surge_multiplier` | REAL | Множитель |
| `test_group` | TEXT | Группа теста |
| `surge_bonus` | REAL | Аддитивный бонус |

---

## 3. API DPE (точка встраивания)

### Ключевой эндпоинт: `POST /api/v1/search`

Файл: [`Z:\pet-project\dynamic-pricing-engine\src\api\main.py`](file:///Z:/pet-project/dynamic-pricing-engine/src/api/main.py)

**Текущий flow** (строки 106–200):
```
SearchRequest → snap_to_node → get_features_graph → compute_surge_price → PriceResponse
```

**Где встраивается CPE:**
```
SearchRequest → snap_to_node → get_features_graph
    → HTTP GET к CPE /predict_uplift (фичи водителя/поездки)    ← НОВОЕ
    → если uplift > 0 → применяем Surge / скидку
    → если uplift < 0 (Sleeping Dog) → не трогаем тариф         ← НОВОЕ
    → compute_surge_price → PriceResponse
```

### API CPE (уже реализовано)

**`POST /predict_uplift`** — [`Z:\pet-project\causal-pricing-engine\src\api\main.py`](file:///Z:/pet-project/causal-pricing-engine/src/api/main.py)

Request:
```json
{
  "user_id": "driver_008",
  "features": {
    "past_trips": 12.0,
    "avg_surge": 1.25,
    "price_sensitivity": 0.8,
    "hour_of_day": 18.0,
    "segment": 2.0
  }
}
```

Response:
```json
{
  "user_id": "driver_008",
  "uplift_score": 0.23,
  "recommended_treatment": "DISCOUNT_10_PCT",
  "optimal_discount_pct": 10.0,
  "model_name": "XLearner"
}
```

---

## 4. Контракт FEATURE_COLUMNS (CPE)

Текущий контракт в [`src/data/synthetic.py`](file:///Z:/pet-project/causal-pricing-engine/src/data/synthetic.py):
```python
FEATURE_COLUMNS = ["past_trips", "avg_surge", "price_sensitivity", "hour_of_day", "segment"]
```

Для данных DPE потребуется **расширить или адаптировать** контракт. Варианты:
- **Вариант A (рекомендуемый):** Добавить новые колонки DPE (`distance_km`, `duration_sec`, `surge_bonus`) в `FEATURE_COLUMNS` и пересоздать синтетику.
- **Вариант B:** Создать отдельный `DPE_FEATURE_COLUMNS` и маппинг-слой в коннекторе.

---

## 5. План задач Фазы 3

### T13: Data Connector для DPE

**Файл:** `src/data/dpe_connector.py`

**Что должен делать:**
1. Принять путь к `dpe_database.db` (по умолчанию — `Z:\pet-project\dynamic-pricing-engine\dpe_database.db`).
2. Прочитать таблицу `simulation_analytics` через `sqlite3`.
3. Сформировать DataFrame с колонками:

| Целевая колонка | Источник | Логика |
|:---|:---|:---|
| `user_id` | `driver_id` | Прямое копирование |
| `treatment` | `test_group` | `1 if "ADDITIVE" else 0` |
| `conversion` | `accepted` | Прямое копирование |
| `revenue` | `driver_utility` | Прямое копирование |
| `distance_km` | `distance_km` | Прямое копирование |
| `duration_sec` | `duration_sec` | Прямое копирование |
| `price` | `price` | Прямое копирование |
| `surge_bonus` | `surge_bonus` | Прямое копирование |
| `hour_of_day` | `timestamp` | `datetime.fromtimestamp(ts).hour` |
| `past_trips` | `driver_id` | Агрегат: `COUNT(*)` поездок этого водителя до текущей строки (cumulative count) |
| `avg_surge` | `driver_id` | Агрегат: скользящее среднее `surge_bonus` по водителю |

4. Также подгрузить данные из таблицы `prices`, чтобы обогатить признаки (необязательно в первой версии).

**Интерфейс:**
```python
def load_dpe_data(
    db_path: str | Path = "Z:/pet-project/dynamic-pricing-engine/dpe_database.db",
) -> pd.DataFrame:
    """Load simulation_analytics from DPE SQLite and transform to CPE training format."""
```

**Критерий приёмки:**
- Возвращает DataFrame, совместимый с `scripts/train.py` (колонки: `user_id`, features, `treatment`, `conversion`, `revenue`).
- Тест в `tests/test_dpe_connector.py`: загрузить реальную БД, проверить schema, проверить `treatment ∈ {0, 1}`, проверить `conversion ∈ {0, 1}`.

---

### T14: Интеграция в pricing pipeline DPE

**Файлы (в проекте DPE!):**
- `Z:\pet-project\dynamic-pricing-engine\src\api\main.py` — добавить HTTP-вызов к CPE.

**Что делать:**
1. В функцию `request_price()` (строка ~166) **перед** `compute_surge_price()` добавить опциональный вызов к CPE API:

```python
# --- Causal Uplift Check (optional) ---
causal_treatment = None
try:
    import httpx
    resp = httpx.post(
        "http://localhost:8100/predict_uplift",
        json={
            "user_id": driver_id_if_available,
            "features": {
                "distance_km": trip_dist_km,
                "duration_sec": trip_duration_sec,
                "hour_of_day": float(hour),
                "past_trips": ...,   # из feature_store или фиксированное значение
                "avg_surge": ...,
            }
        },
        timeout=0.1,  # 100 мс жёсткий таймаут — не замедляем DPE
    )
    if resp.status_code == 200:
        causal_treatment = resp.json()
except Exception:
    pass  # Fail-open: при недоступности CPE — работаем как раньше

# Если uplift < 0 (Sleeping Dog) — не даём Surge этому водителю
if causal_treatment and causal_treatment["uplift_score"] < -0.05:
    surge_bonus = 0.0
    test_group = "CAUSAL_NO_SURGE"
```

2. **Изменить порт CPE** на `:8100` (в `configs/config.yaml` и `docker-compose.yml` CPE), чтобы не конфликтовать с DPE на `:8000`.

**Критерий приёмки:**
- При запущенном CPE (`uvicorn src.api.main:app --port 8100`) DPE при расчёте цены обращается к CPE.
- При недоступном CPE — DPE работает как раньше (fail-open, <30 мс SLA не нарушен).
- Добавить `httpx` в `requirements.txt` DPE.

---

### T15: CUPED Evaluation Pipeline

**Файл:** `scripts/evaluate_experiment.py`

**Что должен делать:**
1. Загрузить данные из DPE через `dpe_connector.load_dpe_data()`.
2. Разделить на control (`treatment=0`) и test (`treatment=1`).
3. Применить `cuped_adjust()` из `src/experiments/power_analysis.py`, используя `past_trips` как ковариату.
4. Рассчитать:
   - Δ (разница средних conversion rate)
   - p-value (двусторонний t-тест или bootstrap)
   - 95% Confidence Interval
   - Размер эффекта (Cohen's d)
5. Вывести markdown-отчёт в stdout:

```
## Experiment Evaluation Report

| Metric                     | Value         |
|:---------------------------|:--------------|
| Control Conversion Rate    | 0.48          |
| Treatment Conversion Rate  | 0.53          |
| Δ (raw)                    | +0.050        |
| Δ (CUPED-adjusted)         | +0.048        |
| p-value                    | 0.023         |
| 95% CI                     | [0.007, 0.089]|
| Cohen's d                  | 0.31          |
| Verdict                    | ✅ SIGNIFICANT |

Variance reduction from CUPED: 34.2%
Required sample size (α=0.05, power=0.80): 1200 per group
Current sample size: 385 per group
⚠️ Experiment is underpowered. Need 815 more observations per group.
```

**Критерий приёмки:**
- Скрипт запускается командой `python scripts/evaluate_experiment.py`.
- Корректно обрабатывает случай, когда данных недостаточно (выводит предупреждение о мощности).
- Тест: `tests/test_evaluation_pipeline.py` — прогнать на синтетических данных из `generate_uplift_dataset()`.

---

### T16: Обновить `scripts/train.py` для работы с DPE-данными

**Что делать:**
1. Добавить аргумент `--source` (`synthetic` | `dpe`):
   - `--source synthetic` (по умолчанию) — текущее поведение.
   - `--source dpe` — загрузить данные через `dpe_connector.load_dpe_data()`.
2. Обновить `FEATURE_COLUMNS` если используется `dpe` source (добавить `distance_km`, `duration_sec`, etc.).

**Критерий приёмки:**
- `python scripts/train.py --source synthetic` — работает как раньше.
- `python scripts/train.py --source dpe` — обучается на данных из DPE SQLite.

---

### T17: Единый docker-compose для обоих сервисов

**Файл:** `Z:\pet-project\causal-pricing-engine\docker-compose.full.yml`

```yaml
services:
  causal-engine:
    build: .
    ports:
      - "8100:8100"
    volumes:
      - ../dynamic-pricing-engine/dpe_database.db:/app/data/dpe_database.db:ro
    environment:
      - CPE_API__PORT=8100

  dpe-api:
    build: ../dynamic-pricing-engine
    ports:
      - "8000:8000"
    depends_on:
      - causal-engine
    environment:
      - CAUSAL_ENGINE_URL=http://causal-engine:8100
```

**Критерий приёмки:**
- `docker compose -f docker-compose.full.yml up` поднимает оба сервиса.
- DPE → CPE интеграция работает через docker-сеть.

---

## 6. Порядок выполнения (зависимости)

```
T13 (connector)
  ├── T15 (evaluation) — использует connector
  ├── T16 (train --source dpe) — использует connector
  └── T14 (DPE integration) — нужен работающий CPE с DPE-данными
        └── T17 (docker-compose) — нужна интеграция
```

**Рекомендуемый порядок:** T13 → T16 → T15 → T14 → T17.

---

## 7. Чего НЕ нужно делать

- **Не менять существующие тесты и модули CPE** — Фазы 1–2 полностью проходят аудит.
- **Не менять схему БД DPE** — работаем только на чтение (`SELECT`).
- **Не добавлять Kafka / Streaming** — это Вектор 2, отдельный проект.
- **Не добавлять Streamlit-дашборд** — только CLI-скрипты и API.
- **Не блокировать DPE при недоступности CPE** — строго fail-open с таймаутом ≤100 мс.

---

## 8. Как проверить, что всё работает

```bash
# 1. Существующие тесты не сломаны
cd Z:\pet-project\causal-pricing-engine
python -m pytest tests/ -v

# 2. Connector загружает данные DPE
python -c "from src.data.dpe_connector import load_dpe_data; df = load_dpe_data(); print(df.shape, df.columns.tolist())"

# 3. Обучение на DPE-данных
python scripts/train.py --source dpe

# 4. Evaluation pipeline
python scripts/evaluate_experiment.py

# 5. Интеграция (запустить оба сервиса в разных терминалах)
# Терминал 1:
cd Z:\pet-project\causal-pricing-engine && uvicorn src.api.main:app --port 8100
# Терминал 2:
cd Z:\pet-project\dynamic-pricing-engine && uvicorn src.api.main:app --port 8000
# Терминал 3 (проверка):
curl http://localhost:8100/health
curl http://localhost:8000/health
```
