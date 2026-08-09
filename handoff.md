# Handoff: causal-pricing-engine

> **Автор:** оркестратор  
> **Дата:** 2026-08-09  
> **Цель документа:** передать воркеру полный контекст и конкретный план действий для доведения скелета проекта до рабочего MVP.

---

## 1. Что это за проект

Causal Pricing Engine — **движок причинно-следственного анализа и Uplift-моделирования** для задач динамического ценообразования. Проект является прямым развитием соседнего pet-проекта [`Z:\pet-project\dynamic-pricing-engine`](file:///Z:/pet-project/dynamic-pricing-engine), который уже реализует:

- Расчёт базового тарифа по дорожному графу Москвы (Dijkstra OD).
- Аддитивный Surge с k-Ring сглаживанием.
- Switchback A/B-тестирование (MULTIPLICATIVE vs ADDITIVE).
- Traffic Simulator, CDC/Outbox, Streamlit Dashboard.

**Данный проект** решает следующую задачу: вместо фиксированных бизнес-правил Surge, научиться определять **для каждого конкретного пользователя**, стоит ли давать скидку / бонус / изменённый тариф, и какой именно, используя Causal ML (Uplift Modeling, Double ML, CUPED).

---

## 2. Текущее состояние кода (аудит)

### 2.1 Файловая структура

```
Z:\pet-project\causal-pricing-engine\
├── README.md              ✅ Готов (документация, формулы, roadmap)
├── pyproject.toml          ✅ Готов (setuptools, pytest config)
├── requirements.txt        ✅ Готов (causalml, econml, catboost, fastapi, etc.)
├── src/
│   ├── __init__.py         ✅ Готов (пустой пакет)
│   ├── causal/
│   │   ├── __init__.py     ✅ Готов (re-export BaseUpliftModel, TLearner)
│   │   └── uplift_models.py ⚠️ Частично — есть только TLearner
│   ├── experiments/
│   │   ├── __init__.py     ✅ Готов (re-export cuped_adjust, sample_size_calculator)
│   │   └── power_analysis.py ⚠️ Частично — есть CUPED и sample_size_calculator
│   └── api/
│       ├── __init__.py     ✅ Готов (пустой пакет)
│       └── main.py         ⚠️ Заглушка — mock-логика, нет подключения модели
└── tests/
    ├── __init__.py         ✅ Готов
    └── test_uplift.py      ⚠️ Частично — 3 теста, нет тестов для API и будущих модулей
```

### 2.2 Что реализовано и работает

| Компонент | Файл | Статус | Комментарий |
|---|---|---|---|
| `TLearner` | `src/causal/uplift_models.py` | **Рабочий** | Полная реализация: fit/predict_uplift, поддержка predict_proba и predict, клонирование base_estimator через sklearn.clone |
| `BaseUpliftModel` | `src/causal/uplift_models.py` | **Рабочий** | Абстрактный базовый класс с NotImplementedError |
| `EstimatorProtocol` | `src/causal/uplift_models.py` | **Рабочий** | typing.Protocol для type-safe интеграции |
| `cuped_adjust` | `src/experiments/power_analysis.py` | **Рабочий** | CUPED: θ = Cov(Y,X)/Var(X), корректная обработка нулевой дисперсии |
| `sample_size_calculator` | `src/experiments/power_analysis.py` | **Рабочий** | Расчёт n на группу для two-sample t-test |
| FastAPI app | `src/api/main.py` | **Заглушка** | `/health` и `/predict_uplift` — mock-ответы без реальной модели |
| Тесты | `tests/test_uplift.py` | **Рабочие** | 3 теста: TLearner fit/predict, CUPED variance reduction, sample size |

### 2.3 Критические пробелы (что ещё НЕ существует)

| # | Отсутствующий модуль | Описание из README |
|---|---|---|
| G1 | `src/causal/dml_engine.py` | Double Machine Learning (EconML) & Synthetic Control — заявлен в README, файл не создан |
| G2 | `src/experiments/switchback_splitter.py` | Пространственно-временной сплиттер — заявлен в README, файл не создан |
| G3 | `configs/config.yaml` | Конфигурация — заявлена в README, директория не создана |
| G4 | `data/.gitkeep` | Директория данных — заявлена в README, не создана |
| G5 | S-Learner, X-Learner | Заявлены в README, реализован только TLearner |
| G6 | Подключение модели в API | `/predict_uplift` отдаёт хардкод, не подгружает обученную модель |
| G7 | Dockerfile / docker-compose.yml | Нет контейнеризации (есть в sibling-проекте DPE) |
| G8 | `.gitignore` / `.git` | Репозиторий не инициализирован |
| G9 | Makefile | Нет скриптов автоматизации (test, lint, serve) |

---

## 3. Контекст sibling-проекта (dynamic-pricing-engine)

Воркеру **необходимо** понимать контекст DPE для интеграции:

- **Путь:** `Z:\pet-project\dynamic-pricing-engine`
- **Demand-модель DPE:** `src/models/demand_model.py` — `DemandElasticityModel` (CatBoostRegressor), обучается на синтетических данных с признаками: `price`, `demand_supply_ratio`, `competitor_price`, `hour`.
- **Симулятор DPE:** `run_simulation.py` — `SimulationRunner` генерирует поездки, рассчитывает цену через API, сохраняет аналитику (trip_id, distance_km, price, surge_bonus, accepted, driver_utility) в SQLite `dpe_database.db`.
- **Switchback в DPE:** Простой сплиттер по виртуальным часам — чётный час = MULTIPLICATIVE, нечётный = ADDITIVE.

Causal Pricing Engine должен уметь:
1. Потреблять данные симуляции DPE (SQLite `simulation_analytics`) как observational data для обучения Uplift-моделей.
2. Вставляться в pipeline расчёта цены DPE как третий этап (после base_price и surge → uplift-based treatment assignment).

---

## 4. Контракты и конвенции кода

### 4.1 Стиль

- **Python ≥ 3.10**, используются union-типы через `X | Y` (не `Optional`/`Union`).
- `from __future__ import annotations` в каждом файле.
- Docstrings — формат Google/NumPy (в текущем коде смешанный — выбрать один и придерживаться).
- Type hints на **все** публичные функции и методы.
- Импорты — абсолютные от корня пакета: `from src.causal.uplift_models import TLearner`.

### 4.2 Интерфейс Uplift-моделей

Все Uplift-модели **обязаны** наследоваться от `BaseUpliftModel` и реализовать:

```python
class BaseUpliftModel:
    def fit(self, X, y, treatment) -> Self: ...
    def predict_uplift(self, X) -> np.ndarray: ...
```

- `X` — признаки (DataFrame или ndarray).
- `y` — целевая переменная (бинарная: конверсия; или непрерывная: revenue).
- `treatment` — бинарный вектор (0 = control, 1 = treatment).
- `predict_uplift` → одномерный ndarray с ITE-скорами (τ̂(x)).

### 4.3 Зависимости

- `causalml` — для готовых Uplift-деревьев и мета-лёрнеров (если нужна сверка с custom-реализацией).
- `econml` — для DML (DoubleML, CausalForestDML).
- `catboost` / `lightgbm` — base learners.
- `fastapi` + `pydantic` — API-слой.
- `scipy` — стат-тесты и калькулятор мощности.

---

## 5. План работы (приоритизированные задачи)

### Фаза 1: Фундамент (MUST — без этого проект не собирается)

> **Цель:** закрыть все заявленные в README пробелы, чтобы `pytest` и `uvicorn` запускались.

| # | Задача | Файлы | Критерий приёмки |
|---|---|---|---|
| T1 | Создать `configs/config.yaml` и загрузчик | `configs/config.yaml`, `src/config.py` | YAML с параметрами модели (base_learner, n_estimators, learning_rate), experiment (alpha, power, mde), api (host, port). Загрузка через Pydantic `BaseSettings` или `pydantic-settings`. |
| T2 | Создать `data/.gitkeep` и скрипт генерации синтетических данных | `data/.gitkeep`, `src/data/synthetic.py` | Функция `generate_uplift_dataset(n=5000) -> pd.DataFrame` с колонками: `user_id, features..., treatment (0/1), conversion (0/1), revenue (float)`. Данные должны содержать реалистичную гетерогенность эффекта (treatment effect > 0 для одного сегмента, ≈ 0 для другого, < 0 для третьего). |
| T3 | Реализовать `SLearner` | `src/causal/uplift_models.py` | S-Learner: единая модель на (X, W) → Y, uplift = predict(X, W=1) - predict(X, W=0). Тест в `test_uplift.py`. |
| T4 | Реализовать `XLearner` | `src/causal/uplift_models.py` | X-Learner: двухэтапная процедура (imputed treatment effects + propensity-weighted blend). Тест в `test_uplift.py`. |
| T5 | Создать `src/causal/dml_engine.py` | `src/causal/dml_engine.py` | Обёртка над `econml.dml.LinearDML` или `CausalForestDML`. Интерфейс: `fit(Y, T, X, W) -> self`, `effect(X) -> ndarray`. Тест. |
| T6 | Создать `src/experiments/switchback_splitter.py` | `src/experiments/switchback_splitter.py` | Класс `SwitchbackSplitter` с методом `assign(unit_id, timestamp) -> "control" \| "treatment"`. Поддержка разбиения по пространственно-временным блокам (grid_cell × time_bucket). Тест. |
| T7 | `.gitignore`, инициализация git | `.gitignore` | Стандартный Python .gitignore + исключения для `data/*.csv`, `*.bin`, `venv/`, `__pycache__/`. Выполнить `git init && git add . && git commit -m "init: project skeleton"`. |

### Фаза 2: Сборка Pipeline (SHOULD — превращает скелет в работающий MVP)

| # | Задача | Файлы | Критерий приёмки |
|---|---|---|---|
| T8 | Training pipeline | `scripts/train.py` | Скрипт: загрузить данные → обучить T/S/X-Learner + DML → сериализовать лучшую модель в `artifacts/model.joblib` → вывести offline-метрики (Qini AUC, Uplift@k). |
| T9 | Offline-метрики Uplift | `src/evaluation/metrics.py` | Реализовать: `qini_auc_score(y, uplift, treatment)`, `uplift_at_k(y, uplift, treatment, k)`, `uplift_by_percentile(...)` (для Uplift Curve). |
| T10 | Подключить модель в API | `src/api/main.py` | Убрать mock-логику. При старте загрузить `artifacts/model.joblib` через lifespan. `/predict_uplift` должен вызывать `model.predict_uplift(features)` и возвращать реальный скор + рекомендованный treatment. |
| T11 | Makefile | `Makefile` | Таргеты: `install`, `train`, `serve` (`uvicorn src.api.main:app`), `test`, `lint` (`ruff check src/ tests/`). |
| T12 | Dockerfile + docker-compose.yml | `Dockerfile`, `docker-compose.yml` | Многостадийный Dockerfile (builder + runtime). `docker compose up` запускает API на :8000. |

### Фаза 3: Интеграция с DPE (COULD — связывает два проекта)

| # | Задача | Описание |
|---|---|---|
| T13 | Data connector для DPE | Скрипт/модуль, который читает `Z:\pet-project\dynamic-pricing-engine\dpe_database.db` (таблица `simulation_analytics`), формирует `treatment` (ADDITIVE=1, MULTIPLICATIVE=0), `conversion` (accepted) и набор признаков для обучения Uplift-модели. |
| T14 | Интеграция в pricing pipeline DPE | Добавить HTTP-вызов к Causal Pricing Engine из DPE API для получения персонального treatment-решения перед расчётом Surge. |
| T15 | Switchback → CUPED evaluation pipeline | End-to-end скрипт: прочитать данные эксперимента из DPE → применить CUPED → рассчитать p-value и confidence interval → вывести отчёт. |

---

## 6. Как проверить, что всё работает

```bash
# 1. Тесты проходят без ошибок
cd Z:\pet-project\causal-pricing-engine
pytest tests/ -v

# 2. API запускается и отвечает
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# GET http://localhost:8000/health → {"status": "ok"}
# POST http://localhost:8000/predict_uplift с телом → реальный uplift_score

# 3. Training pipeline отрабатывает
python scripts/train.py
# → artifacts/model.joblib создан
# → метрики Qini AUC выведены в stdout
```

---

## 7. Чего НЕ нужно делать

- **Не переписывать** DPE — он самодостаточный проект. Интеграция через HTTP или чтение SQLite.
- **Не добавлять** Streamlit-дашборд в первом MVP — приоритет на ядро (модели + API + тесты).
- **Не использовать** `causalml` как основную реализацию — она нужна только для сверки (benchmark). Все мета-лёрнеры (S/T/X) реализуются руками поверх `BaseUpliftModel`.
- **Не фиксировать** конкретный base_learner — архитектура должна позволять подставить любой sklearn-совместимый estimator (CatBoost, LightGBM, LogisticRegression).

---

## 8. Полезные ссылки и литература

| Тема | Источник |
|---|---|
| Meta-Learners (S/T/X) | Künzel et al. (2019) — "Metalearners for estimating heterogeneous treatment effects using machine learning" |
| Double ML | Chernozhukov et al. (2018) — "Double/debiased machine learning for treatment and structural parameters" |
| CUPED | Deng et al. (2013) — "Improving the Sensitivity of Online Controlled Experiments" (Microsoft) |
| Switchback Design | Bojinov & Shephard (2019) — "Time Series Experiments and Causal Estimands" |
| EconML docs | https://econml.azurewebsites.net/ |
| CausalML docs | https://causalml.readthedocs.io/ |
| Qini Curve | Radcliffe (2007) — "Using control groups to target on predicted lift" |
