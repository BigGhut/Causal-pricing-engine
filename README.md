# 🎯 Causal Pricing & Uplift Experimentation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: CausalML / EconML](https://img.shields.io/badge/ML-CausalML%20%7C%20EconML-orange.svg)](https://github.com/uber/causalml)

Сервис причинно-следственного анализа (**Causal ML**), **Uplift-моделирования** и продвинутого A/B-тестирования для решения задач динамического ценообразования, расчета персональных скидок и оптимизации Surge-надбавок.

---

## 🎯 Цели и Задачи Проекта

В отличие от классического ML, который предсказывает абсолютную конверсию $P(\text{Conversion} | \text{Features}, \text{Treatment})$, данный движок фокусируется на **приращенном эффекте (Uplift)**:
$$\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X = x]$$

Где:
- $Y^{(1)}$ — результат при назначении воздействия (например, предоставление скидки 10% или Surge +150₽).
- $Y^{(0)}$ — результат без воздействия (базовый тариф).
- $\tau(x)$ — чистый прирост конверсии/прибыли (Individual Treatment Effect, ITE).

---

## 📐 Архитектура Проекта

```
causal-pricing-engine/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── causal/
│   │   ├── __init__.py
│   │   ├── uplift_models.py      # S-Learner, T-Learner, X-Learner, Uplift Random Forest
│   │   └── dml_engine.py         # Double Machine Learning (EconML) & Synthetic Control
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── switchback_splitter.py # Пространственно-временной сплиттер для A/B тестов
│   │   └── power_analysis.py     # Оценка стат. мощности и бутстрап
│   └── api/
│       ├── __init__.py
│       └── main.py               # FastAPI сервис выдачи оптимального Treatment
└── tests/
    ├── __init__.py
    └── test_uplift.py
```

---

## 🔬 Ключевой Функционал (Roadmap)

### 1. Uplift-моделирование (Causal Tree / Meta-Learners)
- **T-Learner & X-Learner:** Оценка чистого приращенного дохода от воздействия.
- **Сегментация пользователей по квадрантам:**
  - *Persuadables (Колеблющиеся):* Купят только с воздействием ($\tau(x) > 0$) — **Целевая группа!**
  - *Sure Things (Уверенные):* Купят в любом случае ($\tau(x) \approx 0$) — **Не тратим бюджет.**
  - *Lost Causes (Безнадежные):* Не купят в любом случае ($\tau(x) \approx 0$) — **Не тратим бюджет.**
  - *Sleeping Dogs (Раздражаемые):* Воздействие ухудшает конверсию ($\tau(x) < 0$) — **Исключаем.**

### 2. Double Machine Learning (DML)
- Избавление от смещения при нерандомизированных данных (Confounding Bias).
- Использование моделей CatBoost / LightGBM для ортогонализации признаков и воздействия.

### 3. Advanced Experimentation Framework
- **Switchback A/B Splitter:** Пространственно-временное разделение для маркетплейсов и сервисов такси.
- **Synthetic Control:** Оценка причинно-следственного эффекта ценовых изменений на исторических данных без проведения честных A/B тестов.
- **Variance Reduction (CUPED / CUPAC):** Снижение дисперсии метрик для ускорения экспериментов.

---

## 🛠️ Запуск и Разработка

```bash
# 1. Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Запуск тестов
pytest tests/

```

---

## 🚀 Phase 3–4 Integration & Deployment

### Сервисы и Порты
- **Dynamic Pricing Engine (DPE):** `http://localhost:8000`
- **Causal Pricing Engine (CPE):** `http://localhost:8100`

### Быстрый старт & Команды (Makefile)
```bash
# Обучение модели на реальной БД DPE
make train-dpe

# Запуск CUPED / Switchback оценки эксперимента на DPE данных
make evaluate

# Запуск API сервиса CPE
make serve

# Запуск полного набора unit-тестов
make test

# Проверка работоспособности сервисов (Smoke test)
make smoke         # CPE only
make smoke-dpe     # CPE + DPE integration
```

### Docker & Dual Stack Deployment
Для воспроизводимого запуска всей связки (CPE + DPE):
```bash
docker compose -f docker-compose.full.yml up --build
```

### Ключевые переменные окружения (Environment Variables)
- `CPE_DPE__DB_PATH`: Путь к SQLite базовым данным DPE (по умолчанию `/app/data/dpe_database.db` в Docker).
- `CPE_RETRAIN_ON_START`: Автоматическое переобучение модели CPE на DPE-данных при старте контейнера (`1` — вкл).
- `CAUSAL_ENGINE_URL`: URL сервиса CPE для обращений из DPE (по умолчанию `http://localhost:8100`).
- `CAUSAL_ENABLED`: Флаг включения вызовов Causal ML из DPE (`true` / `false`).
- `CAUSAL_ENGINE_TIMEOUT_SEC`: Таймаут HTTP запроса DPE -> CPE (default `0.1` sec, fail-open guarantee ≤200ms).
- `CAUSAL_UPLIFT_THRESHOLD`: Порог определения "Sleeping Dogs" (по умолчанию `0.05`).

### Принцип работы Sleeping Dog Override
При запросе цены DPE запрашивает прогнозируемый ITE (Uplift) у CPE (`POST /predict_uplift`).
Если $Uplift < -\text{threshold}$ (пользователь относится к категории **Sleeping Dogs** — повышение цены существенно снижает вероятность заказа), DPE отменяет Surge-надбавку (`CAUSAL_NO_SURGE`), возвращая базовый тариф для сохранения лояльности пользователя.

---

## ⚡ Phase 5: Serve/Train Parity, Observability & Release Hygiene

### Train / Serve Feature Parity
1. **`past_trips`**: Накопленное количество завершенных поездок водителя (или ячейки H3 при отсутствии истории водителя) **строго до** текущей сессии расчета цены.
2. **`avg_surge`**: Накопленное среднее значения `surge_bonus` предыдущих поездок водителя **строго до** текущего запроса (0.0 при `past_trips == 0`).
3. **Parity Enforcement**: Тест `test_history_matches_cpe_connector_parity` гарантирует 1-в-1 совпадение онлайн-накопления `DriverHistoryStore` с оффлайн-трансформациями `load_dpe_data()`.

### Observability в DPE API Response
В ответ эндпоинта `/api/v1/search` добавлены наглядные поля интеграции:
```json
{
  "h3_index": "881180e001fffff",
  "price": 219.0,
  "surge_multiplier": 1.0,
  "explanation": "Graph-based pricing. ...",
  "causal_uplift_score": 0.1508,
  "causal_override": false,
  "causal_recommended_treatment": "DISCOUNT_10_PCT"
}
```
При срабатывании Sleeping Dog override поле `causal_override` устанавливается в `true`, а в `explanation` добавляется суффикс `Causal override (Sleeping Dog): uplift=... < -threshold`.

### One-Shot Local Smoke Harness
Для автоматической проверки интеграции без ручного запуска серверов:
```bash
make smoke-local
# или напрямую:
python scripts/e2e_smoke.py --start-servers --with-dpe
```
Данная команда поднимет фоновые сервисы Uvicorn (CPE :8100, DPE :8000), обучит базовую модель при отсутствии артефакта, проверит эндпоинты `/health`, `/predict_uplift`, `/api/v1/search` и закроет фоновые процессы при завершении.


