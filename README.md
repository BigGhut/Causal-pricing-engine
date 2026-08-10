# Causal Pricing Engine (CPE)

**Одной строкой:** sidecar на Causal ML / uplift, который оценивает индивидуальный эффект воздействия (ITE) и подсказывает Dynamic Pricing Engine (DPE), **кому не стоит давать агрессивный surge**.

Стек: Python ≥ 3.10 · scikit-learn · FastAPI · (опц.) EconML · Docker

Репозиторий: [BigGhut/Causal-pricing-engine](https://github.com/BigGhut/Causal-pricing-engine)  
Компаньон: [BigGhut/Dynamic-pricing-engine](https://github.com/BigGhut/Dynamic-pricing-engine)

---

## Что это / что это не

**Это:**

- микросервис uplift-скоринга (`POST /predict_uplift`, порт `:8100`);
- связка с DPE: timeout ≤ 200 ms, **fail-open** (если CPE недоступен — обычное rule-based ценообразование);
- правило **Sleeping Dog**: при \(\hat\tau(x) < -\theta\) (по умолчанию \(\theta = 0.05\)) DPE сбрасывает surge к базовому тарифу (`CAUSAL_NO_SURGE`);
- честное демо и portfolio-proof: роли берутся с реальных holdout-строк, evidence пишется в `docs/evidence/`.

**Это не:**

- не claim city-scale lift с продакшен A/B;
- не «магический AUC 0.99» на декоративных фичах;
- дефолтное демо — **синтетика с заложенным HTE** (heterogeneous treatment effect): показывает пайплайн и политику, а не городской эффект.

Подробный разбор задачи, trade-off и оговорок: [CASE_STUDY.md](CASE_STUDY.md).

---

## Почему не классический ML?

Классический supervised ML отвечает: «какая вероятность конверсии *при этих фичах и treatment?*»  
\(P(Y=1 \mid X, T)\)

Проблема: в одну кучу попадают:

| Группа | Поведение | Если всегда «лечить» treatment’ом |
|:---|:---|:---|
| **Persuadables** | конвертятся *из‑за* воздействия | правильный spend |
| **Sure things / lost causes** | исход тот же с treatment и без | зря потраченный стимул |
| **Sleeping Dogs** | воздействие *вредит* (отказ, churn) | активный ущерб |

Uplift / causal ML целится в **чистый прирост** — индивидуальный эффект воздействия (ITE):

\[
\tau(x) = \mathbb{E}[Y^{(1)} - Y^{(0)} \mid X = x]
\]

- \(Y^{(1)}\) — исход *с* воздействием (скидка / surge-бонус),
- \(Y^{(0)}\) — исход *без* воздействия (control),
- \(\tau(x)\) — то, что можно отнести именно к treatment.

**Аналогия:** классика говорит «человеку куртка влезет», uplift — «ему *нужна* именно эта куртка, или без неё всё равно / хуже».

> **Границы claim.** Qini в отчётах — **нормализованный** (\(\approx [-1, 1]\), random \(\approx 0\)). Логи симуляции DPE — observational / switchback, возможна post-treatment запутанность признаков (см. case study).

---

## Архитектура

```text
                                +-------------------------------+
                                |  Dynamic Pricing Engine (DPE) |
                                |       FastAPI  :8000          |
                                +---------------+---------------+
                                                |
                                    POST /predict_uplift
                                  (timeout ≤ 200ms, fail-open)
                                                |
                                                v
+-----------------------+       +---------------+---------------+
|  DriverHistoryStore   | ----> |  Causal Pricing Engine (CPE)  |
| (train/serve parity)  |       |       FastAPI  :8100          |
+-----------------------+       +---------------+---------------+
                                                |
                                        [T-Learner / DML]
                                                |
                                                v
                                 ITE-score + recommended_treatment
```

### Ключевые идеи

- **Persuadables** (\(\tau > 0\)) — treatment помогает; **Sleeping Dogs** (\(\tau < 0\)) — treatment вредит.
- **Sleeping Dog override** в DPE: \(\hat\tau < -0.05\) → base fare, `CAUSAL_NO_SURGE`.
- **Fail-open:** CPE лежит или тормозит — DPE не блокирует выдачу цены.
- **Train/serve parity:** `past_trips`, `avg_surge` считаются *as-of prior trip* и офлайн (`load_dpe_data`), и онлайн (`DriverHistoryStore`).

---

## Быстрый старт

Нужен **Python ≥ 3.10**. Команды ниже работают в PowerShell, cmd и bash (Make опционален).

```powershell
cd causal-pricing-engine
pip install -r requirements.txt
pip install -e .

# Честное in-process демо (роли = знаки score на holdout)
python scripts/demo.py

# Portfolio proof: живой HTTP CPE + то же правило, что у DPE → docs/evidence/
python scripts/portfolio_proof.py
# или: make proof
```

Опционально:

```powershell
# Тяжёлый causal-стек (EconML, CatBoost, LightGBM)
pip install -e ".[causal]"

# Сервис CPE
uvicorn src.api.main:app --host 0.0.0.0 --port 8100
# make serve

# Два сервиса + proof против live DPE
# Терминал 1: uvicorn src.api.main:app --port 8100
# Терминал 2: cd ../dynamic-pricing-engine && uvicorn src.api.main:app --port 8000
python scripts/portfolio_proof.py --with-dpe
```

Полезные цели Makefile: `make install` · `make demo` · `make proof` · `make train` · `make test` · `make smoke-local`.  
Без Make те же скрипты вызываются через `python scripts/...`.  
Для `make lint` нужен `ruff` (`pip install -e ".[dev]"`).

**Дальше читать:** [CASE_STUDY.md](CASE_STUDY.md) · [docs/evidence/latest_proof.md](docs/evidence/latest_proof.md) · [репозиторий DPE](https://github.com/BigGhut/Dynamic-pricing-engine) (локально: `CAUSAL.md` рядом с DPE)

---

## Пример метрик (captured evidence)

Источник: автогенерация `scripts/portfolio_proof.py` от **2026-08-09 11:04:40 UTC**  
файл: [`docs/evidence/latest_proof.md`](docs/evidence/latest_proof.md) — **не** ручной маркетинг.

| Сигнал | Значение |
|:---|:---|
| Порог политики | ±0.05 |
| Holdout Qini (нормализованный) | **+0.0838** (random null +0.0092) |
| Holdout Uplift@30% | **+0.2302** |
| Модель в proof | `t_learner` · source `portfolio_proof_synthetic_hte` |
| Sleeping Dog (пример) | \(\hat\tau \approx -0.415\) → `causal_override = true` |

Интерпретация узкая: *на данных с HTE мы умеем скорить ITE, провести через HTTP и применить то же override-правило, что DPE — без падения ценового пути при недоступности модели.*

---

## Честное демо и proof

### Демо (`scripts/demo.py`)

Не рисует декоративные векторы фичей. Скорит holdout и выбирает **три реальные строки**:

| Роль | Как выбираем | Политика при ±0.05 |
|:---|:---|:---|
| **persuadable** | max \(\hat\tau\) | применить treatment |
| **neutral** | \(\hat\tau \approx 0\) | baseline |
| **sleeping_dog** | min \(\hat\tau\) | подавить surge |

Если явных +/− HTE нет — демо **падает** (или уходит в synthetic retry), а не «придумывает» красивую картинку.

### Proof (`scripts/portfolio_proof.py`)

1. Те же честные picks  
2. Поднимает CPE при необходимости  
3. `POST /predict_uplift` на каждую роль (assert кодов treatment)  
4. То же правило, что DPE: \(\hat\tau < -\theta \rightarrow\) causal_override  
5. Пишет captured output в `docs/evidence/latest_proof.md`  

`--with-dpe` дополнительно дергает live `/api/v1/search` (graph-фичи могут отличаться; override там информативный, не жёсткий assert).

---

## API (эскиз)

Сервис: FastAPI `:8100`  
Liveness: `GET /health`

### `POST /predict_uplift`

**Request:**

```json
{
  "user_id": "u_42",
  "features": {
    "distance_km": 7.0,
    "duration_sec": 900.0,
    "price": 350.0,
    "surge_bonus": 50.0,
    "hour_of_day": 18.0,
    "past_trips": 12.0,
    "avg_surge": 1.25
  }
}
```

**Response (пример формы):**

```json
{
  "user_id": "u_42",
  "uplift_score": 0.6899,
  "recommended_treatment": "DISCOUNT_10_PCT",
  "optimal_discount_pct": 10.0,
  "model_name": "t_learner"
}
```

Коды `recommended_treatment` (порог по умолчанию 0.05):

| Условие | Код | Смысл |
|:---|:---|:---|
| \(\hat\tau > +\theta\) | `DISCOUNT_10_PCT` | treatment / скидочный путь |
| \(\|\hat\tau\| \le \theta\) | `NO_DISCOUNT` | нейтрально |
| \(\hat\tau < -\theta\) | `NO_DISCOUNT_AVOID` | Sleeping Dog — избегать стимула / surge |

Перед первым serve: `python scripts/train.py` (артефакт в `artifacts/`).

---

## Структура проекта

```text
causal-pricing-engine/
├── artifacts/                 # joblib-модели (игнорируются git, кроме .gitkeep)
├── configs/                   # config.yaml
├── docs/
│   ├── archive/               # handoff / audit / multi-agent процесс
│   └── evidence/              # captured proof (latest_proof.md)
├── scripts/
│   ├── demo.py                # честное in-process демо
│   ├── portfolio_proof.py     # HTTP + evidence для портфолио
│   ├── train.py               # обучение и выбор кандидата
│   ├── e2e_smoke.py           # e2e / smoke
│   ├── evaluate_experiment.py # offline eval (в т.ч. DPE source)
│   └── diagnostics_leakage.py # диагностика leakage
├── src/
│   ├── api/main.py            # FastAPI :8100
│   ├── causal/                # meta-learners, DML
│   ├── data/                  # feature contract, DPE connector, synthetic
│   ├── evaluation/            # Qini, Uplift@k
│   └── experiments/           # switchback / power helpers
├── tests/
├── CASE_STUDY.md
├── Dockerfile / docker-compose*.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Что улучшил бы дальше

1. Больше switchback-логов + CUPED-readout политики override (не только Qini на holdout).  
2. Multi-treatment (уровни surge / глубины скидки), а не binary treat/control.  
3. Общий online feature store, чтобы graph-фичи под нагрузкой совпадали с offline training columns.

---

## Процесс и ссылки

- Архив multi-agent handoff/audit: [`docs/archive/`](docs/archive/README.md)  
- Интеграция с DPE: [Dynamic-pricing-engine](https://github.com/BigGhut/Dynamic-pricing-engine) · локально `../dynamic-pricing-engine/CAUSAL.md` (на remote может ещё не быть запушен)  
- Case study: [CASE_STUDY.md](CASE_STUDY.md)  
- Evidence: [docs/evidence/latest_proof.md](docs/evidence/latest_proof.md)

Локально компаньон обычно лежит рядом: `../dynamic-pricing-engine` (на GitHub — отдельный public-репо по ссылке выше).
