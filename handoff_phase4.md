# Handoff Phase 4: Production Hardening CPE ↔ DPE

> **Автор:** оркестратор  
> **Дата:** 2026-08-09  
> **Предыдущий handoff:** [handoff_phase3.md](handoff_phase3.md) — **CLOSED (PASS)**  
> **Fix:** [handoff_phase3_fix.md](handoff_phase3_fix.md) — порт Docker 8100  
> **Аудит Phase 3:** [audit_report_phase3.md](audit_report_phase3.md)  
> **Цель:** убрать «заглушки» интеграции, выровнять feature contract train↔serve↔DPE, сделать Docker/e2e воспроизводимыми, не ломая Phase 1–3.

---

## 1. Контекст: что уже работает (не переписывать)

| Компонент | Путь | Статус |
|:---|:---|:---|
| Meta-learners S/T/X + DML | `src/causal/*` | ✅ |
| Switchback, CUPED, metrics | `src/experiments/*`, `src/evaluation/*` | ✅ |
| DPE connector | `src/data/dpe_connector.py` | ✅ |
| train `--source synthetic\|dpe` | `scripts/train.py` | ✅ |
| evaluate_experiment | `scripts/evaluate_experiment.py` | ✅ |
| CPE API :8100 | `src/api/main.py`, Dockerfile | ✅ |
| DPE → CPE HTTP fail-open | `dynamic-pricing-engine/src/api/main.py` | ✅ MVP |
| docker-compose.full | `docker-compose.full.yml` | ✅ порты |

**Регрессия обязательна:** `pytest tests/ -v` в CPE — **≥26 passed** (можно больше). Не ломать Phase 1–3 API-контракты без миграции.

**Порты (зафиксированы):**

| Сервис | Порт |
|:---|:---|
| DPE API | `:8000` |
| CPE API | `:8100` |

---

## 2. Проблемы, которые Phase 4 закрывает

### P1. Feature zero-fill в runtime (критично для смысла uplift)

В DPE сейчас:

```python
"past_trips": 0.0,
"avg_surge": 0.0,
```

При этом train на DPE (`DPE_FEATURE_COLUMNS`) учится на **реальных** `past_trips` / `avg_surge` из connector.  
→ online-скор смотрит на «водителей без истории» → uplift систематически смещён.

### P2. Жёсткий путь к БД

```python
DEFAULT_DPE_DB_PATH = Path("Z:/pet-project/dynamic-pricing-engine/dpe_database.db")
```

В Docker volume: `/app/data/dpe_database.db` — connector по умолчанию **не видит**.

### P3. Docker train только synthetic

`Dockerfile`: `RUN python scripts/train.py` → artifact на synthetic-фичах  
(`past_trips, avg_surge, price_sensitivity, hour_of_day, segment`),  
а DPE шлёт **DPE_FEATURE_COLUMNS**. API zero-fill'ит чужие ключи → scoring «технически жив», но бесполезен.

### P4. Нет e2e-проверки цепочки

Нет автоматического сценария: CPE health → DPE search → (опционально) `CAUSAL_NO_SURGE`.  
Нет логов uplift в DPE response/explanation.

### P5. Магические константы

- threshold Sleeping Dog `-0.05` захардкожен в DPE (дублирует `cfg.api.uplift_threshold` CPE)
- timeout `0.1` не конфигурируется

### P6. Процесс отчётности

Воркер **не перезаписывает** `audit_report_phase3.md` / будущий `audit_report_phase4.md`.  
Сдавать работу файлом **`worker_report_phase4.md`**. Финальный PASS пишет оркестратор.

---

## 3. Контракт признаков (зафиксировать)

Единый online/offline контракт для DPE-пути (уже есть):

```python
DPE_FEATURE_COLUMNS = [
    "distance_km",
    "duration_sec",
    "price",
    "surge_bonus",
    "hour_of_day",
    "past_trips",
    "avg_surge",
]
```

| Признак | Offline (connector) | Online (DPE → CPE) |
|:---|:---|:---|
| `distance_km` | `simulation_analytics` | OD distance trip |
| `duration_sec` | DB | estimated duration |
| `price` | DB | **proposed_price** после surge (как сейчас) |
| `surge_bonus` | DB | **surge_bonus** после compute (как сейчас) |
| `hour_of_day` | from timestamp | local hour |
| `past_trips` | cumcount per driver | **нужно заполнить** (T18) |
| `avg_surge` | expanding mean shift(1) | **нужно заполнить** (T18) |

Synthetic path (`FEATURE_COLUMNS`) **оставляем** для unit-тестов и dev; production/Docker default для dual-stack — **DPE schema**.

---

## 4. План задач

### T18: Driver feature store / history в DPE (MUST)

**Проблема:** `past_trips` / `avg_surge` = 0.

**Где:** преимущественно `Z:\pet-project\dynamic-pricing-engine`

**Что сделать (выбрать один вариант, предпочтение A):**

#### Вариант A (рекомендуемый): in-memory / SQLite history sidecar

1. Модуль `src/causal_features.py` (или `src/features/driver_history.py`) в **DPE**:
   - `get_driver_features(driver_id: str) -> dict` → `{past_trips, avg_surge}`
   - `update_driver_features(driver_id, surge_bonus)` после завершения поездки / accept (если есть точка)
2. Источник bootstrap (опционально): прочитать `dpe_database.db` → `simulation_analytics`, посчитать те же cumstats, что connector (согласованность offline/online).
3. В `request_price` подставлять реальные значения вместо `0.0`.

Если `driver_id` в search request **нет** — использовать `payload.search_id` **нельзя** как driver history key (это search, не водитель). Тогда:
- добавить опциональное поле `driver_id` в `SearchRequest` **или**
- fallback: агрегаты по `h3_cell` / node (`past_trips` = count recent trips in cell, `avg_surge` = mean recent surge) — задокументировать в коде.

#### Вариант B (быстрее, слабее): cell-level features

Без driver_id: `past_trips` / `avg_surge` из Redis/feature_store по node/h3 за последнее окно.

**Критерий приёмки:**

- [ ] В теле POST к CPE `past_trips` и `avg_surge` **не всегда 0** на повторных вызовах с одним driver/cell (тест или скрипт).
- [ ] Fail-open CPE сохранён (timeout ≤ configurable, default 100ms).
- [ ] Тест: mock history → payload содержит ожидаемые фичи (unit в DPE или e2e в CPE `tests/`).

**Не делать:** менять схему таблицы `simulation_analytics` (только read + optional cache table `driver_feature_cache` — можно).

---

### T19: Конфигурируемый путь к DPE DB (MUST)

**Файлы CPE:** `src/data/dpe_connector.py`, `src/config.py`, `configs/config.yaml`

1. Добавить в config секцию, например:

```yaml
dpe:
  db_path: "Z:/pet-project/dynamic-pricing-engine/dpe_database.db"
```

2. Резолв пути (приоритет):
   1. явный аргумент `db_path` / CLI `--dpe-db-path`
   2. env `CPE_DPE__DB_PATH` (через pydantic-settings nested)
   3. YAML
   4. fallbacks по порядку существования:
      - `/app/data/dpe_database.db` (Docker)
      - `../dynamic-pricing-engine/dpe_database.db` relative to project root
      - legacy `Z:/pet-project/dynamic-pricing-engine/dpe_database.db`

3. `load_dpe_data()` использует этот резолвер; **убрать единственный hardcode** как единственный default.

**Критерий приёмки:**

- [ ] `CPE_DPE__DB_PATH=/app/data/dpe_database.db load_dpe_data()` работает без правки кода.
- [ ] Тест с `tmp_path` sqlite fixture (не только real DB) — schema + treatment ∈ {0,1}.
- [ ] Старые вызовы `load_dpe_data()` без аргументов на dev-машине с DPE DB — не ломаются.

---

### T20: Docker artifact = DPE schema (MUST)

**Файлы:** `Dockerfile`, опционально `scripts/docker_entrypoint.sh`, `docker-compose.full.yml`

1. **Builder:** если на этапе build БД недоступна — оставить synthetic **только** как fallback, но:
2. **Runtime entrypoint (предпочтительно):**
   - если существует `$CPE_DPE_DB` / `/app/data/dpe_database.db` **и** нет валидного artifact / флаг `CPE_RETRAIN_ON_START=1`:
     - `python scripts/train.py --source dpe --dpe-db-path ...`
   - затем `uvicorn ... --port 8100`
3. В `docker-compose.full.yml`:
   - volume DB (уже есть)
   - env: `CPE_DPE__DB_PATH=/app/data/dpe_database.db`
   - опционально `CPE_RETRAIN_ON_START=1` для dev

4. Artifact payload уже содержит `feature_columns` — API должен **логировать** `model_name` + `feature_columns` при старте (уже частично есть).

**Критерий приёмки:**

- [ ] После `docker compose -f docker-compose.full.yml up --build` (или solo + volume):
  - `curl localhost:8100/health` → model_loaded true
  - `POST /predict_uplift` с **DPE_FEATURE_COLUMNS** → 200, не 503
- [ ] `feature_columns` в загруженном artifact ⊆ DPE schema (или равен), если retrain-on-start отработал.
- [ ] Документировать в README 5–10 строк: dual compose, ports, retrain env.

**Не делать:** убирать synthetic train path для локальной разработки (`make train`).

---

### T21: E2E smoke script (MUST)

**Файл CPE:** `scripts/e2e_smoke.py` (или `scripts/smoke_dpe_cpe.py`)

Сценарий:

1. `GET {CPE}/health` — model_loaded
2. `POST {CPE}/predict_uplift` с dummy DPE features — 200 + uplift_score float
3. Опционально (`--with-dpe`): `POST {DPE}/api/v1/search` (минимальный валидный payload DPE) — 200; при недоступном CPE DPE всё равно 200 (fail-open)
4. Exit code ≠ 0 при провале обязательных шагов
5. Makefile: `make smoke` → `python scripts/e2e_smoke.py`

**Критерий приёмки:**

- [ ] Скрипт работает против уже поднятых сервисов (не обязан сам поднимать docker).
- [ ] Тест **или** CI-док: как запускать smoke локально.
- [ ] Не требует GPU / econml-only path.

---

### T22: Конфиг интеграции DPE (SHOULD → почти MUST)

**Файлы DPE:** `src/config.py`, env

```python
CAUSAL_ENGINE_URL: str = ...
CAUSAL_ENGINE_TIMEOUT_SEC: float = 0.1
CAUSAL_UPLIFT_THRESHOLD: float = 0.05  # Sleeping Dog if score < -threshold
CAUSAL_ENABLED: bool = True
```

В `main.py`:

- если `not CAUSAL_ENABLED` — не вызывать HTTP
- timeout / threshold из config
- при override: писать в `explanation` / response поле (если есть) `causal_uplift_score`, `causal_override: true`

**Критерий приёмки:**

- [ ] `CAUSAL_ENABLED=false` → ни одного HTTP к CPE (можно проверить mock/log).
- [ ] threshold не захардкожен числом в `main.py`.

---

### T23: Тесты на train dpe + connector fixture (SHOULD)

**Файлы CPE:** `tests/test_train_dpe.py`, расширить `test_dpe_connector.py`

1. Минимальный sqlite fixture (несколько строк `simulation_analytics`) в `tmp_path`
2. `load_dpe_data(tmp_db)` без skip
3. `train_and_select(source="dpe", dpe_db_path=tmp_db, random_state=0)` → artifact keys: `model`, `feature_columns`, `source=="dpe"`
4. `feature_columns == DPE_FEATURE_COLUMNS`

**Критерий приёмки:**

- [ ] Тесты **не skip** на машине без `Z:\...dpe_database.db`
- [ ] Полный suite green

---

### T24: Наблюдаемость и safety (COULD)

1. CPE: middleware/log latency `predict_uplift` (ms)
2. CPE: `/health` расширить: `feature_columns`, `source` из artifact (без утечки модели)
3. DPE: counter/log `causal_override_count` (print ok для pet)
4. Guard: если len(features)==0 или NaN — CPE 422, DPE fail-open

---

### T25: Документация (SHOULD)

**Файл:** `README.md` (CPE)

Секция **Phase 3–4 Integration**:

- порты 8000 / 8100
- `make train-dpe`, `make evaluate`, `make smoke`
- `docker compose -f docker-compose.full.yml up`
- env: `CPE_DPE__DB_PATH`, `CPE_RETRAIN_ON_START`, `CAUSAL_ENGINE_URL`, `CAUSAL_ENABLED`
- квадранты uplift + Sleeping Dog override

---

## 5. Порядок выполнения

```
T19 (db path config) ─────────────────────────────┐
T23 (sqlite fixture tests) ───────────────────────┤
                                                   ├─→ T20 (Docker retrain) ─→ T21 (e2e smoke)
T18 (online features DPE) ────────────────────────┤
T22 (DPE config flags) ───────────────────────────┘
T24 (observability) — parallel, optional
T25 (README) — в конце
```

**Рекомендуемый порядок:** T19 → T23 → T18 → T22 → T20 → T21 → T25 → (T24).

---

## 6. Чего НЕ нужно делать

- **Не** переписывать S/T/X/DML и offline metrics Phase 1–2.
- **Не** менять схему `simulation_analytics` / ломать DPE switchback A/B.
- **Не** блокировать DPE при падении CPE (fail-open священен).
- **Не** поднимать timeout CPE-вызова выше **200 ms** без явного обоснования в worker_report.
- **Не** добавлять Streamlit / Kafka / новый сервис «feature-store» как отдельный контейнер (in-process / sqlite cache — ок).
- **Не** перезаписывать `audit_report_*.md` оркестратора — только `worker_report_phase4.md`.
- **Не** менять default port CPE с 8100 и DPE с 8000.

---

## 7. Критерии закрытия Phase 4

| # | Критерий |
|:---|:---|
| 1 | `pytest` CPE green, новые тесты T23 не skip-only |
| 2 | Online payload: `past_trips` / `avg_surge` заполняются осмысленно (T18) |
| 3 | `CPE_DPE__DB_PATH` / Docker path работает (T19) |
| 4 | Docker (full или documented path): health + predict на DPE-schema (T20) |
| 5 | `make smoke` / e2e script зелёный против running stack (T21) |
| 6 | DPE threshold/timeout/enable из config (T22) |
| 7 | README обновлён (T25) |
| 8 | Сдан `worker_report_phase4.md` (не затирая audit) |

Оркестратор после сдачи: re-audit → `audit_report_phase4.md` → PASS/FAIL.

---

## 8. Как проверить локально (чеклист воркера)

```bash
# CPE
cd Z:\pet-project\causal-pricing-engine
python -m pytest tests/ -v
python scripts/train.py --source dpe
python scripts/evaluate_experiment.py --source dpe

# CPE serve
uvicorn src.api.main:app --host 0.0.0.0 --port 8100

# DPE serve (другой терминал)
cd Z:\pet-project\dynamic-pricing-engine
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Smoke
cd Z:\pet-project\causal-pricing-engine
python scripts/e2e_smoke.py --with-dpe

# Docker (если доступен)
docker compose -f docker-compose.full.yml up --build
curl http://localhost:8100/health
curl http://localhost:8000/health
```

---

## 9. Формат сдачи воркера

Создать **`worker_report_phase4.md`**:

```markdown
# Worker Report Phase 4
## Сделано
- T18: ...
- T19: ...
## Тесты
```
pytest output
```
## Smoke
...
## Не сделано / отклонения
...
```

Без перезаписи `handoff_*.md` и `audit_report_*.md`.

---

## 10. Связь с литературой / продуктом (кратко)

| Тема | Зачем в Phase 4 |
|:---|:---|
| Persuadables / Sleeping Dogs | Online features должны отражать сегмент, иначе override случайный |
| Train/serve skew | Главный риск текущей интеграции |
| CUPED eval (Phase 3) | После T18 имеет смысл переснять evaluate на новых симуляциях |

---

**Итог для воркера:** Phase 4 — hardening, не greenfield. Приоритет: **train/serve feature parity + config + smoke**. Красота моделей вторична.
