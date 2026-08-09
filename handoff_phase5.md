# Handoff Phase 5: Serve/Train Parity, Observability & Release Hygiene

> **Автор:** оркестратор  
> **Дата:** 2026-08-09  
> **Предыдущий:** [handoff_phase4.md](handoff_phase4.md) — **CLOSED (PASS)**  
> **Аудит:** [audit_report_phase4.md](audit_report_phase4.md)  
> **Сдача Phase 4:** [worker_report_phase4.md](worker_report_phase4.md)  
> **Цель:** закрыть train/serve skew, сделать causal-override наблюдаемым end-to-end, зафиксировать release hygiene (тесты/smoke/коммиты). **Не** greenfield-модели.

---

## 1. Контекст (не ломать)

| Проект | Путь | Порт |
|:---|:---|:---|
| CPE | `Z:\pet-project\causal-pricing-engine` | `:8100` |
| DPE | `Z:\pet-project\dynamic-pricing-engine` | `:8000` |

**Регрессия baseline (должна остаться зелёной или вырасти):**

- CPE: `pytest tests/ -v` → **≥ 29 passed**
- DPE: `pytest tests/ -v` → **≥ 15 passed**

**Священное:**

- Fail-open DPE→CPE (`CAUSAL_ENABLED`, timeout ≤ **200 ms**)
- Порты 8100 / 8000
- Не переписывать S/T/X/DML ядро без необходимости
- Не затирать `handoff_*.md`, `audit_report_*.md` → сдавать **`worker_report_phase5.md`**

---

## 2. Проблемы Phase 4 → задачи Phase 5

| # | Проблема (из audit P4) | Задача |
|:---|:---|:---|
| P1 | Online bootstrap `past_trips` = **полный** count; offline connector = **cumcount до текущей** | **T26** |
| P2 | Cell-fallback cold-start: bootstrap DB только по `driver_id` | **T27** |
| P3 | Override Sleeping Dog не виден в API response (только `test_group` / formula) | **T28** |
| P4 | Live smoke хрупок (нужны уже поднятые сервисы); нет «одной кнопки» | **T29** |
| P5 | Много uncommitted в CPE + DPE; нет atomic snapshot | **T30** |
| P6 | README/DPE docs не описывают Phase 5 parity | **T31** |

---

## 3. Задачи

### T26: Online/offline parity для `past_trips` / `avg_surge` (MUST)

**Где:** `dynamic-pricing-engine/src/features/driver_history.py` (+ тесты)

**Контракт offline (источник истины — CPE connector):**

```text
past_trips[i]  = число поездок driver_id со timestamp < ts_i   # cumcount, без текущей
avg_surge[i]   = mean(surge_bonus предыдущих поездок)        # shift(1).expanding().mean, 0 если пусто
```

**Сейчас online (баг parity):** после bootstrap `get_driver_features` возвращает count **включая** все строки истории → на «следующей» поездке это уже N, тогда как offline на N-й строке было N-1.

**Требование:**

1. `get_driver_features(driver_id)` должен возвращать статистику **строго до** текущего вызова (как offline).
2. `record_trip(...)` вызывается **после** pricing / causal call (уже так в `main.py`) — убедиться, что score видит pre-update state.
3. Bootstrap из SQLite должен воспроизводить **последнее pre-next-trip** состояние:
   - после обработки всех historical rows store держит counts = total historical trips;
   - следующий `get` без `record` → past_trips = total (это корректно для **новой** поездки после истории);
   - для **симуляции N-й поездки в replay** нужен API `get_features_as_of(driver_id, before_count)` **или** тест, который:
     - bootstrap N rows;
     - `get` → past_trips == N (следующая поездка);
     - `record` → next get → N+1.
4. Добавить unit-тест, сверяющий **батч** `DriverHistoryStore` replay с `load_dpe_data()` на одной и той же tmp sqlite:
   - для каждой строки в порядке времени: `get_driver_features` **до** `record_trip` должен совпасть с `past_trips`/`avg_surge` из connector (tolerance `1e-6` на avg).

**Критерий приёмки:**

- [ ] Тест `test_history_matches_cpe_connector_parity` (DPE или CPE) **PASS**
- [ ] Документ 5–10 строк в docstring `DriverHistoryStore`: определение past_trips/avg_surge
- [ ] Существующие DPE history tests не сломаны

---

### T27: Cell-level bootstrap + fallback parity (SHOULD → почти MUST)

**Где:** `driver_history.py`, bootstrap query

1. При bootstrap читать из `simulation_analytics` не только `driver_id`, но и пространственный ключ, если есть:
   - если в таблице **нет** h3 — skip cell bootstrap с логом (не падать);
   - если DPE при `record_trip` пишет cell — online cell stats должны расти.
2. Опционально: при bootstrap, если `trip_id`/`search` не даёт h3, cell stats остаются runtime-only (задокументировать).
3. Тест: два `record_trip` с одним `h3_cell` без `driver_id` → `get_driver_features(None, h3)` → past_trips==2 после records; **до** первого record == 0.

**Критерий приёмки:**

- [ ] Cell fallback покрыт тестом (уже частично есть — усилить)
- [ ] Нет exception при bootstrap на текущей схеме DB

---

### T28: Наблюдаемость causal override в DPE API (MUST)

**Где:** `dynamic-pricing-engine/src/api/main.py`, schemas/response

1. При успешном ответе CPE сохранять в локальные переменные:
   - `causal_uplift_score: float | None`
   - `causal_override: bool`
   - `causal_model_name: str | None` (если есть в JSON)
2. Прокинуть в **response** `PriceResponse` (или расширить schema):
   - `causal_uplift_score: Optional[float] = None`
   - `causal_override: bool = False`
   - `causal_recommended_treatment: Optional[str] = None`
3. Добавить в `explanation` суффикс при override, например:  
   `Causal override (Sleeping Dog): uplift=-0.12 < -threshold.`
4. При fail-open / `CAUSAL_ENABLED=false`: поля `None`/`False`, без exception.
5. Тест: mock httpx или unit на mapping score→override fields (не обязателен live CPE).

**Критерий приёмки:**

- [ ] JSON search response содержит новые поля
- [ ] При `CAUSAL_ENABLED=false` поля пустые/false
- [ ] Fail-open сохранён
- [ ] `e2e_smoke.py --with-dpe` печатает `causal_*` если присутствуют (мягкая проверка)

---

### T29: One-shot local smoke harness (MUST)

**Где:** CPE `scripts/e2e_smoke.py` и/или `scripts/smoke_stack.ps1` / `Makefile`

Сейчас smoke падает, если сервисы не подняты. Нужно:

**Вариант A (предпочтительный):** Makefile targets:

```make
smoke-local:
	# start CPE+DPE in background, wait health, run e2e_smoke --with-dpe, tear down
```

**Вариант B:** `e2e_smoke.py --start-servers` поднимает uvicorn subprocess (CPE + optional DPE), health-wait ≤30s, run checks, terminate.

Требования:

1. Windows-friendly (PowerShell или Python subprocess — **не** bash-only).
2. Не требует Docker.
3. Если порт занят — reuse existing server (не fail), задокументировать.
4. Exit code 0 только при green checks.
5. `make smoke-local` или `python scripts/e2e_smoke.py --start-servers --with-dpe`

**Критерий приёмки:**

- [ ] На чистой машине с установленными deps (без уже запущенных серверов) одна команда даёт PASS **или** понятный skip с reason (если DPE graph data отсутствует — skip DPE part, CPE part обязателен).
- [ ] Задокументировано в README (T31)

---

### T30: Git snapshot commits (SHOULD)

**Два репозитория**, **отдельные** коммиты (не force-push, не amend чужого history).

#### CPE (`causal-pricing-engine`)

Предлагаемый message (можно 1–2 коммита):

```text
feat: phases 3–5 causal pricing engine (DPE connector, train/eval, docker, smoke)

- uplift MVP + DML + API :8100
- DPE connector, evaluate CUPED, train --source dpe
- docker entrypoint retrain, e2e smoke, config CPE_DPE__*
- handoffs/audits/worker reports
```

**Не коммитить:**

- `artifacts/model.joblib` (если в `.gitignore` — ок)
- `__pycache__`, `.pytest_cache`, venv
- `.worker_prompt_phase*.txt` (можно удалить или gitignore)

#### DPE (`dynamic-pricing-engine`)

```text
feat: causal engine integration (fail-open uplift, driver history, config flags)
```

**Критерий приёмки:**

- [ ] `git status` clean **или** только осознанно untracked docs
- [ ] `git log -1` на каждом репо показывает commit
- [ ] **Не** push на remote без явного запроса оркестратора/user

Если `git` credentials/user.name не настроены — настроить local `user.email`/`user.name` только для repo или пропустить T30 с записью в worker_report.

---

### T31: Документация Phase 5 (SHOULD)

**CPE README** — секция «Phase 5»:

- train/serve feature definition (past_trips as-of)
- causal response fields в DPE
- `make smoke-local` / `--start-servers`
- ports, env table

**DPE** — короткий `CAUSAL.md` **или** абзац в README:

- env: `CAUSAL_*`
- DriverHistoryStore bootstrap
- Sleeping Dog behavior

---

### T32: Guardrails & small hardening (COULD)

1. CPE: reject NaN features → 422; DPE fail-open if 422.
2. CPE `/predict_uplift`: log latency ms (print ok).
3. `scripts/evaluate_experiment.py`: optional `--output reports/eval.md` write file.
4. Config: document `CPE_RETRAIN_ON_START` default 0 in solo compose, 1 in full.

---

## 4. Порядок выполнения

```
T26 (parity) ──► T27 (cell bootstrap)
       │
       ├──► T28 (response fields) ──► T29 (smoke-local) ──► T31 (docs)
       │
       └──► T32 (optional)
T30 (git commits) — в конце, после зелёных тестов
```

**Рекомендуемый порядок:** T26 → T27 → T28 → T29 → T31 → T32 → T30.

---

## 5. Чего НЕ делать

- Не менять порты 8000/8100.
- Не блокировать DPE на CPE errors.
- Не поднимать timeout > 200ms без justification в worker_report.
- Не переписывать meta-learners «для красоты».
- Не добавлять Kafka/Streamlit/отдельный feature-store service.
- Не `git push` / force-push.
- Не затирать `audit_report_*.md` / `handoff_*.md`.
- Не коммитить секреты, venv, крупные бинарники (model.joblib — по `.gitignore`).

---

## 6. Критерии закрытия Phase 5

| # | Критерий |
|:---|:---|
| 1 | CPE pytest ≥ 29 green |
| 2 | DPE pytest ≥ 15 green (+ parity tests) |
| 3 | T26: connector ↔ history parity test PASS |
| 4 | T28: search JSON содержит causal_* поля |
| 5 | T29: one-shot smoke path documented + works (or skip with reason) |
| 6 | `worker_report_phase5.md` сдан |
| 7 | T30: commits (или явное «skipped» с reason) |

Оркестратор: re-audit → `audit_report_phase5.md`.

---

## 7. Локальная проверка

```bash
# CPE
cd Z:\pet-project\causal-pricing-engine
python -m pytest tests/ -v

# DPE
cd Z:\pet-project\dynamic-pricing-engine
python -m pytest tests/ -v

# Smoke (после T29)
cd Z:\pet-project\causal-pricing-engine
python scripts/e2e_smoke.py --start-servers --with-dpe
# или: make smoke-local
```

---

## 8. Формат сдачи

Создать **`worker_report_phase5.md`**:

```markdown
# Worker Report Phase 5
## Сделано (T26–T32)
## Тесты (pytest CPE + DPE raw output)
## Smoke
## Git (commit hashes or skipped)
## Отклонения
```

Без перезаписи handoff/audit оркестратора.
