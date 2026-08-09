# Handoff Phase 6: Portfolio Polish (L1 + selective L2)

> **Автор:** оркестратор  
> **Дата:** 2026-08-09  
> **Цель проекта:** **портфолио**, не production platform  
> **Предыдущие фазы:** 1–5 CLOSED (PASS)  
> **Стратегия:** [практическая рекомендация] — **не** вырезать causal story / HTTP CPE↔DPE / fail-open / driver history.  
> **Срезать:** шум docs, тяжёлые default-deps, dual feature contracts, «лишние» default-модели, загромождение first-run.

---

## 1. Portfolio narrative (не менять смысл)

Проект должен объясняться за 60 секунд:

> Сервис **uplift (ITE)** для динамического ценообразования: учимся на switchback-симуляции DPE, скорим персональный эффект treatment, в runtime DPE **не даёт surge Sleeping Dogs** (fail-open, train/serve parity).

**Не трогать (святое):**

| Компонент | Почему для портфолио |
|:---|:---|
| FastAPI CPE `:8100` + DPE HTTP fail-open | engineering signal (integration) |
| Sleeping Dog override + `causal_*` fields | понятный product rule |
| `DriverHistoryStore` as-of parity | train/serve maturity |
| T-Learner (sklearn) + train/predict | core ML story |
| `load_dpe_data` + train on DPE | связь двух pet-проектов |
| Qini (или одна offline metric) при train | «как выбирали/мерили» |

**Не делать Phase 6:**

- Не monorepo / in-process only (L3)
- Не удалять DPE integration
- Не переписывать S/T/X/DML «с нуля» — **default path** упростить
- Не `git push`
- Не затирать `handoff_*.md` / `audit_report_*.md` — только **перенос в archive**
- Сдача: **`worker_report_phase6.md`** only

---

## 2. Задачи

### P1: README portfolio rewrite (MUST — highest ROI)

**Файл:** `README.md` (CPE) — **полная перепись** под портфолио.

Структура (строго):

1. **Title + 2–3 предложения** (проблема / uplift / DPE)  
2. **Why not classic ML** (формула ITE в 2–3 строки)  
3. **Architecture** — mermaid или ascii: DPE `:8000` ↔ CPE `:8100`  
4. **Key ideas** — Persuadables / Sleeping Dogs; fail-open; train/serve parity  
5. **Quick start** (Windows-friendly, ≤8 команд):
   ```text
   pip install -r requirements.txt
   pip install -e .
   python scripts/demo.py          # или documented fallback
   # optional dual:
   # CPE uvicorn :8100 / DPE uvicorn :8000
   ```
6. **Results** — пример метрик train (Qini) + пример JSON predict / causal_override  
7. **Project layout** — только важные пути (не список всех handoff)  
8. **What I'd improve next** — 3 bullet  
9. **Archive** — ссылка `docs/archive/` на process notes  

Тон: technical blog, без эмодзи-спама (1–2 ок). Убрать roadmap «T1–T32».

**DPE:** короткий абзац или ссылка на существующий `CAUSAL.md` (не раздувать).

---

### P2: Archive process docs (MUST)

1. Создать `docs/archive/` в CPE.  
2. Перенести (git mv предпочтительно) **не удаляя историю смысла**:
   - `handoff.md`, `handoff_phase3.md`, `handoff_phase3_fix.md`, `handoff_phase4.md`, `handoff_phase5.md`, `handoff_phase6_portfolio.md` (копию handoff6 можно оставить в root **или** тоже archive после — **root оставить** только `handoff_phase6_portfolio.md` до сдачи; остальные handoff → archive)
   - `audit_report.md`, `audit_report_phase3.md`, `audit_report_phase4.md`, `audit_report_phase5.md`
   - `worker_report_phase4.md`, `worker_report_phase5.md`
3. `docs/archive/README.md` — 5 строк: «process artifacts from multi-agent build; not required to run».  
4. Удалить/gitignore: `.worker_prompt_phase*.txt`  

**Не переносить:** `README.md`, `CAUSAL.md` (DPE), код, configs.

---

### P3: Slim default dependencies (MUST)

**Файл:** `requirements.txt` (+ согласовать `pyproject.toml` optional)

**Default install (лёгкий, для demo):**

```text
numpy, pandas, scikit-learn, scipy
fastapi, uvicorn, pydantic, pydantic-settings
pyyaml, joblib
httpx   # smoke / optional client
pytest  # or in [dev]
```

**Убрать из default** (если есть):

- `econml` → optional `[causal]` or documented skip  
- `catboost`, `lightgbm` → optional  
- `causalml` → optional / remove if unused  

**Код:**

- `DMLEngine` / CatBoost path: **не ломать import**, если пакет нет — train **не** включает DML/catboost в default candidates.  
- `scripts/train.py` default candidates: **`t_learner` only** OR `t_learner` + `s_learner` + `x_learner` **без DML**.  
  - Рекомендация портфолио: **T-Learner only** в default; S/X оставить в коде, флаг `--learners t,s,x` optional.  
- Минимально: `python scripts/train.py` работает **без econml/catboost**.

**Критерий:** fresh venv + `pip install -r requirements.txt` + train synthetic **or** dpe — success.

---

### P4: Unified feature contract (MUST)

1. **Один** canonical list для portfolio path = **DPE_FEATURE_COLUMNS**  
   (`distance_km`, `duration_sec`, `price`, `surge_bonus`, `hour_of_day`, `past_trips`, `avg_surge`).  
2. Synthetic generator: **либо** генерит **те же** колонки + treatment/conversion/revenue, **либо** train default = `--source dpe` when DB exists else synthetic **same schema**.  
3. `FEATURE_COLUMNS` в `synthetic.py`:  
   - deprecate old 5-col schema **или** alias: `FEATURE_COLUMNS = DPE_FEATURE_COLUMNS` and rewrite generator.  
4. API default / tests: обновить под 7 DPE features.  
5. Старые тесты на 5-col synthetic — починить, не skip.

**Критерий:** один schema train↔API↔DPE payload; нет zero-fill `price_sensitivity`/`segment` на DPE path.

---

### P5: `scripts/demo.py` happy path (MUST)

Один скрипт для README:

1. Resolve DPE db (config/env/fallback); if missing → synthetic **same schema**.  
2. Train **T-Learner** (fast, small n if synthetic e.g. 2000).  
3. Save artifact (or use temp).  
4. Score 3 sample rows: low/mid/high uplift if possible.  
5. Print markdown-friendly summary:
   - model name, n samples, qini if cheap  
   - table: features snippet → uplift → recommended treatment  
6. Exit 0.  
7. **Не** требует уже запущенных uvicorn (in-process fit/predict).

Makefile: `demo: python scripts/demo.py`

---

### P6: Train CLI portfolio defaults (SHOULD)

- Default `--source`: `dpe` if DB exists else `synthetic`.  
- Default learners: T only (see P3).  
- Keep `--source dpe|synthetic`, `--dpe-db-path`.  
- Print short «portfolio blurb» at end of train (1–2 lines).

---

### P7: Test diet (SHOULD — не ломать coverage смысла)

**Не** цель «удалить все тесты». Цель: убрать шум, оставить confidence.

1. Сохранить: meta-learner basic (хотя бы T), dpe connector fixture, parity **or** reference, API score, nan 422, train path.  
2. Можно объединить дубли parity (CPE vs DPE) — **один** достаточный parity test.  
3. Не удалять DPE causal tests.  
4. **Критерий:**  
   - CPE: `pytest tests/ -q` **green** (число может ↓ или ≈)  
   - DPE: **green** (≥ current 17 or обоснованно чуть меньше)

---

### P8: Ops noise reduction (SHOULD)

1. README: primary run = local uvicorn / demo; Docker — **optional** subsection (1 Dockerfile ok).  
2. Не удалять Dockerfile/compose если работают — **не** делать full-compose обязательным в Quick start.  
3. `e2e_smoke.py` оставить; в README secondary: `make smoke-local`.

---

### P9: Git commit local (SHOULD)

Один (или два) commit **только local**, no push:

```text
docs+chore: portfolio polish (README, archive process docs, slim defaults, demo)
```

---

## 3. Порядок

```
P2 archive docs
P3 slim deps + train defaults
P4 unified features + fix tests
P5 demo.py
P1 README (после demo — точные команды)
P6 train defaults polish
P7 test diet (если нужно)
P8 ops README notes
P9 commit
worker_report_phase6.md
```

---

## 4. Критерии закрытия Phase 6

| # | Критерий |
|:---|:---|
| 1 | README читается как portfolio landing (структура P1) |
| 2 | Process docs в `docs/archive/` |
| 3 | `pip install -r requirements.txt` без econml/catboost **и** train/demo работают |
| 4 | Unified DPE feature schema train↔API |
| 5 | `python scripts/demo.py` exit 0, печатный summary |
| 6 | CPE + DPE pytest green |
| 7 | DPE↔CPE integration код **не** вырезан |
| 8 | `worker_report_phase6.md` сдан |

---

## 5. Чего НЕ делать

- Не удалять `src/causal/dml_engine.py` / S/X **файлы** обязательно — можно оставить unused; **default path** без них.  
- Не ломать порты 8100/8000.  
- Не добавлять новые heavy frameworks.  
- Не раздувать handoff дальше в README.  
- Не push remote.

---

## 6. Проверка воркера

```bash
cd Z:\pet-project\causal-pricing-engine
pip install -r requirements.txt
pip install -e .
python scripts/demo.py
python -m pytest tests/ -q
python scripts/train.py   # defaults work

cd Z:\pet-project\dynamic-pricing-engine
python -m pytest tests/ -q
```

---

## 7. Сдача

**`worker_report_phase6.md`:**

- Что сделано P1–P9  
- pytest raw summary  
- demo output (укороченный)  
- commit hash  
- отклонения  

Без overwrite audit/handoff в root (кроме переноса в archive по P2).
