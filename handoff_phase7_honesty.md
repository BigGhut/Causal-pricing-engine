# Handoff Phase 7: Honesty Hardening (оставшиеся недочёты)

> **Автор:** оркестратор  
> **Дата:** 2026-08-09  
> **Цель:** закрыть **оставшийся** technical debt витрины и causal-честности — **не** новые модели, не monorepo, не Kafka.  
> **Контекст:** Phase 6 portfolio polish + honest demo + normalized Qini — DONE.  
> **Сдача:** только `worker_report_phase7.md` (не затирать `handoff_*.md` / `audit_report_*.md` / `CASE_STUDY.md` целиком — можно точечно дополнять).

---

## 0. Что уже НЕ трогать (закрыто)

| Тема | Статус |
|:---|:---|
| Честный demo (argmax/argmin holdout) | DONE |
| Qini normalized coefficient (~[-1,1]) | DONE |
| Unified DPE feature schema | DONE |
| Slim default deps | DONE |
| docs/archive process notes | DONE |
| Fail-open DPE↔CPE, causal_* fields, DriverHistoryStore parity | DONE — **не ломать** |

**Священное:** порты 8100/8000, fail-open ≤200ms, `python scripts/demo.py` и `portfolio_proof.py` остаются зелёными.

**Регрессия:**

```bash
cd Z:\pet-project\causal-pricing-engine
python -m pytest tests/ -q
python scripts/demo.py
python scripts/portfolio_proof.py

cd Z:\pet-project\dynamic-pricing-engine
python -m pytest tests/ -q
```

---

## 1. Проблематика (зачем Phase 7)

Оркестраторский разбор «модель врёт» выявил:

| # | Находка | Риск |
|:---|:---|:---|
| H1 | Unit-тесты часто `fit(X); predict(X)` на **тех же** данных | Оптимизм; слабый сигнал о generalization |
| H2 | Synthetic DGP: `true_τ = f(X)` by design | Норм для demo, но **не** заявлено везде явно |
| H3 | DPE logs: `surge_bonus ≡ 0` на control (MULTIPLICATIVE); treatment-entangled features | CATE на DPE ≠ чистый RCT |
| H4 | `docs/evidence/latest_proof.md` устаревает после смены Qini | Витрина врёт цифрами |
| H5 | Git: worker reports / evidence могут быть untracked | Грязный snapshot |
| H6 | Null bar Qini (random) есть в demo, не везде в train/proof output consistency | Мелочь |

Phase 7 закрывает **H1–H5** поэтапно. **Не** цель: multi-treatment, CUPAC, monorepo.

---

## 2. Этапы и задачи

### Этап A — Витрина и артефакты (MUST, сначала)

#### A1. Перегенерировать evidence после Qini-fix

```bash
cd Z:\pet-project\causal-pricing-engine
python scripts/portfolio_proof.py
```

- [ ] `docs/evidence/latest_proof.md` содержит **normalized** Qini (порядок |coef| ≲ 1.5, не тысячи).
- [ ] Sleeping Dog HTTP → `NO_DISCOUNT_AVOID`, policy `causal_override=true`.

#### A2. CASE_STUDY / README — явный disclaimer synthetic

В `CASE_STUDY.md` (и 2–3 строки в README, если уместно):

- Synthetic: **seeded HTE**, `τ` — кусочная функция от `past_trips`/`surge_bonus`; demo доказывает **pipeline + policy**, не city-scale lift.
- Qini = **normalized coefficient**, random null ≈ 0.
- DPE train: observational/switchback design; features may be treatment-linked (см. этап C).

**Критерий:** читатель за 30 сек понимает границы claim.

#### A3. `portfolio_proof` / `demo` — единый язык метрик

- Везде: `Qini coefficient (normalized)` + optional `random null`.
- В proof evidence setup: одна строка random null Qini.

---

### Этап B — Тесты без train=test optimism (MUST)

#### B1. Метрики / HTE detection — только holdout

Файлы: `tests/test_uplift.py` (и любые тесты, где `fit(X); predict_uplift(X)` **и** assert на качество).

| Тест | Требование |
|:---|:---|
| `test_meta_learners_detect_heterogeneity` | split train/test; assert на **test** |
| `test_qini_*` | уже holdout — не ухудшить |
| fit/predict shape tests | same-X **допустимо** (только shape/finite) |

#### B2. Добавить `test_qini_beats_random_on_holdout`

- synthetic n≥1500, holdout 30%  
- T-Learner train → Qini_model > Qini_random + margin (например 0.02) **или** model > 0 и |random| < 0.15  
- Не требовать Qini > 0.5 (после normalization realistic ~0.05–0.3)

#### B3. Не раздувать suite

Цель: confidence, не +50 тестов. CPE pytest остаётся green.

---

### Этап C — DPE feature honesty (SHOULD → почти MUST)

**Проблема:** `treatment=MULTIPLICATIVE` ⇒ `surge_bonus=0` всегда; `price`/`surge_bonus` частично **post-treatment / arm-linked**.

#### C1. Документировать в `dynamic-pricing-engine/CAUSAL.md`

Короткий раздел **«Training features caveats»**:

- `surge_bonus` на control часто 0 by construction  
- для строгого CATE предпочтительны pre-treatment features  
- online DPE шлёт post-compute `price`/`surge_bonus` — train/serve alignment vs causal purity trade-off  

#### C2. Pre-treatment feature set для offline train (CPE)

В `src/data/dpe_connector.py` (или рядом):

```python
# Example — exact names must match available columns / derived fields
DPE_PRE_TREATMENT_FEATURES = [
    "distance_km",
    "duration_sec",
    "hour_of_day",
    "past_trips",
    "avg_surge",
]
# Explicitly EXCLUDE by default for "causal" train mode: surge_bonus, price
# OR keep full set as DPE_FEATURE_COLUMNS for serve-parity mode
```

**Интерфейс:**

```python
def load_dpe_data(..., feature_mode: Literal["serve_parity", "pre_treatment"] = "serve_parity")
```

| Mode | Features | Use |
|:---|:---|:---|
| `serve_parity` (default) | full `DPE_FEATURE_COLUMNS` | текущий prod path, не ломать API |
| `pre_treatment` | без `price`, `surge_bonus` (и др. post-treatment) | честный offline uplift train / evaluate |

#### C3. Train CLI

```bash
python scripts/train.py --source dpe --feature-mode pre_treatment
python scripts/train.py --source dpe --feature-mode serve_parity   # default
```

- Artifact payload must store `feature_columns` actually used.  
- API already uses artifact columns — **pre_treatment model will not match DPE online payload** unless DPE also sends only those features.

**Важно (продуктовое правило Phase 7):**

- Default **serve path** остаётся `serve_parity` (7 features), чтобы DPE integration не сломать.  
- `pre_treatment` — для offline analysis / optional second artifact path `artifacts/model_pretreat.joblib` **или** document that pre_treatment is **eval-only** (no serve).

**Рекомендация оркестратора (выбрать одно и задокументировать):**

**Вариант R1 (проще, предпочтителен):**  
`pre_treatment` только для `train` metrics + evaluate script; serve всегда full schema. В worker_report явно: «pre_treatment = research mode».

**Вариант R2:**  
DPE online тоже шлёт только pre-treatment features (price/surge исключить из CPE call). Тогда один schema. Больше правок DPE.

→ **Делать R1**, если нет времени на R2. R2 — COULD.

#### C4. Тест

- `load_dpe_data(feature_mode="pre_treatment")` не содержит `surge_bonus`/`price` в feature matrix used by train helper.  
- `serve_parity` — полный набор (регрессия).

---

### Этап D — Propensity / diagnostics (SHOULD)

#### D1. Скрипт `scripts/diagnostics_leakage.py` (короткий)

Печатает:

- synthetic: AUC(T|X), corr(û, true_τ) if segment present, Qini model/random/oracle  
- dpe: AUC(T|X), mean surge by T, Qini holdout  

Exit 0 always (diagnostic). Document in README one line under archive or case study.

#### D2. Не вшивать diagnostics в pytest как flaky assert на DPE AUC

---

### Этап E — Git hygiene (SHOULD)

#### E1. Local commits only, **no push**

Предлагаемые commits (1–2):

```text
fix(eval): normalized Qini coefficient + honest demo/proof labels
chore: phase7 honesty — holdout tests, pre_treatment train mode, evidence refresh
```

Включить: metrics, tests, demo/proof, CASE_STUDY, evidence, connector/train flags, CAUSAL.md, diagnostics.

#### E2. Не коммитить

- `__pycache__`, `.pytest_cache`, venv  
- `.worker_prompt_*.txt` (gitignore)  
- крупные secrets  

`artifacts/model.joblib` — по существующему `.gitignore`.

---

### Этап F — Optional R2 (COULD)

Если останется время после A–E:

- DPE `predict_uplift` payload: убрать или занулить post-treatment fields;  
- CPE model train default на pre_treatment + serve aligned;  
- e2e proof with DPE still green.

Иначе skip + note in worker_report.

---

## 3. Порядок выполнения

```
A1 evidence refresh
A2 disclaimers CASE_STUDY/README
A3 metric wording consistency
B1–B2 holdout tests
C1 CAUSAL.md caveats
C2–C4 pre_treatment mode (R1)
D1 diagnostics script
E1 local git commit
F optional
→ worker_report_phase7.md
```

---

## 4. Критерии закрытия Phase 7

| # | Критерий |
|:---|:---|
| 1 | `docs/evidence/latest_proof.md` с normalized Qini (\|q\| ≲ 1.5) |
| 2 | CASE_STUDY (и README) явно ограничивают synthetic/DPE claims |
| 3 | HTE/Qini quality asserts на **holdout**, не same-X |
| 4 | `feature_mode=pre_treatment` работает offline; default serve_parity не сломан |
| 5 | CAUSAL.md: training caveats |
| 6 | CPE + DPE pytest green; demo + portfolio_proof exit 0 |
| 7 | `worker_report_phase7.md` с raw pytest + отклонениями |
| 8 | Local commits, no push |

---

## 5. Чего НЕ делать

- Не «улучшать» Qini, подгоняя модель под coef → 0.9 (toy DGP).  
- Не удалять S/X/DML файлы ради чистоты (можно не трогать).  
- Не monorepo / не Streamlit.  
- Не ломать DPE fail-open.  
- Не менять порты.  
- Не push remote.  
- Не перезаписывать audit_report оркестратора.

---

## 6. Формат сдачи

**`worker_report_phase7.md`:**

```markdown
# Worker Report Phase 7
## Этапы A–E (чеклист)
## pytest CPE / DPE
## demo + proof (краткий stdout)
## pre_treatment: как запускать
## Git hashes
## Не сделано (F/R2?)
```

---

## 7. Definition of Done (одной фразой)

> Витрина и тесты **не врут масштабом и same-X оптимизмом**; DPE-train **документирован** и имеет **pre_treatment research mode**; evidence обновлён; integration serve_parity жив.

Оркестратор после сдачи: re-audit → `docs/archive/audit_report_phase7.md` или root audit — по согласованию.
