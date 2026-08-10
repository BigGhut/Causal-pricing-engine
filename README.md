# Causal Pricing Engine (CPE)

**Uplift-sidecar** для [Dynamic Pricing Engine (DPE)](https://github.com/BigGhut/Dynamic-pricing-engine): считает индивидуальный эффект воздействия (ITE) и подсказывает, **кому не давать агрессивный surge** (Sleeping Dogs).

Python ≥ 3.10 · scikit-learn · FastAPI · Docker  
Порт CPE `:8100` · DPE `:8000` · вызов `POST /predict_uplift` · timeout ≤ 200 ms · **fail-open**

| | DPE | CPE |
|:---|:---|:---|
| Роль | цена, граф, surge | ITE / policy hint |
| Док интеграции | [CAUSAL.md](https://github.com/BigGhut/Dynamic-pricing-engine/blob/main/CAUSAL.md) | этот репо |

**Честно:** дефолтное демо — синтетика с заложенным HTE, не city-scale prod lift. Qini — нормализованный. Подробности и trade-off → [CASE_STUDY.md](CASE_STUDY.md).

```text
DPE :8000  --POST /predict_uplift (≤200ms, fail-open)-->  CPE :8100
                τ̂ < -0.05  →  base fare, CAUSAL_NO_SURGE
```

## Запуск

```powershell
pip install -r requirements.txt && pip install -e .
python scripts/demo.py              # честные роли с holdout
python scripts/portfolio_proof.py   # HTTP + docs/evidence/
# make demo | make proof | make train | make test | make serve
```

Опционально: `pip install -e ".[causal]"` (EconML и др.) · dual-stack с DPE: CPE `:8100` + DPE `:8000`, затем `python scripts/portfolio_proof.py --with-dpe`.

## API

`GET /health` · `POST /predict_uplift`  
Тело: `{ "user_id", "features": { distance_km, duration_sec, price, surge_bonus, hour_of_day, past_trips, avg_surge } }`  
Ответ: `uplift_score`, `recommended_treatment` (`DISCOUNT_10_PCT` / `NO_DISCOUNT` / `NO_DISCOUNT_AVOID`), `model_name`.  
Перед serve: `python scripts/train.py`.

## Evidence (пример)

Captured `scripts/portfolio_proof.py` (2026-08-09): Qini **+0.084**, Uplift@30% **+0.23**, Sleeping Dog \(\hat\tau\approx-0.42\) → override.  
Полный JSON → [docs/evidence/latest_proof.md](docs/evidence/latest_proof.md).

## Структура

`src/api` · `src/causal` · `src/data` · `src/evaluation` · `scripts/` (demo, proof, train) · `docs/evidence` · `docs/archive` · `tests/`

## Дальше

[CASE_STUDY.md](CASE_STUDY.md) · [evidence](docs/evidence/latest_proof.md) · [DPE](https://github.com/BigGhut/Dynamic-pricing-engine) · [CAUSAL.md](https://github.com/BigGhut/Dynamic-pricing-engine/blob/main/CAUSAL.md) · [archive](docs/archive/README.md)
