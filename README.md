# Causal Pricing Engine (CPE)

Sidecar для [Dynamic Pricing Engine](https://github.com/BigGhut/Dynamic-pricing-engine). Оценивает **один** эффект: как **аддитивная надбавка** меняет вероятность того, что **водитель примет** заказ, относительно **базового тарифа**.

DPE спрашивает CPE только в аддитивный виртуальный час и только если в поиске есть `driver_id`. Вызов fail-open, таймаут 200 мс. `ranking_supports_decision` записывается один раз по калибровке. Он `true`, только если нижняя граница Qini на калибровке больше нуля и `false_override_rate` не больше `0.10`. Ложный override — это `score < −0.05` при `true_uplift ≥ 0`; доля считается среди строк, где override сработал. Порог `0.10` зафиксирован в `PLAN.md` до прогона. Сервер флаг читает и заново не считает. Qini в доказательстве считается на нетронутом тесте.

Это не скидка и не switchback «аддитивная формула против мультипликативной». Та рука назначается чётностью виртуального часа, так что водителю воздействие не рандомизировано. Модель на этих логах не учится.

Обучение — синтетические заказы, где надбавка случайна на уровне заказа. Сегменты в генераторе — пороги по `past_trips` и `distance_km`, и эти колонки модель видит. Попасть в знак эффекта здесь не открытие. Цифры доказательства, с отдельными интервалами на калибровке и на нетронутом тесте: [docs/evidence/latest_proof.md](docs/evidence/latest_proof.md). Тест `test_qini_beats_random_when_the_planted_effect_is_large` сажает эффект ±0.45 и требует нижнюю границу Qini выше 0.20 и на 0.15 выше случайных скоров. Это проверка метрики на сильном эффекте, не портфельный результат.

Python ≥ 3.10 · scikit-learn · FastAPI  
CPE `:8100` · DPE `:8000`

Признаки: `distance_km`, `duration_sec`, `hour_of_day`, `past_trips`, `avg_surge`. `price` и `surge_bonus` в модель не входят.

```text
DPE :8000  --POST /predict_uplift (аддитивный час, есть driver_id)-->  CPE :8100
                τ̂ < -0.05 и Qini целиком выше 0  →  базовый тариф, CAUSAL_NO_SURGE
```

## Запуск

```powershell
pip install -r requirements.txt
pip install -e .
python scripts/train.py --source synthetic
python scripts/demo.py
python scripts/portfolio_proof.py
```

`python scripts/train.py --source dpe` модель не пишет: печатает, почему лог switchback не является этим экспериментом. Путь к базе — соседний checkout `dynamic-pricing-engine/dpe_database.db` или `CPE_DPE__DB_PATH`.

## API

`GET /health` · `POST /predict_uplift`

```json
{
  "driver_id": "driver_008",
  "features": {
    "distance_km": 7.0,
    "duration_sec": 900.0,
    "hour_of_day": 11.0,
    "past_trips": 4.0,
    "avg_surge": 12.0
  }
}
```

Ответ: `uplift_score`, `recommended_treatment` (`SURCHARGE` / `KEEP_QUOTE` / `NO_SURCHARGE`), `model_name`.

Интеграция на стороне DPE: [CAUSAL.md](https://github.com/BigGhut/Dynamic-pricing-engine/blob/main/CAUSAL.md). Разбор ограничений: [CASE_STUDY.md](CASE_STUDY.md).
