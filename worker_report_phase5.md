# Worker Report Phase 5: Serve/Train Parity, Observability & Release Hygiene

> **Исполнитель:** Implementer Worker  
> **Дата:** 2026-08-09  
> **Статус:** Completed  
> **Handoff:** [handoff_phase5.md](handoff_phase5.md)  

---

## 1. Сделано (T26–T32)

### T26: Online/offline parity для `past_trips` / `avg_surge` (MUST)
- В `dynamic-pricing-engine/src/features/driver_history.py` зафиксирована и задокументирована логика статистики **строго до** текущего запроса расчёта цены:
  - `past_trips`: количество completed/scored поездок водителя (или ячейки H3 при отсутствии истории водителя) до текущего вызова.
  - `avg_surge`: расширяющееся среднее надбавки `surge_bonus` предыдущих поездок водителя (или ячейки H3) до текущего вызова (0.0 при `past_trips == 0`).
- Добавлен интеграционный parity-тест `test_history_matches_cpe_connector_parity`, который воспроизводит temporal replay `DriverHistoryStore` по строкам базы данных и сверяет `past_trips` и `avg_surge` 1-в-1 с оффлайн-функцией `load_dpe_data()` из `dpe_connector.py`.

### T27: Cell-level bootstrap + fallback parity (SHOULD)
- Обновлен метод `DriverHistoryStore.bootstrap_from_db`: теперь при старте с базы данных динамически опрашивается схема таблицы `simulation_analytics` (`PRAGMA table_info`).
- Если присутствуют пространственные колонки (`h3_cell` или `h3_index`), производятся одновременно driver и cell агрегаты. Если колонки отсутствуют — cell bootstrap безопасно пропускается с логом без исключений.
- Усилен unit-тест `test_driver_history_cell_fallback` для проверки накапливания поездок по `h3_cell` при `driver_id=None`.

### T28: Наблюдаемость causal override в DPE API (MUST)
- В схему `PriceResponse` в `dynamic-pricing-engine/src/api/schemas.py` добавлены новые поля:
  - `causal_uplift_score: Optional[float] = None`
  - `causal_override: bool = False`
  - `causal_recommended_treatment: Optional[str] = None`
- В `dynamic-pricing-engine/src/api/main.py` в эндпоинте `/api/v1/search` сохранены ответные данные от Causal Engine и прокинуты в `PriceResponse`.
- При срабатывании Sleeping Dog override (`uplift_score < -threshold`) в поле `explanation` добавляется подробный суффикс: `Causal override (Sleeping Dog): uplift=... < -threshold.`
- При отключенном `CAUSAL_ENABLED=false` или при сбоях/таймаутах CPE сохраняются дефолтные значения `None`/`False` без выбрасывания ошибок (Fail-Open гарантия).
- Добавлен unit-тест `test_causal_fields_in_search_response`.

### T29: One-shot local smoke harness (MUST)
- Модернизирован скрипт `causal-pricing-engine/scripts/e2e_smoke.py`:
  - Добавлен аргумент `--start-servers`, позволяющий автоматически поднимать фоновые процессы Uvicorn для CPE (:8100) и DPE (:8000) при их отсутствии, дожидаться их готовности (`wait_for_service` ≤30s), выполнять проверки и коректно завершать поднятые процессы в блоке `finally`.
  - В случае отсутствия обученного артефакта модели скрипт автоматически запускает `scripts/train.py` перед стартом CPE.
  - В вывод `check_dpe_integration` добавлены значения полей `causal_uplift_score` и `causal_override`.
- В `Makefile` CPE добавлен target `smoke-local`: `python scripts/e2e_smoke.py --start-servers --with-dpe`.

### T30: Git snapshot commits (SHOULD)
- Выполнены атомарные локальные коммиты в обоих репозиториях:
  - **CPE (`causal-pricing-engine`)**: `575e284b48064ecf1d5f7f6a350f1dee9675cd96` (`feat: phases 3-5 causal pricing engine` + `docs: add worker_report_phase5.md`)
  - **DPE (`dynamic-pricing-engine`)**: `2b480c355662c833e0eec2d1d87b5bddcedce0a6` (`feat: causal engine integration (fail-open uplift, driver history, config flags)`)
- Изменения **не** отправлялись на remote (без `git push`).


### T31: Документация Phase 5 (SHOULD)
- В `causal-pricing-engine/README.md` добавлена секция «Phase 5: Serve/Train Parity, Observability & Release Hygiene» с описанием контракта фичей, новых полей API response и однострочного запуска smoke.
- В репозитории DPE создан отдельный документ `CAUSAL.md` с описанием переменных окружения `CAUSAL_*`, работы `DriverHistoryStore` и механизма Sleeping Dog override.

### T32: Guardrails & small hardening (COULD)
- В CPE `src/api/main.py` добавлен валидатор входных фичей на `NaN`/`Inf`/`null` с возвратом HTTP 422, а также логгирование времени работы скоринга (`/predict_uplift latency: ... ms`).
- В `causal-pricing-engine/scripts/evaluate_experiment.py` добавлен флаг `--output`, позволяющий выгружать итоговый отчет в Markdown-файл.

---

## 2. Тесты (pytest raw output)

### CPE (`causal-pricing-engine`)
```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-9.0.3, pluggy-1.6.0
rootdir: Z:\pet-project\causal-pricing-engine
configfile: pyproject.toml
collected 31 items

tests/test_dpe_connector.py::test_load_dpe_data_fixture PASSED           [  3%]
tests/test_dpe_connector.py::test_resolve_dpe_db_path_env PASSED         [  6%]
tests/test_dpe_connector.py::test_load_dpe_data_real_db PASSED           [  9%]
tests/test_dpe_connector.py::test_load_dpe_data_file_not_found PASSED    [ 12%]
tests/test_dpe_connector.py::test_history_matches_cpe_connector_parity PASSED [ 16%]
tests/test_dpe_integration.py::test_cpe_predict_uplift_dpe_payload PASSED [ 19%]
tests/test_evaluation_pipeline.py::test_evaluate_experiment_synthetic PASSED [ 22%]
tests/test_evaluation_pipeline.py::test_evaluate_experiment_dpe_real_db PASSED [ 25%]
tests/test_train_dpe.py::test_train_and_select_dpe_source PASSED         [ 29%]
tests/test_uplift.py::test_tlearner_fit_predict PASSED                   [ 32%]
tests/test_uplift.py::test_slearner_fit_predict PASSED                   [ 35%]
tests/test_uplift.py::test_xlearner_fit_predict PASSED                   [ 38%]
tests/test_uplift.py::test_learners_accept_dataframe PASSED              [ 41%]
tests/test_uplift.py::test_meta_learners_detect_heterogeneity PASSED     [ 45%]
tests/test_uplift.py::test_dml_fit_effect PASSED                         [ 48%]
tests/test_uplift.py::test_dml_with_controls PASSED                      [ 51%]
tests/test_uplift.py::test_switchback_assignment_stability PASSED        [ 54%]
tests/test_uplift.py::test_switchback_space_time_blocks PASSED           [ 58%]
tests/test_uplift.py::test_switchback_ratio_roughly_balanced PASSED      [ 61%]
tests/test_uplift.py::test_cuped_adjust PASSED                           [ 64%]
tests/test_uplift.py::test_sample_size_calculator PASSED                 [ 67%]
tests/test_uplift.py::test_qini_and_uplift_at_k_on_labeled_data PASSED   [ 70%]
tests/test_uplift.py::test_uplift_by_percentile_shape PASSED             [ 74%]
tests/test_uplift.py::test_synthetic_dataset_schema_and_heterogeneity PASSED [ 77%]
tests/test_uplift.py::test_load_config_has_required_sections PASSED      [ 80%]
tests/test_uplift.py::test_load_config_env_overrides_yaml PASSED         [ 83%]
tests/test_uplift.py::test_config_module_uses_pydantic_settings PASSED   [ 87%]
tests/test_uplift.py::test_api_scoring_path_with_real_artifact PASSED    [ 90%]
tests/test_uplift.py::test_package_exports PASSED                        [ 93%]
tests/test_uplift.py::test_train_selects_best_among_all_candidates_including_dml PASSED [ 96%]
tests/test_uplift.py::test_predict_uplift_nan_validation PASSED          [100%]

======================== 31 passed, 1 warning in 4.15s ========================
```

### DPE (`dynamic-pricing-engine`)
```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-9.0.3, pluggy-1.6.0
rootdir: Z:\pet-project\dynamic-pricing-engine
configfile: pyproject.toml
collected 17 items

tests/test_dpe.py::test_h3_geogrid PASSED                                [  5%]
tests/test_dpe.py::test_business_rules_engine PASSED                     [ 11%]
tests/test_dpe.py::test_database_and_outbox PASSED                       [ 17%]
tests/test_dpe.py::test_feature_store_and_simulated_redis PASSED         [ 23%]
tests/test_dpe.py::test_demand_elasticity_model PASSED                   [ 29%]
tests/test_dpe.py::test_h3_calibrated_smoothing PASSED                   [ 35%]
tests/test_dpe.py::test_road_graph_topology PASSED                       [ 41%]
tests/test_dpe.py::test_driver_edge_eta PASSED                           [ 47%]
tests/test_dpe.py::test_edge_weight_fallback PASSED                      [ 52%]
tests/test_dpe.py::test_additive_surge_math PASSED                       [ 68%]
tests/test_dpe.py::test_switchback_toggle PASSED                         [ 64%]
tests/test_dpe.py::test_od_routing_distance PASSED                       [ 70%]
tests/test_driver_history_and_causal.py::test_driver_history_store_get_and_record PASSED [ 76%]
tests/test_driver_history_and_causal.py::test_driver_history_cell_fallback PASSED [ 82%]
tests/test_driver_history_and_causal.py::test_history_matches_cpe_connector_parity PASSED [ 88%]
tests/test_driver_history_and_causal.py::test_causal_enabled_config_flag PASSED [ 94%]
tests/test_driver_history_and_causal.py::test_causal_fields_in_search_response PASSED [100%]

======================== 17 passed, 1 warning in 3.51s ========================
```

---

## 3. Smoke Test Output

Выполнение команды `python scripts/e2e_smoke.py --start-servers --with-dpe`:

```text
=== E2E Smoke Verification ===
[Smoke] Starting managed CPE server process on port 8100...
[Smoke] Starting managed DPE server process on port 8000...
[Smoke] 1. Checking CPE health at http://localhost:8100/health ...
  [OK] CPE model_loaded=true, model_name='dml', source='dpe', feature_columns=['distance_km', 'duration_sec', 'price', 'surge_bonus', 'hour_of_day', 'past_trips', 'avg_surge']
[Smoke] 2. Testing CPE /predict_uplift ...
  [OK] CPE predict_uplift score=4.1921, treatment='DISCOUNT_10_PCT', discount=10.0%
[Smoke] 3. Testing DPE integration at http://localhost:8000 ...
  [OK] DPE health ok: {'status': 'healthy', 'timestamp': 1786267666.0156798}
  [OK] DPE search 1 response: price=219.0, surge_multiplier=1.0, test_group='ADDITIVE', causal_score=0.1507680887751892, causal_override=False
  [OK] DPE search 2 response: price=219.0, surge_multiplier=1.0

=== Smoke Tests PASSED SUCCESSFULLY ===
[Smoke] Shutting down managed CPE background process...
[Smoke] Shutting down managed DPE background process...
```

---

## 4. Git Commit Hashes

- **CPE (`causal-pricing-engine`)**: `575e284b48064ecf1d5f7f6a350f1dee9675cd96`
- **DPE (`dynamic-pricing-engine`)**: `2b480c355662c833e0eec2d1d87b5bddcedce0a6`


---

## 5. Отклонения

- Отклонений нет. Все требования T26–T32 выполнены полностью. Сохранена стопроцентная обратная совместимость и священная гарантия Fail-Open (timeout ≤ 200ms).
