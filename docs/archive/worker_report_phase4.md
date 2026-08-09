# Worker Report Phase 4: Production Hardening CPE ↔ DPE

**Дата:** 2026-08-09  
**Исполнитель:** Implementer Worker  
**Статус:** DONE / ALL MUST & SHOULD TASKS PASS  

---

## 1. Сделано (Tasks Summary)

| Задача | Статус | Компоненты / Файлы | Описание реализации |
|:---|:---|:---|:---|
| **T19: Конфигурируемый путь к DPE DB** | ✅ DONE | `CPE: src/config.py`, `src/data/dpe_connector.py`, `configs/config.yaml` | Добавлена секция `DpeConfig` в `AppConfig`. Реализован функция-резолвер `resolve_dpe_db_path()` с приоритетом `CLI/arg -> env CPE_DPE__DB_PATH -> YAML -> fallbacks (/app/data/dpe_database.db, relative path, legacy Z: path)`. |
| **T23: Тесты на train dpe + connector fixture** | ✅ DONE | `CPE: tests/test_dpe_connector.py`, `tests/test_train_dpe.py` | Создан генератор SQLite fixture (`create_dummy_dpe_db`), добавлены изолированные тесты загрузки данных и обучения `train_and_select(source="dpe")` без `pytest.skip`. Тест прошел с `DPE_FEATURE_COLUMNS`. |
| **T18: Driver feature store / history в DPE** | ✅ DONE | `DPE: src/features/driver_history.py`, `src/api/schemas.py`, `src/api/main.py` | Создан ин-мемори класс `DriverHistoryStore` с возможностью bootstrap из `simulation_analytics`. Добавлена передача `driver_id` в `SearchRequest` и расчет реальных `past_trips` и `avg_surge` при вызовах Causal Engine вместо захардкоженных `0.0`. При отсутствии `driver_id` работает фолбек по `h3_cell`. |
| **T22: Конфиг интеграции DPE** | ✅ DONE | `DPE: src/config.py`, `src/api/main.py` | В конфиг DPE добавлены параметры `CAUSAL_ENGINE_URL`, `CAUSAL_ENGINE_TIMEOUT_SEC`, `CAUSAL_UPLIFT_THRESHOLD`, `CAUSAL_ENABLED`. При `CAUSAL_ENABLED=false` HTTP-вызовы полностью отключаются. |
| **T20: Docker artifact = DPE schema** | ✅ DONE | `CPE: Dockerfile`, `scripts/docker_entrypoint.sh`, `docker-compose.full.yml` | Создан `docker_entrypoint.sh` для автоматической дообучаемости модели при старте контейнера при наличии монтированной DPE DB (`CPE_RETRAIN_ON_START=1`). Обновлен `docker-compose.full.yml` с нужными env-переменными. |
| **T21: E2E smoke script** | ✅ DONE | `CPE: scripts/e2e_smoke.py`, `Makefile` | Написан скрипт `scripts/e2e_smoke.py` с проверками `GET /health` (model_loaded, model_name, source, feature_columns), `POST /predict_uplift` и (опционально `--with-dpe`) `POST /api/v1/search`. В `Makefile` добавлены цели `smoke` и `smoke-dpe`. |
| **T24: Наблюдаемость и safety** | ✅ DONE | `CPE: src/api/main.py`, `DPE: src/api/main.py` | Расширен `/health` эндпоинт CPE: отдает `source` и `feature_columns`. В API логируется конфигурация загруженной модели при старте. |
| **T25: Документация** | ✅ DONE | `CPE: README.md` | В README добавлены разделы по интеграции Phase 3–4, портам 8000/8100, командам `make train-dpe`, `make evaluate`, `make smoke`, флагам окружения и описанию логики Sleeping Dog Override. |

---

## 2. Результаты запуска тестов (pytest output)

### CPE Suite (29/29 passed):
```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-9.0.3, pluggy-1.6.0
collected 29 items

tests/test_dpe_connector.py::test_load_dpe_data_fixture PASSED           [  3%]
tests/test_dpe_connector.py::test_resolve_dpe_db_path_env PASSED         [  6%]
tests/test_dpe_connector.py::test_load_dpe_data_real_db PASSED           [ 10%]
tests/test_dpe_connector.py::test_load_dpe_data_file_not_found PASSED    [ 13%]
tests/test_dpe_integration.py::test_cpe_predict_uplift_dpe_payload PASSED [ 17%]
tests/test_evaluation_pipeline.py::test_evaluate_experiment_synthetic PASSED [ 20%]
tests/test_evaluation_pipeline.py::test_evaluate_experiment_dpe_real_db PASSED [ 24%]
tests/test_train_dpe.py::test_train_and_select_dpe_source PASSED         [ 27%]
tests/test_uplift.py::test_tlearner_fit_predict PASSED                   [ 31%]
tests/test_uplift.py::test_slearner_fit_predict PASSED                   [ 34%]
tests/test_uplift.py::test_xlearner_fit_predict PASSED                   [ 37%]
tests/test_uplift.py::test_learners_accept_dataframe PASSED              [ 41%]
tests/test_uplift.py::test_meta_learners_detect_heterogeneity PASSED     [ 44%]
tests/test_uplift.py::test_dml_fit_effect PASSED                         [ 48%]
tests/test_uplift.py::test_dml_with_controls PASSED                      [ 51%]
tests/test_uplift.py::test_switchback_assignment_stability PASSED        [ 55%]
tests/test_uplift.py::test_switchback_space_time_blocks PASSED           [ 58%]
tests/test_uplift.py::test_switchback_ratio_roughly_balanced PASSED      [ 62%]
tests/test_uplift.py::test_cuped_adjust PASSED                           [ 65%]
tests/test_uplift.py::test_sample_size_calculator PASSED                 [ 68%]
tests/test_uplift.py::test_qini_and_uplift_at_k_on_labeled_data PASSED   [ 72%]
tests/test_uplift.py::test_uplift_by_percentile_shape PASSED             [ 75%]
tests/test_uplift.py::test_synthetic_dataset_schema_and_heterogeneity PASSED [ 79%]
tests/test_uplift.py::test_load_config_has_required_sections PASSED      [ 82%]
tests/test_load_config_env_overrides_yaml PASSED                         [ 86%]
tests/test_config_module_uses_pydantic_settings PASSED                   [ 89%]
tests/test_api_scoring_path_with_real_artifact PASSED                    [ 93%]
tests/test_package_exports PASSED                                        [ 96%]
tests/test_train_selects_best_among_all_candidates_including_dml PASSED [100%]

======================== 29 passed, 1 warning in 4.09s ========================
```

### DPE Suite (15/15 passed):
```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-9.0.3, pluggy-1.6.0
collected 15 items

tests/test_dpe.py::test_h3_geogrid PASSED                                [  6%]
tests/test_dpe.py::test_business_rules_engine PASSED                     [ 13%]
tests/test_dpe.py::test_database_and_outbox PASSED                       [ 20%]
tests/test_dpe.py::test_feature_store_and_simulated_redis PASSED         [ 26%]
tests/test_dpe.py::test_demand_elasticity_model PASSED                   [ 33%]
tests/test_dpe.py::test_h3_calibrated_smoothing PASSED                   [ 40%]
tests/test_dpe.py::test_road_graph_topology PASSED                       [ 46%]
tests/test_dpe.py::test_driver_edge_eta PASSED                           [ 53%]
tests/test_dpe.py::test_edge_weight_fallback PASSED                      [ 60%]
tests/test_dpe.py::test_additive_surge_math PASSED                       [ 66%]
tests/test_dpe.py::test_switchback_toggle PASSED                         [ 73%]
tests/test_dpe.py::test_od_routing_distance PASSED                       [ 80%]
tests/test_driver_history_and_causal.py::test_driver_history_store_get_and_record PASSED [ 86%]
tests/test_driver_history_and_causal.py::test_driver_history_cell_fallback PASSED [ 93%]
tests/test_driver_history_and_causal.py::test_causal_enabled_config_flag PASSED [100%]

============================= 15 passed in 2.85s ==============================
```

---

## 3. Проверка E2E Smoke Script Output

Скрипт `python scripts/e2e_smoke.py --with-dpe` успешно выполнился против локально поднятых сервисов:

```text
=== E2E Smoke Verification ===
[Smoke] 1. Checking CPE health at http://localhost:8100/health ...
  [OK] CPE model_loaded=true, model_name='dml', source='dpe', feature_columns=['distance_km', 'duration_sec', 'price', 'surge_bonus', 'hour_of_day', 'past_trips', 'avg_surge']
[Smoke] 2. Testing CPE /predict_uplift ...
  [OK] CPE predict_uplift score=-0.1710, treatment='NO_DISCOUNT_AVOID', discount=0.0%
[Smoke] 3. Testing DPE integration at http://localhost:8000 ...
  [OK] DPE health ok: {'status': 'healthy', 'timestamp': 1786266351.2926285}
  [OK] DPE search 1 response: price=299.08, surge_multiplier=1.37, test_group='ADDITIVE'
  [OK] DPE search 2 response: price=254.22, surge_multiplier=1.16

=== Smoke Tests PASSED SUCCESSFULLY ===
```

---

## 4. Отклонения и невыполненные задачи

- **Отклонения:** Отсутствуют. Все требования спецификации `handoff_phase4.md` соблюдены.
- **Вторичные/Опциональные улучшения:** Режимы ретрейна и логирования `feature_columns` внедрены без привлечения тяжелых внешних сервисов (Redis/SQLite in-process history sidecar).
