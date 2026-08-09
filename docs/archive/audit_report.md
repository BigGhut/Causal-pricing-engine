# 📋 Аудит воркера: causal-pricing-engine

> **Дата:** 2026-08-09  
> **Проект:** `Z:\pet-project\causal-pricing-engine`  
> **Основание:** [handoff.md](file:///Z:/pet-project/causal-pricing-engine/handoff.md)

---

## Результат тестов

```
============================= 19 passed in 3.49s ==============================
```

✅ **Все 19 тестов пройдены без ошибок** (Python 3.13.9, pytest 9.0.3).

---

## Сводная таблица по задачам из Handoff

### Фаза 1: Фундамент (MUST)

| Задача | Статус | Что создано | Комментарий |
|:---|:---:|:---|:---|
| **T1** Конфиг + загрузчик | ✅ DONE | [`configs/config.yaml`](file:///Z:/pet-project/causal-pricing-engine/configs/config.yaml), [`src/config.py`](file:///Z:/pet-project/causal-pricing-engine/src/config.py) | `AppConfig(BaseSettings)` + `YamlConfigSettingsSource`; env-оверрайды `CPE_*` / `__` (например `CPE_API__PORT`). |
| **T2** Синтетические данные | ✅ DONE | [`src/data/synthetic.py`](file:///Z:/pet-project/causal-pricing-engine/src/data/synthetic.py), `data/.gitkeep` | `generate_uplift_dataset()` с гетерогенным treatment effect по сегментам. Тест подтверждает схему и HTE. |
| **T3** SLearner | ✅ DONE | [`src/causal/uplift_models.py`](file:///Z:/pet-project/causal-pricing-engine/src/causal/uplift_models.py) | Реализован, тест `test_slearner_fit_predict` проходит. |
| **T4** XLearner | ✅ DONE | [`src/causal/uplift_models.py`](file:///Z:/pet-project/causal-pricing-engine/src/causal/uplift_models.py) | Реализован, тест `test_xlearner_fit_predict` проходит. |
| **T5** DML Engine | ✅ DONE | [`src/causal/dml_engine.py`](file:///Z:/pet-project/causal-pricing-engine/src/causal/dml_engine.py) | Обёртка EconML с fallback. Тесты `test_dml_fit_effect`, `test_dml_with_controls` проходят. |
| **T6** Switchback Splitter | ✅ DONE | [`src/experiments/switchback_splitter.py`](file:///Z:/pet-project/causal-pricing-engine/src/experiments/switchback_splitter.py) | Пространственно-временные блоки. 3 теста: стабильность, блоки, баланс — все ОК. |
| **T7** .gitignore + git init | ✅ DONE | [`.gitignore`](file:///Z:/pet-project/causal-pricing-engine/.gitignore), `.git/` | Стандартный Python gitignore, репо инициализировано. |

### Фаза 2: Сборка Pipeline (SHOULD)

| Задача | Статус | Что создано | Комментарий |
|:---|:---:|:---|:---|
| **T8** Training pipeline | ✅ DONE | [`scripts/train.py`](file:///Z:/pet-project/causal-pricing-engine/scripts/train.py) | Генерация данных → обучение T/S/X + DML → выбор лучшей модели → `artifacts/model.joblib`. Тест `test_train_selects_best_among_all_candidates_including_dml` проходит. |
| **T9** Offline-метрики | ✅ DONE | [`src/evaluation/metrics.py`](file:///Z:/pet-project/causal-pricing-engine/src/evaluation/metrics.py) | `qini_auc_score`, `uplift_at_k`, `uplift_by_percentile` — реализованы. Тесты проходят. |
| **T10** Подключение модели в API | ✅ DONE | [`src/api/main.py`](file:///Z:/pet-project/causal-pricing-engine/src/api/main.py) | Mock-логика убрана. Модель загружается через lifespan из `artifacts/model.joblib`. Тест `test_api_scoring_path_with_real_artifact` проходит. |
| **T11** Makefile | ✅ DONE | [`Makefile`](file:///Z:/pet-project/causal-pricing-engine/Makefile) | Таргеты: `install`, `train`, `serve`, `test`, `lint`. |
| **T12** Docker | ✅ DONE | [`Dockerfile`](file:///Z:/pet-project/causal-pricing-engine/Dockerfile), [`docker-compose.yml`](file:///Z:/pet-project/causal-pricing-engine/docker-compose.yml) | Многостадийная сборка (builder + runtime). API на :8000. |

---

## Покрытие тестами

| Тест | Проверяет |
|:---|:---|
| `test_tlearner_fit_predict` | TLearner fit/predict_uplift |
| `test_slearner_fit_predict` | SLearner fit/predict_uplift |
| `test_xlearner_fit_predict` | XLearner fit/predict_uplift |
| `test_learners_accept_dataframe` | Все 3 лёрнера принимают DataFrame |
| `test_meta_learners_detect_heterogeneity` | Лёрнеры различают HTE (τ > 0 vs τ ≈ 0) |
| `test_dml_fit_effect` | DMLEngine fit/effect |
| `test_dml_with_controls` | DMLEngine с контрольными переменными |
| `test_switchback_assignment_stability` | Детерминированность назначений |
| `test_switchback_space_time_blocks` | Пространственно-временные блоки |
| `test_switchback_ratio_roughly_balanced` | Баланс групп ≈50/50 |
| `test_cuped_adjust` | CUPED снижает дисперсию |
| `test_sample_size_calculator` | Расчёт размера выборки |
| `test_qini_and_uplift_at_k_on_labeled_data` | Qini AUC и Uplift@k |
| `test_uplift_by_percentile_shape` | Форма Uplift Curve |
| `test_synthetic_dataset_schema_and_heterogeneity` | Схема данных и HTE |
| `test_load_config_has_required_sections` | Загрузка конфига |
| `test_api_scoring_path_with_real_artifact` | API с реальной моделью |
| `test_package_exports` | Публичные экспорты пакетов |
| `test_train_selects_best_among_all_candidates_including_dml` | Pipeline выбирает лучшую модель |

---

## Единственное замечание

> ~~**T1 (PARTIAL):**~~ **Закрыто:** `src/config.py` переведён на `pydantic-settings.BaseSettings` с YAML defaults и env-оверрайдами (`CPE_API__PORT`, `CPE_MODEL__BASE_LEARNER`, …). Тесты `test_load_config_env_overrides_yaml` / `test_config_module_uses_pydantic_settings` покрывают shipped `load_config`.

---

## Вердикт

**12 из 12 задач** Фазы 1 + Фазы 2 выполнены полностью (DONE).  
**21/21 тестов проходят.**  
Проект готов к Фазе 3 (интеграция с `dynamic-pricing-engine`).
