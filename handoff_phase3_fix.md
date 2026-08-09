# Fix-handoff: Phase 3 MUST (Docker port)

> **Автор:** оркестратор  
> **Дата:** 2026-08-09  
> **Аудит:** [audit_report_phase3.md](audit_report_phase3.md)  
> **Скоп:** только BLOCKER T17. Не рефакторить T13–T16.

---

## Проблема

`Dockerfile` поднимает uvicorn на **8000**, а `docker-compose.yml` / `docker-compose.full.yml` публикуют **8100:8100** и ставят `CPE_API__PORT=8100`.  
Env `CPE_API__PORT` **не** меняет порт процесса uvicorn (он зашит в `CMD`).

---

## Задачи

### F1. Dockerfile

Файл: `Dockerfile`

1. `EXPOSE 8100`
2. `CMD` слушает 8100, например:

```dockerfile
EXPOSE 8100
ENV CPE_API__PORT=8100
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8100"]
```

Либо (предпочтительно, если удобно на Windows/Linux):

```dockerfile
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${CPE_API__PORT:-8100}
```

(shell-form, чтобы подставлялся env).

### F2. Согласованность compose

Убедиться, что:

- `docker-compose.yml` → `8100:8100`, `CPE_API__PORT=8100`
- `docker-compose.full.yml` → causal-engine `8100:8100`, `CPE_API__PORT=8100`

Менять порты DPE (`8000`) **не нужно**.

### F3. Приёмка

```bash
cd Z:\pet-project\causal-pricing-engine
docker compose -f docker-compose.yml up --build -d
# дождаться старта
curl http://localhost:8100/health
# ожидание: JSON со status / model_loaded
docker compose -f docker-compose.yml down
```

Если Docker на машине недоступен — минимум: показать diff Dockerfile + утверждение, что `CMD`/`EXPOSE` = 8100; оркестратор проверит статически.

### F4. Регрессия

```bash
python -m pytest tests/ -v
```

Должно остаться **26 passed** (или больше, если добавите тесты).

---

## Чего НЕ делать

- Не трогать логику `dpe_connector`, `evaluate_experiment`, DPE `main.py` (кроме если curl покажет другой баг).
- Не менять default synthetic train path.
- Не расширять Phase 4 (feature store, e2e) в этом fix.

---

## Критерий «Phase 3 CLOSED»

- [ ] F1 + F2 сделаны  
- [ ] F3 health на `:8100` (или статический review Dockerfile = 8100)  
- [ ] F4 pytest green  
- [ ] Оркестратор обновляет `audit_report_phase3.md` → **PASS**
