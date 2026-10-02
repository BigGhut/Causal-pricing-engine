.PHONY: install demo proof train train-dpe evaluate serve test lint smoke smoke-dpe smoke-local

install:
	pip install -r requirements.txt
	pip install -e .

demo:
	python scripts/demo.py

proof:
	python scripts/portfolio_proof.py

train:
	python scripts/train.py --source synthetic

train-dpe:
	python scripts/train.py --source dpe

evaluate:
	python scripts/evaluate_experiment.py --source synthetic

summarize-dpe:
	python scripts/evaluate_experiment.py --source dpe

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8100

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

smoke:
	python scripts/e2e_smoke.py

smoke-dpe:
	python scripts/e2e_smoke.py --with-dpe

smoke-local:
	python scripts/e2e_smoke.py --start-servers --with-dpe
