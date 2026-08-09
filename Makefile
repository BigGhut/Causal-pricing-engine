.PHONY: install train serve test lint

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python scripts/train.py

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
