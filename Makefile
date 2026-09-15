install:
	python -m pip install -e '.[dev]'

lint:
	ruff check .

test:
	pytest

train:
	python -m src.train --limit 10000

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000
