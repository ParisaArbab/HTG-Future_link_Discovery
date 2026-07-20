install:
	pip install -r requirements.txt

install-lite:
	pip install -r requirements-lite.txt

prepare:
	python scripts/prepare_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

api:
	uvicorn app:app --reload

test:
	pytest --cov=src --cov=app --cov-report=term-missing
