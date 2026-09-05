.PHONY: test build dev-backend dev-frontend deploy

test:
	python -m pytest -q
	python -m ruff check backend scripts
	python -m ruff format --check backend scripts

build:
	npm ci --prefix frontend
	npm run build --prefix frontend

dev-backend:
	PYTHONPATH=backend python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

dev-frontend:
	npm run dev --prefix frontend

deploy:
	docker compose -f docker-compose.gpu.yml up -d --build
