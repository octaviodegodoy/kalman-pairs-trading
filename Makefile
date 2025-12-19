. PHONY: help install test lint format clean docker docs

help:
	@echo "Kalman Pairs Trading - Makefile Commands"
	@echo ""
	@echo "  install     Install dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linters"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  docker      Build Docker image"
	@echo "  run         Run dashboard"
	@echo "  docs        Build documentation"

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src tests
	mypy src --ignore-missing-imports

format: 
	black src tests examples
	isort src tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf . pytest_cache
	rm -rf . coverage
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker:
	docker build -t kalman-pairs-trading .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

run:
	streamlit run dashboard/app.py

docs:
	cd docs && make html

benchmark:
	python -m pytest tests/ --benchmark-only

release:
	python -m build
	twine check dist/*
	@echo "Ready to publish:  twine upload dist/*"