.PHONY: setup install test lint format clean train deploy

# Python version
PYTHON = python3

# Installation
setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv venv
	@echo "Please activate the virtual environment with: source venv/bin/activate"

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Installing dev dependencies..."
	pip install pytest pytest-cov black ruff mypy pre-commit xgboost
	pre-commit install

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov=steps

# Code Quality
lint:
	@echo "Linting code..."
	ruff check src/ steps/ tests/
	mypy src/ steps/

format:
	@echo "Formatting code..."
	black src/ steps/ tests/

# Execution
train:
	@echo "Running training pipeline..."
	$(PYTHON) pipelines/training_pipeline.py

deploy:
	@echo "Running Cloud Run deployment..."
	./scripts/deploy_cloudrun.sh

deploy-ab:
	@echo "Running A/B Test deployment..."
	./scripts/deploy_ab_test.sh

# Cleaning
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf mlruns/
