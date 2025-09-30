.PHONY: setup test fmt lint serve clean install

# Default target
all: setup

# Setup development environment
setup:
	pip install -e ".[dev]"
	pre-commit install

# Install package
install:
	pip install -e .

# Run tests
test:
	pytest tests/ -v --cov=fabriclite --cov-report=html

# Format code
fmt:
	black src/ tests/ examples/
	isort src/ tests/ examples/

# Lint code
lint:
	flake8 src/ tests/ examples/
	mypy src/

# Start FastAPI server
serve:
	uvicorn examples.server_fastapi:app --reload --host 0.0.0.0 --port 8000

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build:
	python -m build

# Install from source
install-dev:
	pip install -e ".[dev]"

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files

# Create test data directory structure
test-data:
	mkdir -p data/train/{cotton,denim,leather,silk,velvet,wool,linen,synthetic}
	mkdir -p data/val/{cotton,denim,leather,silk,velvet,wool,linen,synthetic}
	mkdir -p data/test/{cotton,denim,leather,silk,velvet,wool,linen,synthetic}

# Generate dummy test images
dummy-images:
	python -c "from PIL import Image; import os; [Image.new('RGB', (224, 224), (i*30, i*40, i*50)).save(f'data/train/cotton/img_{i}.jpg') for i in range(10)]"
	python -c "from PIL import Image; import os; [Image.new('RGB', (224, 224), (i*20, i*30, i*60)).save(f'data/val/cotton/img_{i}.jpg') for i in range(5)]"

# Help
help:
	@echo "Available targets:"
	@echo "  setup       - Install dependencies and setup pre-commit"
	@echo "  install     - Install package in development mode"
	@echo "  test        - Run tests with coverage"
	@echo "  fmt         - Format code with black and isort"
	@echo "  lint        - Lint code with flake8 and mypy"
	@echo "  serve       - Start FastAPI server"
	@echo "  clean       - Clean build artifacts"
	@echo "  build       - Build package"
	@echo "  test-data   - Create test data directory structure"
	@echo "  dummy-images- Generate dummy test images"
	@echo "  help        - Show this help message"
