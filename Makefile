.PHONY: help setup install-training install-backend install-all train-env backend-env clean test

help:
	@echo "GPTOne_2701Project - Available Commands"
	@echo "========================================"
	@echo "setup              - Create both virtual environments"
	@echo "install-training   - Install training dependencies"
	@echo "install-backend    - Install backend dependencies"
	@echo "install-all        - Install all dependencies"
	@echo "train-env          - Activate training environment (info only)"
	@echo "backend-env        - Activate backend environment (info only)"
	@echo "train              - Train a new model version"
	@echo "serve              - Start backend API"
	@echo "test-api           - Test backend API"
	@echo "clean              - Remove virtual environments"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup && make install-all"

setup:
	@echo "Creating virtual environments..."
	cd training && python -m venv venv_training
	cd backend && python -m venv venv_backend
	@echo "✓ Virtual environments created"
	@echo ""
	@echo "Next: Run 'make install-all' to install dependencies"

install-training:
	@echo "Installing training dependencies..."
	cd training && ./venv_training/bin/pip install -r requirements.txt
	@echo "✓ Training dependencies installed"

install-backend:
	@echo "Installing backend dependencies..."
	cd backend && ./venv_backend/bin/pip install -r requirements.txt
	@echo "✓ Backend dependencies installed"

install-all: install-training install-backend
	@echo ""
	@echo "✓ All dependencies installed!"
	@echo ""
	@echo "To activate environments:"
	@echo "  Training: cd training && source venv_training/bin/activate"
	@echo "  Backend:  cd backend && source venv_backend/bin/activate"

train-env:
	@echo "To activate training environment, run:"
	@echo "  cd training && source venv_training/bin/activate"

backend-env:
	@echo "To activate backend environment, run:"
	@echo "  cd backend && source venv_backend/bin/activate"

train:
	@echo "Training new model..."
	cd training && ./venv_training/bin/python train.py --version $$(date +%Y%m%d_%H%M%S) --epochs 5

serve:
	@echo "Starting backend API..."
	cd backend && ./venv_backend/bin/uvicorn main:app --reload

test-api:
	@echo "Testing backend API..."
	cd backend && ./venv_backend/bin/python test_api.py

clean:
	@echo "Removing virtual environments..."
	rm -rf training/venv_training
	rm -rf backend/venv_backend
	@echo "✓ Virtual environments removed"

# Docker commands
docker-build-backend:
	docker build -t ai-detector-backend ./backend

docker-build-training:
	docker build -t ai-detector-training ./training

docker-run-backend:
	docker run -p 8000:8000 ai-detector-backend

# Development helpers
lint-backend:
	cd backend && ./venv_backend/bin/python -m black main.py
	cd backend && ./venv_backend/bin/python -m flake8 main.py

format:
	cd backend && ./venv_backend/bin/python -m black .
	cd training && ./venv_training/bin/python -m black .
