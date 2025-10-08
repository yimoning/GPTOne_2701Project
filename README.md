# GPTOne AI Text Detection

Detects AI-generated text using DistilBERT. Includes FastAPI backend and training pipeline.

## Features

- Binary classification (AI-generated vs human-written)
- FastAPI REST API with batch prediction
- Model versioning and training pipeline
- Storage support (local, S3, HuggingFace Hub)
- Web UI for testing

## Requirements

- Python 3.8+

## Quick Start

```bash
# 1. Setup environments and install dependencies
make setup
make install-all

# 2. Start API
make serve

# 3. Test (in another terminal)
curl http://localhost:8000/health
```

Or manually:
```bash
cd backend
python -m venv venv_backend
source venv_backend/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Project Structure

```
├── backend/              # FastAPI server
├── training/             # Model training pipeline  
├── storage/              # Model storage (S3/HF/local)
├── frontend/             # Web UI
├── data/                 # Dataset
└── .github/workflows/    # CI/CD
```

## Usage

```python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Your text here"}
)
print(response.json())
```

## Makefile Commands

```bash
make help              # Show all available commands
make setup             # Create virtual environments
make install-all       # Install all dependencies
make serve             # Start backend API
make train             # Train new model with timestamp version
make test-api          # Test backend API
make clean             # Remove virtual environments
```

**Common workflows:**
```bash
# Initial setup
make setup && make install-all

# Development
make serve              # Start backend
make train              # Train model
make test-api           # Test API

# Training environment
cd training && source venv_training/bin/activate

# Backend environment
cd backend && source venv_backend/bin/activate
```

## Documentation

- [backend/README.md](backend/README.md) - API docs
- [training/README.md](training/README.md) - Training guide  
- [storage/README.md](storage/README.md) - Storage options

## License

See [LICENSE](LICENSE) file.
