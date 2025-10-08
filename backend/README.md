# Backend API

FastAPI REST API for AI text detection using DistilBERT.

## Requirements

- Python 3.8+
- Model in `saved_model/` or `../training/models/{VERSION}/`

## Setup

```bash
cd backend
python -m venv venv_backend
source venv_backend/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --workers 4
```

Server at `http://localhost:8000`

## API Endpoints

**`GET /health`**
```json
{"status": "healthy", "model_loaded": true, "model_version": "v1.0.0"}
```

**`POST /predict`**
```json
// Request
{"text": "Your essay text"}

// Response
{"prediction": "AI-generated", "confidence": 0.87, "model_version": "v1.0.0"}
```

**`POST /batch-predict`**
```json
// Request
{"texts": ["Text 1", "Text 2"]}

// Response
{"predictions": [{"prediction": "AI-generated", "confidence": 0.87}]}
```

## Usage

```python
import requests
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Your text"}
)
print(response.json())
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text"}'
```

## Environment Variables

```bash
MODEL_VERSION=production    # Model version to load (optional)
MODEL_STORAGE=local         # local, s3, or huggingface (optional)
```

## Testing

```bash
python test_api.py
curl http://localhost:8000/health
```

Or open `../frontend/index.html` in browser.
