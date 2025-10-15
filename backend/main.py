"""
AI Text Detection API
FastAPI backend for detecting AI-generated vs human-written text
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from pathlib import Path
import logging
import os
import json
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Text Detection API",
    description="Detects whether text is AI-generated or human-written using DistilBERT",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")

class PredictionResponse(BaseModel):
    is_ai_generated: bool
    confidence: float
    label: str
    message: str
    model_version: Optional[str] = None

# Global model variables (loaded once at startup)
model = None
tokenizer = None
device = None
model_metadata = None
SEQ_LENGTH = 512
MODEL_VERSION = os.getenv('MODEL_VERSION', 'latest')

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer once when the API starts"""
    global model, tokenizer, device, model_metadata
    
    logger.info(f"Loading model version: {MODEL_VERSION}")
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Path to saved model (supports versioning)
    # Priority: 1. Environment variable, 2. ./saved_model (default)
    if MODEL_VERSION and MODEL_VERSION != 'saved_model':
        default_model_path = Path(__file__).parent.parent / "saved_model"
        model_path = Path(f"../training/models/{MODEL_VERSION}")
        if not model_path.exists():
            model_path = default_model_path
    else:
        model_path = default_model_path
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.info("Please save your trained model to ../saved_model/")
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    try:
        # Load tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Load metadata if available
        metadata_path = model_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                model_metadata = json.load(f)
            logger.info(f"Model metadata loaded - Accuracy: {model_metadata.get('metrics', {}).get('accuracy', 'N/A')}")
        else:
            model_metadata = {'version': MODEL_VERSION}
        
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "AI Text Detection API is online",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not set",
        "model_version": model_metadata.get('version') if model_metadata else MODEL_VERSION,
        "model_accuracy": model_metadata.get('metrics', {}).get('accuracy') if model_metadata else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict whether the given text is AI-generated or human-written
    
    Returns:
    - is_ai_generated: boolean indicating if text is AI-generated
    - confidence: probability score (0-1)
    - label: "AI-generated" or "Human-written"
    - message: descriptive message
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LENGTH,
            padding="max_length"
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get probabilities
            probs = torch.softmax(outputs.logits, dim=1)
            confidence = probs.max().item()
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        # Interpret results (0 = human, 1 = AI)
        is_ai = bool(prediction)
        label = "AI-generated" if is_ai else "Human-written"
        
        message = f"This text appears to be {label.lower()} with {confidence*100:.1f}% confidence."
        
        return PredictionResponse(
            is_ai_generated=is_ai,
            confidence=confidence,
            label=label,
            message=message,
            model_version=model_metadata.get('version') if model_metadata else MODEL_VERSION
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(texts: List[str]):
    """
    Predict multiple texts at once (more efficient)
    Limited to 10 texts per request to prevent abuse
    """
    if len(texts) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 texts per batch request")
    
    results = []
    for text in texts:
        try:
            result = await predict(PredictionRequest(text=text))
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e), "text_preview": text[:50]})
    
    return {"results": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
