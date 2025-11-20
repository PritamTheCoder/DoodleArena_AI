from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import os
from typing import Optional, List
import cv2
import numpy as np

# Imports
from model import load_model, MobileNetV3Doodle, get_model_info
from preprocessor import DoodlePreprocessor
from utils import calculate_confidence, get_top_predictions, get_class_list, set_class_list

# FastAPI 
app = FastAPI(
    title="DoodleNet Recognition API",
    description="MobileNetv3-based doodle recognition for Quick Draw dataset",
    version="1.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
PREPROCESSOR = None
DEVICE = None

# Default classes to prevent crashes if checkpoint is missing them
DEFAULT_CLASSES = [
    "cat", "dog", "bird", "fish", "cow",
    "apple", "banana", "pizza", "cake", "ice cream",
    "car", "bicycle", "airplane", "bus", "train",
    "house", "tree", "flower", "sun", "cloud",
    "star", "moon", "hand", "face", "clock",
    "book", "chair", "shoe", "key", "umbrella"
]

# Request/Response models
class RecognitionRequest(BaseModel):
    prompt: str = Field(..., description="Target class to recognize (e.g., 'cat', 'house')")
    image_base64: str = Field(..., description="Base64 encoded image string")

class RecognitionResponse(BaseModel):
    confidence: float = Field(..., description="Confidence score for target class (0-1)")
    prompt: str = Field(..., description="Target prompt")
    top_predictions: Optional[List[dict]] = Field(None, description="Top 5 predictions")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_info: Optional[dict] = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global MODEL, PREPROCESSOR, DEVICE
    
    # Set device cuda if available
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Initialize preprocessor
    PREPROCESSOR = DoodlePreprocessor(image_size=96)
    
    # Load model path
    model_path = os.getenv('MODEL_PATH', 'model/best.pth')
    
    if os.path.exists(model_path):
        try:
            # 1. Load Model
            MODEL = load_model(model_path, num_classes=30, device=DEVICE)
            print(f"Model loaded successfully from {model_path}")

            # 2. Initialize Class List from Checkpoint 
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "classes" in checkpoint:
                set_class_list(checkpoint["classes"])
                print(f"Loaded {len(checkpoint['classes'])} classes from checkpoint.")
            else:
                set_class_list(DEFAULT_CLASSES)
                print("Checkpoint missing classes, using default list.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            set_class_list(DEFAULT_CLASSES)
    else:
        print(f"Model file not found at {model_path}")
        print("Using untrained model with default classes")
        MODEL = MobileNetV3Doodle(num_classes=30).to(DEVICE)
        MODEL.eval()
        set_class_list(DEFAULT_CLASSES)

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "model_info": get_model_info() if MODEL else None
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if MODEL is not None else "model_not_loaded",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "model_info": get_model_info()
    }

@app.get("/classes")
async def get_classes():
    """Get list of all available classes."""
    try:
        classes = get_class_list()
        return {"classes": classes, "total": len(classes)}
    except:
        return {"classes": [], "total": 0}

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_doodle(request: RecognitionRequest):
    """Recognize a doodle and return confidence score."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        image_tensor = PREPROCESSOR.preprocess_base64(request.image_base64)
        # 3d -> 4d image tensor | (C, H, W) -> (B, C, H, W)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)
        
        with torch.no_grad():
            output = MODEL(image_tensor)
        
        confidence = calculate_confidence(output, request.prompt.lower())
        top_preds = get_top_predictions(output, top_k=5)
        
        return RecognitionResponse(
            confidence=confidence,
            prompt=request.prompt,
            top_predictions=top_preds
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition error: {str(e)}")

@app.post("/batch_recognize")
async def batch_recognize(requests: List[RecognitionRequest]):
    """Batch recognition endpoint."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for req in requests:
        try:
            image_tensor = PREPROCESSOR.preprocess_base64(req.image_base64)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(DEVICE)
            
            arr = image_tensor.squeeze(0).cpu().numpy()
            vis = ((arr * 0.5 + 0.5) * 255).astype(np.uint8)
            vis = np.squeeze(vis)
            cv2.imwrite("debug_what_model_sees.png", vis)
            with torch.no_grad():
                output = MODEL(image_tensor)
            
            confidence = calculate_confidence(output, req.prompt.lower())
            top_preds = get_top_predictions(output, top_k=5)
            
            results.append(RecognitionResponse(
                confidence=confidence,
                prompt=req.prompt,
                top_predictions=top_preds
            ))
        except Exception as e:
            # Fail gracefully for individual items in batch
            results.append({
                "confidence": 0.0,
                "prompt": req.prompt,
                "top_predictions": [],
                "error": str(e)
            })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)