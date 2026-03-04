import os
import io
import base64
import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Local Imports
from model import load_model, MobileNetV3Doodle, get_model_info
from preprocessor import DoodlePreprocessor
from utils import calculate_confidence, get_top_predictions, set_class_list, get_class_list

# Global variables for model state
state: Dict[str, Any] = {
    "model": None,
    "preprocessor": None,
    "device": None,
}

DEFAULT_CLASSES = [
    "cat", "dog", "bird", "fish", "cow",
    "apple", "banana", "pizza", "cake", "ice cream",
    "car", "bicycle", "airplane", "bus", "train",
    "house", "tree", "flower", "sun", "cloud",
    "star", "moon", "hand", "face", "clock",
    "book", "chair", "shoe", "key", "umbrella"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan manager for model initialization."""
    # Initialization
    state["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state["preprocessor"] = DoodlePreprocessor(image_size=96)
    
    model_path = os.getenv('MODEL_PATH', 'model/best.pth')
    
    if os.path.exists(model_path):
        try:
            state["model"] = load_model(model_path, num_classes=30, device=state["device"])
            
            # Load classes from checkpoint if available
            checkpoint = torch.load(model_path, map_location=state["device"])
            if isinstance(checkpoint, dict) and "classes" in checkpoint:
                set_class_list(checkpoint["classes"])
            else:
                set_class_list(DEFAULT_CLASSES)
            print(f"Model and classes loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            set_class_list(DEFAULT_CLASSES)
            state["model"] = MobileNetV3Doodle(num_classes=30).to(state["device"])
            state["model"].eval()
    else:
        print(f"Model file not found at {model_path}. Using untrained model.")
        state["model"] = MobileNetV3Doodle(num_classes=30).to(state["device"])
        state["model"].eval()
        set_class_list(DEFAULT_CLASSES)
        
    yield
    # Cleanup (if needed)
    state.clear()

app = FastAPI(
    title="DoodleNet Recognition API",
    description="High-performance MobileNetV3-based doodle recognition service.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class RecognitionRequest(BaseModel):
    prompt: str = Field(..., description="Target class to recognize", example="cat")
    image_base64: str = Field(..., description="Base64 encoded image string")

class RecognitionResponse(BaseModel):
    confidence: float = Field(..., description="Confidence score for target class (0-1)")
    prompt: str = Field(..., description="Target prompt")
    top_predictions: Optional[List[Dict[str, Any]]] = Field(None, description="Top 5 predictions")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_info: Optional[Dict[str, Any]] = None

class VisualizeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string")

class VisualizeResponse(BaseModel):
    original_size: List[int] = Field(..., description="Original image dimensions [width, height]")
    preprocessed_size: List[int] = Field(..., description="Preprocessed tensor size [H, W]")
    preprocessed_base64: str = Field(..., description="Base64 PNG of what the model sees (96×96)")
    pipeline_steps: List[str] = Field(..., description="Preprocessing steps applied")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_loaded": state["model"] is not None,
        "device": str(state["device"]),
        "model_info": get_model_info() if state["model"] else None
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if state["model"] is not None else "degraded",
        "model_loaded": state["model"] is not None,
        "device": str(state["device"]),
        "model_info": get_model_info() if state["model"] else None
    }

@app.get("/classes")
async def get_classes():
    """Get list of all available recognition classes."""
    try:
        classes = get_class_list()
        return {"classes": classes, "total": len(classes)}
    except Exception:
        return {"classes": [], "total": 0}

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_doodle(request: RecognitionRequest):
    """Recognize a single doodle from base64."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model initialization failed.")
    
    try:
        image_tensor = state["preprocessor"].preprocess_base64(request.image_base64)
        image_tensor = image_tensor.unsqueeze(0).to(state["device"])
        
        with torch.no_grad():
            output = state["model"](image_tensor)
        
        confidence = calculate_confidence(output, request.prompt.lower())
        top_preds = get_top_predictions(output, top_k=5)
        
        return RecognitionResponse(
            confidence=confidence,
            prompt=request.prompt,
            top_predictions=top_preds
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/visualize", response_model=VisualizeResponse)
async def visualize_preprocessing(request: VisualizeRequest):
    """
    Pipeline transparency endpoint.

    Returns the preprocessed 96×96 image that the model actually sees,
    along with metadata about the preprocessing steps applied.
    Useful for debugging, demos, and understanding model behavior.
    """
    try:
        # Decode the original image to get its dimensions
        preprocessor = state["preprocessor"]
        original_image = preprocessor.decode_base64_image(request.image_base64)
        orig_w, orig_h = original_image.size

        # Run the full preprocessing pipeline
        tensor = preprocessor.preprocess_base64(request.image_base64)
        h, w = tensor.shape[1], tensor.shape[2]

        # Convert tensor back to a viewable image (undo normalization)
        arr = tensor.squeeze(0).cpu().numpy()  # (H, W)
        arr = (arr * 0.5 + 0.5) * 255.0         # [-1,1] -> [0,255]
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Encode as base64 PNG
        pil_img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        preprocessed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return VisualizeResponse(
            original_size=[orig_w, orig_h],
            preprocessed_size=[h, w],
            preprocessed_base64=preprocessed_b64,
            pipeline_steps=[
                "1. RGB → Grayscale",
                "2. Binarize (threshold=200)",
                "3. Bounding-box crop (ink region)",
                "4. Center + white-pad to square",
                "5. Resize to 96×96 (INTER_AREA)",
                "6. Float [0,1] → Invert → Normalize [-1,1]",
            ]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

@app.post("/batch_recognize", response_model=List[RecognitionResponse])
async def batch_recognize(requests: List[RecognitionRequest]):
    """Perform recognition on a batch of doodles."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model initialization failed.")
    
    results = []
    for req in requests:
        try:
            image_tensor = state["preprocessor"].preprocess_base64(req.image_base64)
            image_tensor = image_tensor.unsqueeze(0).to(state["device"])
            
            with torch.no_grad():
                output = state["model"](image_tensor)
            
            confidence = calculate_confidence(output, req.prompt.lower())
            top_preds = get_top_predictions(output, top_k=5)
            
            results.append(RecognitionResponse(
                confidence=confidence,
                prompt=req.prompt,
                top_predictions=top_preds
            ))
        except Exception:
            results.append(RecognitionResponse(
                confidence=0.0,
                prompt=req.prompt,
                top_predictions=[]
            ))
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)