from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import base64
import io
from huggingface_hub import hf_hub_download
import uvicorn
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None
transform = None

# Initialize FastAPI app
app = FastAPI(title="Age Prediction API", version="1.0.0")

# Add CORS middleware to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

# Response model
class PredictionResponse(BaseModel):
    age: float
    message: str = "Prediction successful"

class ErrorResponse(BaseModel):
    error: str

def load_model():
    """Load the age prediction model"""
    global model, transform
    
    try:
        logger.info("Loading model...")
        
        # 1. Download checkpoint
        repo_id = "sai9390/age_25epochs"
        filename = "best_effnetv2s_20to50.pt"
        
        logger.info(f"Downloading model from {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        # 2. Define model architecture (regression)
        logger.info("Creating model architecture...")
        model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=1)
        
        # 3. Load checkpoint
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", "").replace("module.", ""): v
                          for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint
        
        # 4. Load weights
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # 5. Define preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove the data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Age Prediction API is running!", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_age(request: ImageRequest):
    """Predict age from uploaded image"""
    
    # Check if model is loaded, if not try to load it
    if model is None or transform is None:
        logger.info("Model not loaded, attempting to load...")
        success = load_model()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="Model failed to load. Please check logs and try again later."
            )
    
    try:
        # Decode base64 image
        image = decode_base64_image(request.image)
        logger.info(f"Decoded image size: {image.size}")
        
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0)
        logger.info(f"Input tensor shape: {input_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor).item()
        
        # Clamp age to model's training range (20-50 years)
        predicted_age = max(20, min(50, output))  # Clamp between 20-50
        
        logger.info(f"Predicted age: {predicted_age:.2f}")
        
        return PredictionResponse(
            age=round(predicted_age, 2),
            message="Prediction successful"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Handle any other errors
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_age_batch(request: dict):
    """Predict ages for multiple images (batch processing)"""
    
    images = request.get("images", [])
    
    if model is None or transform is None:
        success = load_model()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Model failed to load. Please check logs and try again later."
            )
    
    try:
        predictions = []
        
        for i, base64_image in enumerate(images):
            try:
                # Decode and process each image
                image = decode_base64_image(base64_image)
                input_tensor = transform(image).unsqueeze(0)
                
                # Make prediction
                with torch.no_grad():
                    output = model(input_tensor).item()
                
                # Clamp age to model's training range (20-50 years)
                predicted_age = max(20, min(50, output))
                predictions.append({
                    "index": i,
                    "age": round(predicted_age, 2),
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                predictions.append({
                    "index": i,
                    "age": None,
                    "status": "error",
                    "error": str(e)
                })
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Load model on startup (this will run when the module is imported)
logger.info("Attempting to load model on startup...")
load_model()

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "version1:app",  # Make sure this matches your filename
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )