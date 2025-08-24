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
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
import tempfile
import os
import subprocess
import sys

# Configure logging EARLY so logger is available for optional import warnings
import logging as _logging
_logging.basicConfig(level=_logging.INFO)
logger = _logging.getLogger(__name__)

# Try to import additional audio processing libraries
try:
    from pydub import AudioSegment
    from pydub.utils import which
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    # logger available now
    logger.warning("pydub not available. Install with: pip install pydub")

try:
    import av  # PyAV for handling various audio/video formats
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    logger.warning("PyAV not available. Install with: pip install av")

try:
    import mutagen  # For audio metadata and basic decoding
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logger.warning("mutagen not available. Install with: pip install mutagen")

# Helper to summarize optional audio stack
def ensure_audio_packages():
    missing = []
    if not PYDUB_AVAILABLE:
        missing.append("pydub")
    if not PYAV_AVAILABLE:
        missing.append("av")
    if not MUTAGEN_AVAILABLE:
        missing.append("mutagen")
    if missing:
        logger.warning(
            "Optional audio packages missing (%s). Core decoding still works via soundfile/librosa; install with: pip install %s",
            ", ".join(missing),
            " ".join(missing)
        )
    else:
        logger.info("All optional audio packages present.")

# Global model variables
image_model = None
image_transform = None
audio_model = None
audio_processor = None

# Initialize FastAPI app
app = FastAPI(title="Multi-Modal Age Prediction API", version="1.0.0")

# Add CORS middleware to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Audio Model Architecture ===
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()
    
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = self.gender(hidden_states)
        return hidden_states, logits_age, logits_gender

# Request models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class AudioRequest(BaseModel):
    audio: str  # Base64 encoded audio

class MultiModalRequest(BaseModel):
    image: Optional[str] = None  # Base64 encoded image
    audio: Optional[str] = None  # Base64 encoded audio

# Response models
class ImagePredictionResponse(BaseModel):
    status: str
    age: Optional[float] = None
    error: Optional[str] = None

class AudioPredictionResponse(BaseModel):
    status: str
    age_years: Optional[float] = None
    gender_probs: Optional[dict] = None
    error: Optional[str] = None

class MultiModalResponse(BaseModel):
    image_prediction: Optional[ImagePredictionResponse] = None
    audio_prediction: Optional[AudioPredictionResponse] = None
    combined_age: Optional[float] = None
    message: str = "Prediction successful"

def load_image_model():
    """Load the image age prediction model"""
    global image_model, image_transform
    
    try:
        logger.info("Loading image model...")
        
        # Download checkpoint
        repo_id = "sai9390/age_25epochs"
        filename = "best_effnetv2s_20to50.pt"
        
        logger.info(f"Downloading image model from {repo_id}...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        # Define model architecture (regression)
        logger.info("Creating image model architecture...")
        image_model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=1)
        
        # Load checkpoint
        logger.info("Loading image checkpoint...")
        checkpoint = torch.load(model_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", "").replace("module.", ""): v
                          for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint
        
        # Load weights
        image_model.load_state_dict(state_dict, strict=False)
        image_model.eval()
        
        # Define preprocessing transforms
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        logger.info("✅ Image model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading image model: {str(e)}")
        return False

def load_audio_model():
    """Load the audio age/gender prediction model"""
    global audio_model, audio_processor
    
    try:
        logger.info("Loading audio model...")
        
        MODEL_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"
        
        logger.info(f"Loading audio processor from {MODEL_ID}...")
        audio_processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        
        logger.info(f"Loading audio model from {MODEL_ID}...")
        audio_model = AgeGenderModel.from_pretrained(MODEL_ID)
        audio_model.eval()
        
        logger.info("✅ Audio model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading audio model: {str(e)}")
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

def decode_base64_audio(base64_string: str) -> np.ndarray:
    """
    Decode base64 string into a 16kHz mono float32 numpy array.
    This function prioritizes using PyAV for robust decoding of various formats.
    """
    try:
        if base64_string.startswith('data:audio'):
            base64_string = base64_string.split(',')[1]

        audio_bytes = base64.b64decode(base64_string)

        if len(audio_bytes) < 100:
            raise ValueError("Audio payload is too small to be valid.")

        # Primary decoding strategy: Use PyAV for everything.
        # It's more robust for container formats like webm, ogg, mp4, etc.
        if not PYAV_AVAILABLE:
            logger.error("PyAV is not installed, which is required for robust audio decoding.")
            raise ImportError("PyAV library not found. Please install it via 'pip install av'.")

        try:
            with av.open(io.BytesIO(audio_bytes)) as container:
                audio_stream = container.streams.audio[0]
                
                # Decode all frames and concatenate
                frames = []
                for frame in container.decode(audio_stream):
                    frames.append(frame.to_ndarray())
                
                if not frames:
                    raise ValueError("No audio frames could be decoded from the file.")

                audio_array = np.concatenate(frames, axis=1)
                
                # Convert to mono float32
                if audio_array.shape[0] > 1:
                    audio_array = np.mean(audio_array, axis=0)
                else:
                    audio_array = audio_array.flatten()
                
                audio_array = audio_array.astype(np.float32)
                
                # Resample to 16kHz if necessary
                source_sr = audio_stream.rate
                if source_sr != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=source_sr, target_sr=16000)
                
                logger.info(f"Successfully decoded audio with PyAV. Original SR: {source_sr}, Final samples: {len(audio_array)}")
                return audio_array

        except Exception as e:
            logger.error(f"PyAV decoding failed: {e}. This might indicate a corrupted file or an unsupported codec within the container.")
            # Add a specific hint for a common issue on Windows
            if "Permission denied" in str(e) and sys.platform == "win32":
                 logger.error("A 'Permission denied' error on Windows with PyAV can sometimes be related to antivirus software or missing system libraries. Ensure your FFmpeg installation is from a trusted source.")
            raise ValueError(f"Failed to decode audio with PyAV: {e}")

    except Exception as e:
        logger.error(f"Audio decode failure: {e}")
        error_msg = str(e)
        if "PyAV" in error_msg or "FFmpeg" in error_msg:
             final_detail = (
                "Audio decoding failed. The server's audio-video library (PyAV/FFmpeg) could not process the file. "
                "Please ensure the uploaded file is a standard, uncorrupted audio format. "
                "If the issue persists, the server environment may need FFmpeg installed and configured correctly."
            )
        else:
            final_detail = f"Invalid audio data provided: {e}"
        
        raise HTTPException(status_code=400, detail=final_detail)

def predict_image_age(image: Image.Image) -> ImagePredictionResponse:
    """Predict age from image"""
    try:
        if image_model is None or image_transform is None:
            return ImagePredictionResponse(
                status="error",
                error="Image model not loaded"
            )
        
        # Preprocess image
        input_tensor = image_transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = image_model(input_tensor).item()
        
        # Clamp age to model's training range (20-50 years)
        predicted_age = max(20, min(50, output))
        
        return ImagePredictionResponse(
            status="success",
            age=round(predicted_age, 2)
        )
        
    except Exception as e:
        logger.error(f"Image prediction error: {str(e)}")
        return ImagePredictionResponse(
            status="error",
            error=str(e)
        )

def predict_audio_age(audio_array: np.ndarray) -> AudioPredictionResponse:
    """Predict age and gender from audio"""
    try:
        if audio_model is None or audio_processor is None:
            return AudioPredictionResponse(
                status="error",
                error="Audio model not loaded"
            )
        
        # Process audio
        inputs = audio_processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs["input_values"]
        
        # Make prediction
        with torch.no_grad():
            _, logits_age, logits_gender = audio_model(input_values)
        
        # Process age (normalize from 0-1 to 0-100 years)
        age_norm = logits_age.squeeze().item()
        age_years = age_norm * 100
        
        # Process gender probabilities
        gender_probs = torch.softmax(logits_gender.squeeze(), dim=0).tolist()
        labels = ["child", "female", "male"]
        gender_dict = dict(zip(labels, gender_probs))
        
        return AudioPredictionResponse(
            status="success",
            age_years=round(age_years, 2),
            gender_probs=gender_dict
        )
        
    except Exception as e:
        logger.error(f"Audio prediction error: {str(e)}")
        return AudioPredictionResponse(
            status="error",
            error=str(e)
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Multi-Modal Age Prediction API is running!",
        "image_model_loaded": image_model is not None,
        "audio_model_loaded": audio_model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "image_model_loaded": image_model is not None,
        "audio_model_loaded": audio_model is not None,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/predict/image")
async def predict_image_only(request: ImageRequest):
    """Predict age from image only"""
    
    # Check if image model is loaded
    if image_model is None or image_transform is None:
        success = load_image_model()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Image model failed to load"
            )
    
    try:
        # Decode and predict
        image = decode_base64_image(request.image)
        result = predict_image_age(image)
        
        if result.status == "success":
            return {"age": result.age, "message": "Image prediction successful"}
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/audio")
async def predict_audio_only(request: AudioRequest):
    """Predict age from audio only"""
    
    # Check if audio model is loaded
    if audio_model is None or audio_processor is None:
        success = load_audio_model()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Audio model failed to load"
            )
    
    try:
        # Decode and predict
        audio_array = decode_base64_audio(request.audio)
        result = predict_audio_age(audio_array)
        
        if result.status == "success":
            return {
                "age_years": result.age_years,
                "gender_probs": result.gender_probs,
                "message": "Audio prediction successful"
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/multimodal", response_model=MultiModalResponse)
async def predict_multimodal(request: MultiModalRequest):
    """Predict age using both image and audio (or just one)"""
    
    if not request.image and not request.audio:
        raise HTTPException(
            status_code=400,
            detail="At least one input (image or audio) is required"
        )
    
    # Load models if needed
    if request.image and (image_model is None or image_transform is None):
        success = load_image_model()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Image model failed to load"
            )
    
    if request.audio and (audio_model is None or audio_processor is None):
        success = load_audio_model()
        if not success:
            raise HTTPException(
                status_code=503,
                detail="Audio model failed to load"
            )
    
    try:
        response = MultiModalResponse()
        ages_for_average = []
        
        # Process image if provided
        if request.image:
            image = decode_base64_image(request.image)
            image_result = predict_image_age(image)
            response.image_prediction = image_result
            
            if image_result.status == "success":
                ages_for_average.append(image_result.age)
        
        # Process audio if provided
        if request.audio:
            audio_array = decode_base64_audio(request.audio)
            audio_result = predict_audio_age(audio_array)
            response.audio_prediction = audio_result
            
            if audio_result.status == "success":
                ages_for_average.append(audio_result.age_years)
        
        # Calculate combined age if we have successful predictions
        if ages_for_average:
            response.combined_age = round(sum(ages_for_average) / len(ages_for_average), 2)
            response.message = "Multi-modal prediction successful"
        else:
            response.message = "No successful predictions"
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-modal prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-modal prediction failed: {str(e)}"
        )

# Ensure required packages and load models on startup
logger.info("Ensuring audio processing packages are available...")
ensure_audio_packages()

logger.info("Loading models on startup...")
load_image_model()
load_audio_model()

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Correct module:app reference
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )