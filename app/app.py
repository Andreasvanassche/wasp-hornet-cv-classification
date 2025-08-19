"""
Hornet & Wasp Classification Web Application
FastAPI backend with model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hornet & Wasp Classifier",
    description="AI-powered classification of hornet and wasp species",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
class Config:
    MODEL_PATH = "./model/best_model_resnet50_cropped.pth"  # Updated for Docker
    CLASS_NAMES = ["Vespa_crabro", "Vespa_velutina", "Vespula_sp"]
    IMAGE_SIZE = (224, 224)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Response models
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# ResNet Transfer Learning Model (same as training)
class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=3, pretrained=False):
        super(ResNetTransfer, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# Global model instance
model = None
transform = None

def load_model():
    """Load the trained model"""
    global model, transform
    
    try:
        logger.info("Loading model...")
        model = ResNetTransfer(num_classes=len(config.CLASS_NAMES), pretrained=False)
        
        # Load model weights
        if os.path.exists(config.MODEL_PATH):
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
            logger.info(f"Model loaded from {config.MODEL_PATH}")
        else:
            logger.error(f"Model file not found: {config.MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {config.MODEL_PATH}")
        
        model.to(config.DEVICE)
        model.eval()
        
        # Define image preprocessing transform
        transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    global transform
    
    if transform is None:
        raise RuntimeError("Transform not initialized - model may not be loaded properly")
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(config.DEVICE)

def predict_image(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on preprocessed image"""
    global model
    import time
    start_time = time.time()
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get prediction
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = config.CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        # Create probability dictionary
        prob_dict = {
            class_name: prob.item() 
            for class_name, prob in zip(config.CLASS_NAMES, probabilities)
        }
        
        processing_time = time.time() - start_time
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "processing_time": processing_time
        }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup!")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hornet & Wasp Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #999; background-color: #f9f9f9; }
            .result { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .confidence { font-size: 1.2em; font-weight: bold; }
            .probabilities { margin: 10px 0; }
            .prob-bar { background-color: #f0f0f0; height: 20px; margin: 5px 0; position: relative; }
            .prob-fill { height: 100%; background-color: #4CAF50; transition: width 0.5s; }
            .prob-label { position: absolute; left: 10px; top: 2px; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>🐝 Hornet & Wasp Classifier</h1>
        <p>Upload an image to classify hornet and wasp species using AI.</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p>Click here or drag and drop an image</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
        </div>
        
        <div id="result" style="display: none;"></div>
        
        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                
                if (fileInput.files.length === 0) return;
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                resultDiv.innerHTML = '<p>Processing image...</p>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let html = `
                            <h3>Prediction Results</h3>
                            <p class="confidence">Predicted: <strong>${result.predicted_class}</strong></p>
                            <p>Confidence: <strong>${(result.confidence * 100).toFixed(1)}%</strong></p>
                            <div class="probabilities">
                                <h4>All Probabilities:</h4>`;
                        
                        for (const [species, prob] of Object.entries(result.probabilities)) {
                            html += `
                                <div class="prob-bar">
                                    <div class="prob-fill" style="width: ${prob * 100}%"></div>
                                    <div class="prob-label">${species}: ${(prob * 100).toFixed(1)}%</div>
                                </div>`;
                        }
                        
                        html += `</div>
                            <p><small>Processing time: ${result.processing_time.toFixed(3)}s</small></p>`;
                        
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `<p style="color: red;">Error: ${result.detail}</p>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(config.DEVICE)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict hornet/wasp species from uploaded image"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        prediction = predict_image(image_tensor)
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get list of available classes"""
    return {"classes": config.CLASS_NAMES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)