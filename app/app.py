"""
Hornet & Wasp Classification Web Application
FastAPI backend with model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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
import uuid
import shutil

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

# Create uploads directory if it doesn't exist
Path("uploads").mkdir(exist_ok=True)

# Model configuration
class Config:
    MODEL_PATH = "./model/best_model_resnet50_cropped.pth"  # Updated path
    CLASS_NAMES = ["Vespa_crabro", "Vespa_velutina", "Vespula_sp"]
    DISPLAY_NAMES = {
        "Vespa_crabro": "European Hornet",
        "Vespa_velutina": "Asian Hornet", 
        "Vespula_sp": "Common Wasp"
    }
    IMAGE_SIZE = (224, 224)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# Response models
class PredictionResponse(BaseModel):
    predicted_class: str
    display_name: str
    confidence: float
    probabilities: Dict[str, float]
    display_probabilities: Dict[str, float]
    processing_time: float
    image_filename: str

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
        
        # Create display probabilities
        display_prob_dict = {
            config.DISPLAY_NAMES[class_name]: prob.item() 
            for class_name, prob in zip(config.CLASS_NAMES, probabilities)
        }
        
        return {
            "predicted_class": predicted_class,
            "display_name": config.DISPLAY_NAMES[predicted_class],
            "confidence": confidence,
            "probabilities": prob_dict,
            "display_probabilities": display_prob_dict,
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
            .confidence { font-size: 1.2em; font-weight: bold; color: #2E7D32; }
            .probabilities { margin: 10px 0; }
            .prob-bar { background-color: #f0f0f0; height: 25px; margin: 8px 0; position: relative; border-radius: 3px; }
            .prob-fill { height: 100%; background-color: #4CAF50; transition: width 0.5s; border-radius: 3px; }
            .prob-label { position: absolute; left: 10px; top: 4px; font-size: 0.9em; font-weight: 500; color: #333; }
            .image-container { display: flex; gap: 20px; align-items: flex-start; margin: 20px 0; }
            .uploaded-image { flex: 1; max-width: 300px; }
            .prediction-results { flex: 2; }
        </style>
    </head>
    <body>
        <h1>🐝 Hornet & Wasp Classifier</h1>
        <p>Upload an image to identify <strong>European Hornets</strong>, <strong>Asian Hornets</strong>, or <strong>Common Wasps</strong> using AI-powered classification.</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <p>Click here or drag and drop an image</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
        </div>
        
        <div id="imagePreview" style="display: none; margin: 20px 0;">
            <h3>Uploaded Image:</h3>
            <img id="previewImg" style="max-width: 400px; max-height: 300px; border: 1px solid #ddd; border-radius: 5px;" />
        </div>
        
        <div id="result" style="display: none;"></div>
        
        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                
                if (fileInput.files.length === 0) return;
                
                const file = fileInput.files[0];
                
                // Show image preview
                const imagePreview = document.getElementById('imagePreview');
                const previewImg = document.getElementById('previewImg');
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    <!--imagePreview.style.display = 'block';-->
                };
                reader.readAsDataURL(file);
                
                const formData = new FormData();
                formData.append('file', file);
                
                resultDiv.innerHTML = '<p>🔍 Analyzing image...</p>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let html = `
                            <div class="image-container">
                                <div class="uploaded-image">
                                    <h3>📸 Your Image</h3>
                                    <img src="/uploads/${result.image_filename}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />
                                </div>
                                <div class="prediction-results">
                                    <h3>🎯 Classification Results</h3>
                                    <p class="confidence">Species: <strong>${result.display_name}</strong></p>
                                    <p>Confidence: <strong>${(result.confidence * 100).toFixed(1)}%</strong></p>
                                    <div class="probabilities">
                                        <h4>All Species Probabilities:</h4>`;
                        
                        for (const [species, prob] of Object.entries(result.display_probabilities)) {
                            const isTop = prob === Math.max(...Object.values(result.display_probabilities));
                            const barColor = isTop ? '#4CAF50' : '#81C784';
                            html += `
                                <div class="prob-bar">
                                    <div class="prob-fill" style="width: ${prob * 100}%; background-color: ${barColor};"></div>
                                    <div class="prob-label">${species}: ${(prob * 100).toFixed(1)}%</div>
                                </div>`;
                        }
                        
                        html += `</div>
                                    <p><small>⏱️ Processing time: ${result.processing_time.toFixed(3)}s</small></p>
                                </div>
                            </div>`;
                        
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
        
        # Save uploaded image with unique filename
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix or '.jpg'
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = uploads_dir / unique_filename
        
        # Save image
        image.save(file_path, format='JPEG' if file_extension.lower() in ['.jpg', '.jpeg'] else 'PNG')
        
        # Preprocess and predict
        image_tensor = preprocess_image(image)
        prediction = predict_image(image_tensor)
        
        # Add filename to response
        prediction['image_filename'] = unique_filename
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get list of available classes"""
    return {
        "classes": config.CLASS_NAMES,
        "display_names": config.DISPLAY_NAMES
    }

@app.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):
    """Serve uploaded images"""
    file_path = Path("uploads") / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)