# 🚀 Hornet & Wasp Classification App

A production-ready FastAPI web application for classifying hornets and wasps using the trained ResNet50 model.

## 🎯 Features

- ✅ **FastAPI Backend** - Modern, fast, and automatic API documentation
- ✅ **Web Interface** - Simple drag-and-drop image upload
- ✅ **Real-time Classification** - Instant predictions with confidence scores
- ✅ **Docker Deployment** - Ready for production with multi-stage builds
- ✅ **Health Monitoring** - Built-in health checks and logging
- ✅ **Production Ready** - Nginx reverse proxy, rate limiting, security

## 🏗️ Architecture

```
app/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── Dockerfile         # Multi-stage Docker build
├── docker-compose.yml # Container orchestration
├── nginx.conf         # Reverse proxy configuration
├── run.sh            # Quick start script
├── test_api.py       # API testing suite
└── README.md         # This file
```

## 🚀 Quick Start

### Option 1: One-Command Deploy
```bash
cd app
./run.sh
```

### Option 2: Manual Docker
```bash
cd app
docker-compose up -d
```

### Option 3: Development Mode
```bash
cd app
pip install -r requirements.txt
python app.py
```

## 🌐 Access Points

After starting the application:

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Redoc Documentation**: http://localhost:8000/redoc

## 🧪 Testing

Test all API endpoints:
```bash
python test_api.py
```

## 📡 API Endpoints

### `GET /`
Main web interface for image uploads

### `POST /predict`
**Upload image for classification**
- **Input**: Image file (JPEG, PNG, etc.)
- **Output**: JSON with prediction results

```json
{
  "predicted_class": "Vespa_crabro",
  "confidence": 0.94,
  "probabilities": {
    "Vespa_crabro": 0.94,
    "Vespa_velutina": 0.04,
    "Vespula_sp": 0.02
  },
  "processing_time": 0.123
}
```

### `GET /health`
**System health check**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### `GET /classes`
**Available classification classes**
```json
{
  "classes": ["Vespa_crabro", "Vespa_velutina", "Vespula_sp"]
}
```

## 🐳 Docker Deployment

### Development
```bash
docker-compose up
```

### Production (with Nginx)
```bash
docker-compose --profile production up -d
```

### Custom Configuration
```bash
# Build with custom model path
docker build --build-arg MODEL_PATH=./custom_model.pth .

# Run with environment variables
docker run -e MODEL_PATH=/app/models/custom.pth hornet-wasp-app
```

## ⚙️ Configuration

### Environment Variables
- `MODEL_PATH`: Path to trained model file
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `WORKERS`: Number of worker processes

### Model Requirements
- Uses `best_model_resnet50_cropped.pth` from model training
- ResNet50 architecture with 3-class output
- Input: 224x224 RGB images
- Preprocessing: ImageNet normalization

## 🔒 Security Features

- **Rate Limiting**: 10 requests/minute per IP (production)
- **File Validation**: Only accepts image files
- **Non-root User**: Docker runs as non-privileged user
- **Health Checks**: Automatic container health monitoring
- **CORS**: Configurable cross-origin requests

## 📊 Performance

### Expected Performance
- **Inference Time**: ~100-300ms per image
- **Memory Usage**: ~500MB-1GB RAM
- **Throughput**: 10-100 requests/second
- **Model Size**: ~100MB

### Hardware Requirements
- **Minimum**: 2GB RAM, 1 CPU core
- **Recommended**: 4GB RAM, 2+ CPU cores
- **GPU**: Optional, automatically detected

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Adding Features
1. **New Endpoints**: Add to `app.py`
2. **Frontend**: Modify HTML in root endpoint
3. **Models**: Update `ResNetTransfer` class
4. **Tests**: Add to `test_api.py`

### Code Structure
```python
# Main components
- Config: Application configuration
- ResNetTransfer: Model architecture
- FastAPI routes: API endpoints
- Preprocessing: Image preparation
- Prediction: Model inference
```

## 🔧 Troubleshooting

### Common Issues

**1. Model Not Found**
```bash
# Ensure model file exists
ls -la ../model_training/best_model_resnet50_cropped.pth

# Check Docker volume mount
docker-compose logs
```

**2. Out of Memory**
```bash
# Reduce batch size or use smaller images
# Monitor memory usage
docker stats
```

**3. Permission Denied**
```bash
# Make scripts executable
chmod +x run.sh test_api.py
```

**4. Port Already in Use**
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### Debug Commands
```bash
# View application logs
docker-compose logs -f

# Check container health
docker ps
docker inspect hornet-wasp-api

# Test endpoints manually
curl http://localhost:8000/health
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

## 📈 Monitoring

### Health Checks
- **Endpoint**: `/health`
- **Docker**: Built-in health check every 30s
- **Nginx**: Health check bypass

### Logging
- **Level**: INFO (configurable)
- **Format**: Structured JSON logs
- **Output**: stdout (captured by Docker)

### Metrics (Future)
- Request latency
- Prediction accuracy
- Error rates
- Resource usage

## 🚀 Deployment Options

### Cloud Platforms
- **AWS**: ECS, Fargate, or EC2
- **Google Cloud**: Cloud Run or GKE
- **Azure**: Container Instances or AKS
- **Heroku**: Container deployment

### Production Checklist
- [ ] Use production WSGI server
- [ ] Configure logging and monitoring  
- [ ] Set up SSL/HTTPS
- [ ] Configure rate limiting
- [ ] Add authentication (if needed)
- [ ] Set up backup/recovery
- [ ] Performance testing

---

**Status**: ✅ Production Ready - Deploy with confidence!