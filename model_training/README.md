# 🧠 Model Training

This directory contains all machine learning training code, notebooks, and results for the hornet/wasp classification project.

## 📂 Contents

- `hornet_wasp_classifier_with_yolo.ipynb` - Training notebook with YOLO integration
- `best_model_resnet50_cropped.pth` - Best performing trained model
- `model_comparison_results_with_yolo.csv` - Performance metrics comparison

## 🎯 Training Overview

### Models Implemented
1. **Custom CNN** - Built from scratch with 5 convolutional layers
2. **ResNet50 Transfer Learning** - Pre-trained on ImageNet, fine-tuned
3. **Vision Transformer (ViT)** - Transformer architecture for images
4. **YOLO-Enhanced Classification** - Uses bounding boxes for cropped regions

### Key Innovations
- **YOLO Integration**: Crops images to hornet/wasp regions before classification
- **Transfer Learning**: Different learning rates for backbone vs classifier
- **Comprehensive Evaluation**: Confusion matrices, per-class metrics
- **Production Ready**: Model saving and inference pipeline

## 🏆 Results Summary

**Best Model**: ResNet50 on YOLO-cropped regions
- **Accuracy**: 95.2%
- **Precision**: 0.951
- **Recall**: 0.952  
- **F1-Score**: 0.951
- **Training Time**: 120 seconds

### Performance Comparison
| Approach | Advantage |
|----------|-----------|
| **YOLO + Classification** | ✅ Best accuracy, focuses on relevant regions |
| **Full Image Classification** | ⚡ Simpler pipeline, faster preprocessing |

## 🚀 How to Use

### 1. Training from Scratch
```bash
jupyter lab hornet_wasp_classifier_with_yolo.ipynb
```

### 2. Loading Trained Model
```python
import torch
from torchvision import models

# Load the best model
model = ResNetTransfer(num_classes=3)
model.load_state_dict(torch.load('best_model_resnet50_cropped.pth'))
model.eval()
```

### 3. Making Predictions
```python
# For new images with YOLO bounding boxes
prediction = predict_with_confidence(model, image_path, transform, class_names)
print(f"Predicted: {prediction['predicted_class']} ({prediction['confidence']:.1%})")
```

## 📊 Training Configuration

- **Epochs**: 20 (for comparison speed)
- **Batch Size**: 32
- **Learning Rate**: 0.001 (classifier), 0.0001 (backbone)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler
- **Augmentation**: Flip, rotation, color jitter, affine transforms

## 🔍 Dataset Details

- **Training**: 3,088 images with YOLO bounding boxes
- **Validation**: 297 images with YOLO bounding boxes  
- **Classes**: 
  - Vespa_crabro: 954/100 (train/val)
  - Vespa_velutina: 1,102/100 (train/val)
  - Vespula_sp: 1,032/97 (train/val)

## 📈 Key Findings

1. **YOLO cropping improves accuracy** by ~6% over full images
2. **ResNet50 outperforms** custom CNN and ViT on this dataset
3. **Transfer learning is crucial** - pre-trained features generalize well
4. **Balanced dataset** leads to consistent per-class performance

## 🔧 Technical Notes

- **Image preprocessing**: 224x224 resolution, ImageNet normalization
- **YOLO mapping**: Class 0→Vespa_velutina, 1→Vespa_crabro, 2→Vespula_sp
- **Model architecture**: ResNet50 backbone + custom classifier head
- **Hardware**: Optimized for both CPU and GPU training

## 📝 Future Improvements

- [ ] Ensemble methods combining multiple models
- [ ] Data augmentation with more diverse backgrounds
- [ ] Real-time object detection + classification pipeline
- [ ] Mobile-optimized model variants (MobileNet, EfficientNet)