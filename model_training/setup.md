# 🛠️ Setup Instructions for Model Training

Follow these steps to set up the environment for training hornet/wasp classification models.

## 📋 Prerequisites

- Python 3.8+ (recommended: Python 3.10)
- Git (for cloning the repository)
- At least 4GB RAM (8GB recommended)
- GPU optional but recommended for faster training

## 🚀 Quick Setup

### 1. Clone Repository & Navigate
```bash
git clone <repository-url>
cd cloud-for-ai/model_training
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Launch Jupyter Lab
```bash
jupyter lab
```

### 5. Open Training Notebook
- Navigate to `hornet_wasp_classifier_with_yolo.ipynb`
- Run all cells to start training

## 🔧 Alternative Installation Methods

### Using Conda
```bash
# Create conda environment
conda create -n hornet-wasp python=3.10
conda activate hornet-wasp

# Install PyTorch with CUDA (if you have a GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining packages
pip install -r requirements.txt
```

### GPU Support (Optional)
For faster training with NVIDIA GPUs:
```bash
# Check if CUDA is available
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📦 Package Descriptions

| Package | Purpose |
|---------|---------|
| **torch** | PyTorch deep learning framework |
| **torchvision** | Computer vision utilities and models |
| **numpy** | Numerical computing |
| **scikit-learn** | Machine learning metrics and utilities |
| **pandas** | Data manipulation and analysis |
| **matplotlib** | Plotting and visualization |
| **seaborn** | Statistical data visualization |
| **pillow** | Image processing |
| **opencv-python** | Computer vision operations |
| **jupyter** | Interactive notebook environment |

## 🐛 Troubleshooting

### Common Issues

**1. PyTorch Installation Problems**
```bash
# If torch installation fails, try:
pip install torch torchvision --no-cache-dir
```

**2. Out of Memory Errors**
- Reduce batch size in the notebook (change `batch_size = 32` to `batch_size = 16`)
- Close other applications to free up RAM

**3. CUDA Not Available**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
# If False, training will use CPU (slower but still works)
```

**4. Jupyter Kernel Not Found**
```bash
# Register the virtual environment as a Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=hornet-wasp --display-name="Hornet-Wasp ML"
```

### Performance Tips

- **GPU Training**: 10-20x faster than CPU
- **RAM Usage**: Close browser tabs and other apps during training
- **Batch Size**: Adjust based on available memory
- **Number of Workers**: Set `num_workers=0` if you get multiprocessing errors

## 📊 Expected Training Time

| Hardware | Training Time (20 epochs) |
|----------|---------------------------|
| CPU only | ~15-30 minutes |
| GPU (GTX 1060+) | ~2-5 minutes |
| GPU (RTX 3070+) | ~1-2 minutes |

## ✅ Verification Checklist

Before starting training, verify:
- [ ] Virtual environment is activated
- [ ] All packages installed successfully
- [ ] Jupyter Lab launches without errors
- [ ] PyTorch detects GPU (if available)
- [ ] Dataset directory exists at `../dataset/`
- [ ] Can import all required libraries in notebook

## 💡 Tips for Success

1. **Start Small**: Train for a few epochs first to test everything works
2. **Monitor Resources**: Use `htop` (Linux/Mac) or Task Manager (Windows)
3. **Save Progress**: Notebooks auto-save, but manually save important results
4. **GPU Memory**: If you get CUDA out of memory, restart the notebook kernel

## 🆘 Getting Help

If you encounter issues:
1. Check the error message carefully
2. Ensure all requirements are installed correctly
3. Try restarting the Jupyter kernel
4. Create a new virtual environment if problems persist

---
**Happy Training!** 🐝🤖