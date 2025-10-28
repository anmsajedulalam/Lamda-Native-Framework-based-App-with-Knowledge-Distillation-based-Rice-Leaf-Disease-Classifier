#!/bin/bash
# setup_gpu.sh

echo "🎯 Setting up GPU-optimized MobileNet V4 Environment..."

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

# Check CUDA version
echo "🔍 Checking CUDA version..."
nvidia-smi

# Create conda environment
echo "📦 Creating conda environment..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "✅ Environment created successfully!"
else
    echo "❌ Environment creation failed. Trying alternative approach..."
    conda create -n mobilenet-v4-gpu python=3.9
    conda activate mobilenet-v4-gpu
    conda install -y -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio cudatoolkit=11.8
    pip install tensorflow-gpu onnx onnxruntime-gpu tf2onnx torchdistill opencv-python pillow scikit-learn
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p train_data/class_{0..4}
mkdir -p val_data/class_{0..4}
mkdir -p mobilenet_app/plugins
mkdir -p mobilenet_app/ui

# Test GPU availability
echo "🧪 Testing GPU availability..."
conda activate mobilenet-v4-gpu
python -c "
import torch
print(f'🎯 PyTorch version: {torch.__version__}')
print(f'🎯 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎯 CUDA version: {torch.version.cuda}')
    print(f'🎯 GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'🎯 GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ No GPU detected')
"

echo ""
echo "✅ Setup completed!"
echo ""
echo "📝 Next steps:"
echo "1. Add your training images to train_data/class_0 to class_4"
echo "2. Add your validation images to val_data/class_0 to class_4" 
echo "3. Run: python train_mobilenet_gpu.py"
echo "4. Monitor GPU usage: python gpu_monitor.py (in another terminal)"
echo "5. Copy generated files to mobilenet_app/plugins/"
echo "6. Build your Lambda Native app"