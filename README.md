# Lamda-Native-Framework-based-App-with-Knowledge-Distillation-based-Rice-Leaf-Disease-Classifier

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

A complete mobile image classification app built with Lambda Native framework, featuring knowledge distillation and GPU-accelerated training. The app can classify images into 5 custom categories with real-time inference on mobile devices.

## 🚀 Features
Knowledge Distillation: Teacher-student training for optimized mobile performance

GPU-Accelerated Training: Fast model training using PyTorch with CUDA

Cross-Platform Mobile App: Built with Lambda Native (Scheme) for Android/iOS

Model Quantization: ONNX and TFLite formats for efficient mobile inference

Real-time Classification: Camera integration and gallery support

Customizable Classes: Train with your own 5 image categories

## 📋 Prerequisites
Hardware
NVIDIA GPU (for training)

Android/iOS device (for deployment)

8GB+ RAM recommended

Software
Ubuntu 20.04+ (development)

Python 3.9

CUDA 11.8+

Android SDK (for mobile builds)

## 🛠️ Installation & Setup
1. Clone the Repository
bash
git clone https://github.com/yourusername/mobilenet-v4-classifier.git
cd mobilenet-v4-classifier
2. Environment Setup
bash
```
Create and activate conda environment
conda env create -f environment.yml
conda activate mobilenet-v4-gpu
```
### Or use the setup script
```
chmod +x setup_gpu.sh
./setup_gpu.sh
```

#### 3. Verify GPU Setup
bash
```
python test_gpu_setup.py
```
## 📁 Dataset Preparation
```
Directory Structure
text
project/
├── train_data/
│   ├── class_0/     # Training images for class 0
│   ├── class_1/     # Training images for class 1
│   ├── class_2/     # Training images for class 2
│   ├── class_3/     # Training images for class 3
│   └── class_4/     # Training images for class 4
└── val_data/
    ├── class_0/     # Validation images for class 0
    ├── class_1/     # Validation images for class 1
    ├── class_2/     # Validation images for class 2
    ├── class_3/     # Validation images for class 3
    └── class_4/     # Validation images for class 4
Supported Image Formats
JPEG (.jpg, .jpeg)
PNG (.png)
Recommended size: 224x224 pixels
Minimum: 100 images per class for training
```
## Setup Directories

### Auto-create directory structure
```
chmod +x setup_directories.sh
./setup_directories.sh
```

### Verify structure
```
python check_dataset.py
```
### 🧠 Model Training
1. Start Training with Knowledge Distillation

bash
```
python train_mobilenet_gpu.py
```
Training Process
Teacher Model: ResNet50 (high accuracy)

Student Model: MobileNet V2 (optimized for mobile)

Knowledge Distillation: Transfers learning from teacher to student

Training Time: ~30-60 minutes on RTX 3060

Expected Accuracy: 75-85% on validation set

## 2. Monitor Training

## Monitor GPU usage (separate terminal)
```
python gpu_monitor.py
```

## Check training logs
```
tail -f training_logs.txt
```
## 3. Model Conversion
After training, models are automatically converted to:

ONNX format: mobilenet_model.onnx

TFLite format: model_quantized.tflite (quantized for mobile)

Class names: class_names.json

## 📱 Mobile App Development
## Lambda Native Setup
bash
## Install Lambda Native dependencies
```
chmod +x setup_lambda_native.sh
./setup_lambda_native.sh
```
## Verify installation
```
cd ~/lambdanative
./configure
App Structure
text
lambdanative/
├── apps/
│   └── mobilenetclassifier/
│       ├── main.scm              # Main application logic
│       ├── manifest.scm          # App configuration
│       ├── ui/
│       │   └── layout.scm        # UI definitions
│       └── plugins/
│           ├── mobilenet_model.onnx
│           ├── model_quantized.tflite
│           └── class_names.json
```
### Build Mobile App
bash
## Navigate to Lambda Native
```
cd ~/lambdanative
```
## Create app (if not already created)
```
./createapp mobilenetclassifier
```

## Copy model files
```
cp mobilenet_model.onnx apps/mobilenetclassifier/plugins/
cp class_names.json apps/mobilenetclassifier/plugins/
cp model_quantized.tflite apps/mobilenetclassifier/plugins/
```

## Build for different platforms
```
make android    # Build Android APK
make ios        # Build iOS app (requires macOS)
make desktop    # Build for desktop testing
```
## 🎯 App Features
User Interface
Camera Integration: Take photos directly in the app

Gallery Access: Choose images from device gallery

Real-time Results: Instant classification with confidence scores

Clean Design: Simple and intuitive user interface

Classification Capabilities
5 Custom Classes: Configured during training

Confidence Scores: Percentage-based predictions

Fast Inference: Optimized for mobile devices

Offline Operation: No internet connection required

## 🚀 Deployment
Android Deployment
bash
## Build release APK
```
cd ~/lambdanative
make android
```

## Install on connected device
```
adb install -r apps/mobilenetclassifier/mobilenetclassifier.apk
```

## Or distribute the APK file
```
ls apps/mobilenetclassifier/mobilenetclassifier.apk
```
iOS Deployment
bash

## Build for iOS (requires macOS and Xcode)
make ios

## Deploy via Xcode or TestFlight
Desktop Testing
bash

## Test on desktop before mobile deployment
make desktop
./apps/mobilenetclassifier/mobilenetclassifier
## 🔧 Customization
Modify Classes
Edit class_names.json or retrain with new datasets:

json
```
["cats", "dogs", "birds", "cars", "flowers"]
```
Adjust Model Parameters
Modify train_mobilenet_gpu.py:
```
python
config = {
    'training': {
        'epochs': 50,           # Training iterations
        'lr': 0.001,           # Learning rate
        'temperature': 3,      # Distillation temperature
        'alpha': 0.7           # Distillation weight
    }
}
```
## UI Customization
Edit ui/layout.scm to modify the app interface.

## 📊 Performance Optimization
For Better Accuracy
Increase training dataset size

Use data augmentation

Train for more epochs

Adjust knowledge distillation parameters

For Faster Inference
Use TFLite with quantization

Reduce model complexity

Optimize image preprocessing

Use GPU delegation on supported devices

## 🐛 Troubleshooting
Common Issues
GPU Not Detected

bash
```
nvidia-smi  # Check GPU status
python test_gpu_setup.py
```
Training Accuracy Low

Check dataset balance

Verify image quality and labels

Adjust learning rate

Mobile App Build Fails

bash
```
make clean
./configure
make android
Model Conversion Errors
```
bash
```
python convert_to_tflite.py  # Manual conversion
Logs and Debugging
```
bash
##  Check training logs
```
tail -f training_logs.txt
```
## Mobile app debugging
```
adb logcat | grep -i lambdanative
```
## Desktop testing
```
./apps/mobilenetclassifier/mobilenetclassifier --debug
```
📈 Results and Metrics
Typical Performance
Training Accuracy: 75-85%

Inference Time: <100ms on mobile

Model Size: 8-15MB (quantized)

Supported Devices: Android 8.0+, iOS 12+

Evaluation
bash
## Generate evaluation report
```
python evaluate_model.py
```
## Test on sample images
```
python test_inference.py --image test_image.jpg
```
🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

## 📝 License
This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.


Copyright (c) 2025, A. N. M. Sajedul Alam

## Funding

This project is funded through [NGI Mobifree Fund](https://nlnet.nl/mobifree), a fund established by [NLnet](https://nlnet.nl) with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) program. Learn more at the [NLnet project page](https://nlnet.nl/project/F-Droid-LambdaNative).

[<img src="https://nlnet.nl/logo/banner.png" alt="NLnet foundation logo" width="20%" />](https://nlnet.nl) &#x20;


## 🙏 Acknowledgments
Lambda Native framework developers

PyTorch and TensorFlow teams

NVIDIA for CUDA support

MobileNet and ResNet research teams
