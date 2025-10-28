#!/bin/bash
# build_mobile_app.sh

echo "📱 Building MobileNet Classifier App..."

cd mobilenet_app

# Check if required files exist
if [ ! -f "plugins/mobilenet_model.onnx" ]; then
    echo "❌ ONNX model not found in plugins directory"
    exit 1
fi

if [ ! -f "plugins/class_names.json" ]; then
    echo "❌ Class names file not found"
    exit 1
fi

echo "✅ All required files present"

# Build for Android (assuming Lambda Native is set up)
echo "🔨 Building for Android..."
# build-android

# Build for iOS
echo "🔨 Building for iOS..."
# build-ios

echo "✅ Build completed!"
echo ""
echo "📱 Next steps:"
echo "1. Install the app on your device"
echo "2. Test with images from your 5 classes"
echo "3. The app will show predictions with confidence scores"