# test_mobile_integration.py
import json
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

def test_onnx_model():
    """Test the ONNX model before mobile deployment"""
    print("🧪 Testing ONNX model...")
    
    if not os.path.exists("mobilenet_model.onnx"):
        print("❌ ONNX model not found")
        return False
    
    try:
        # Create inference session
        session = ort.InferenceSession("mobilenet_model.onnx", 
                                     providers=['CPUExecutionProvider'])
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        inputs = {session.get_inputs()[0].name: dummy_input}
        outputs = session.run(None, inputs)
        
        print(f"✅ ONNX model test successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Output sample: {outputs[0][0][:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX model test failed: {e}")
        return False

def test_class_names():
    """Test class names file"""
    print("\n🧪 Testing class names...")
    
    if not os.path.exists("class_names.json"):
        print("❌ Class names file not found")
        return False
    
    try:
        with open("class_names.json", 'r') as f:
            class_names = json.load(f)
        
        print(f"✅ Class names loaded: {class_names}")
        return True
        
    except Exception as e:
        print(f"❌ Class names test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Mobile Integration Test")
    print("=" * 50)
    
    onnx_ok = test_onnx_model()
    classes_ok = test_class_names()
    
    print("\n" + "=" * 50)
    if onnx_ok and classes_ok:
        print("✅ All tests passed! Ready for mobile deployment.")
    else:
        print("❌ Some tests failed. Please check the issues above.")