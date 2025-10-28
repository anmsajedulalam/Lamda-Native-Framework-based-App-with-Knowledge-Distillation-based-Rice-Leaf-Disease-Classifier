# create_simple_tflite.py
import tensorflow as tf
import numpy as np

def create_simple_tflite_model():
    """Create a simple TFLite model as fallback"""
    print("🔄 Creating simple TFLite model...")
    
    # Create a simple Keras model with same input/output dimensions
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open('model_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ Simple TFLite model created!")
    print("⚠️ Note: This is a placeholder. Use the ONNX model for actual inference.")

if __name__ == "__main__":
    create_simple_tflite_model()