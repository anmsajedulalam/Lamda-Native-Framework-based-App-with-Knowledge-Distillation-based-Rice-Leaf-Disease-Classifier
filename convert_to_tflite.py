# convert_to_tflite.py
import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
import os

def convert_onnx_to_tflite():
    print("🔄 Converting ONNX to TFLite...")
    
    try:
        # Method 1: Direct conversion using tf.compat.v1
        import onnx_tf
        import onnx
        
        # Load ONNX model
        onnx_model = onnx.load("mobilenet_model.onnx")
        
        # Convert to TensorFlow
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        
        # Export as saved model
        tf_rep.export_graph("saved_model")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open('model_quantized.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print("✅ Model converted to TFLite successfully!")
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        print("🔄 Trying alternative method...")
        convert_using_alternative_method()

def convert_using_alternative_method():
    """Alternative method using ONNX Runtime for inference and manual conversion"""
    try:
        # Create a simple representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                yield [data]

        # Create a simple model wrapper for TFLite conversion
        class SimpleModel(tf.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                # We'll use ONNX Runtime for inference instead
                self.session = ort.InferenceSession("mobilenet_model.onnx", 
                                                   providers=['CPUExecutionProvider'])
            
            @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32)])
            def predict(self, x):
                # Convert to ONNX format (NCHW to NHWC)
                x_np = x.numpy()
                x_onnx = np.transpose(x_np, (0, 3, 1, 2))  # NHWC to NCHW
                
                # Run ONNX inference
                inputs = {self.session.get_inputs()[0].name: x_onnx.astype(np.float32)}
                outputs = self.session.run(None, inputs)
                
                return tf.constant(outputs[0])

        # Create and save the model
        model = SimpleModel()
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [model.predict.get_concrete_function()])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open('model_quantized.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print("✅ Model converted to TFLite using alternative method!")
        
    except Exception as e:
        print(f"❌ Alternative method also failed: {e}")
        print("📱 Proceeding with ONNX model only for mobile deployment.")

if __name__ == "__main__":
    convert_onnx_to_tflite()