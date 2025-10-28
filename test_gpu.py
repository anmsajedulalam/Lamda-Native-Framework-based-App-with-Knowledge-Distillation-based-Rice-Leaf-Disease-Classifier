# test_gpu.py
import torch
import tensorflow as tf

print("🎯 GPU Test Results:")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
    print(f"PyTorch current device: {torch.cuda.current_device()}")
    print(f"PyTorch device name: {torch.cuda.get_device_name()}")

print(f"TensorFlow GPU available: {tf.test.is_gpu_available()}")
if tf.test.is_gpu_available():
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")

# Test GPU performance
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(10000, 10000).to(device)
    y = torch.randn(10000, 10000).to(device)
    
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"GPU Matrix multiplication time: {end-start:.4f} seconds")