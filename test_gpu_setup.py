# test_gpu_setup.py
import torch
import subprocess
import sys

def check_system_gpu():
    print("🔍 Checking System GPU Setup...")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA drivers are installed")
            # Extract GPU info from nvidia-smi
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'Driver' in line:
                    print(f"   {line.strip()}")
                if 'CUDA Version' in line:
                    print(f"   {line.strip()}")
        else:
            print("❌ NVIDIA drivers not working properly")
    except FileNotFoundError:
        print("❌ nvidia-smi not found. Please install NVIDIA drivers.")
        return False
    
    return True

def check_pytorch_gpu():
    print("\n🔍 Checking PyTorch GPU Support...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Test GPU tensor operations
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)
        print("✅ GPU tensor operations working correctly")
        return True
    else:
        print("❌ PyTorch cannot access GPU")
        return False

def check_tensorflow_gpu():
    print("\n🔍 Checking TensorFlow GPU Support...")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU devices available: {len(tf.config.list_physical_devices('GPU'))}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                print(f"✅ TensorFlow can access: {gpu}")
            return True
        else:
            print("❌ TensorFlow cannot access GPU")
            return False
    except ImportError:
        print("❌ TensorFlow not installed")
        return False

if __name__ == "__main__":
    print("🎯 Comprehensive GPU Setup Check")
    print("=" * 50)
    
    system_ok = check_system_gpu()
    pytorch_ok = check_pytorch_gpu()
    tensorflow_ok = check_tensorflow_gpu()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"System GPU: {'✅' if system_ok else '❌'}")
    print(f"PyTorch GPU: {'✅' if pytorch_ok else '❌'}")
    print(f"TensorFlow GPU: {'✅' if tensorflow_ok else '❌'}")
    
    if pytorch_ok:
        print("\n🎉 Your GPU setup is ready for training!")
    else:
        print("\n❌ Please check your GPU setup before proceeding.")