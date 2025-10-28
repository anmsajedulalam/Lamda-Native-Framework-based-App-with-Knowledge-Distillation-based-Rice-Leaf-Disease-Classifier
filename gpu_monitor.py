# gpu_monitor.py
import torch
import time
import psutil
import GPUtil

def monitor_gpu():
    if torch.cuda.is_available():
        print("🎯 GPU Monitoring Started...")
        while True:
            # PyTorch GPU info
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            
            # System GPU info
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append(f"{gpu.name}: {gpu.load*100:.1f}% load, {gpu.memoryUtil*100:.1f}% memory")
            
            print(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Utilization: {gpu_util}%")
            for info in gpu_info:
                print(f"   {info}")
            
            time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    monitor_gpu()