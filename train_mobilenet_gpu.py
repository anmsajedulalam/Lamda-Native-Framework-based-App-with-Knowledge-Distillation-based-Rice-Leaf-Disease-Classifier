# train_mobilenet_gpu.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import numpy as np
import json
import time
from tqdm import tqdm

# Check GPU availability
def setup_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f"🎯 Using GPU: {gpu_name}")
        print(f"🎯 Available GPUs: {gpu_count}")
        
        # Set CUDA optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        return device
    else:
        print("❌ No GPU available, using CPU")
        return torch.device('cpu')

# Custom Dataset for 5 classes
class FiveClassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"📁 Loaded {len(self.images)} images from {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            dummy_image = Image.new('RGB', (224, 224), color='red')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

# Enhanced MobileNet Student Model with GPU optimizations
class MobileNetStudent(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetStudent, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# Enhanced Teacher Model (ResNet50 for better distillation)
class TeacherModel(nn.Module):
    def __init__(self, num_classes=5):
        super(TeacherModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# Knowledge Distillation Loss with GPU optimization
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        with autocast():
            soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=1)
            soft_student = torch.log_softmax(student_logits / self.temperature, dim=1)
            
            distillation_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
            student_loss = self.ce_loss(student_logits, targets)
            
            return self.alpha * distillation_loss + (1 - self.alpha) * student_loss

def train_with_distillation():
    device = setup_gpu()
    
    # Check if data directories exist
    if not os.path.exists('./train_data'):
        print("📁 Creating sample directory structure...")
        os.makedirs('./train_data', exist_ok=True)
        os.makedirs('./val_data', exist_ok=True)
        for i in range(5):
            os.makedirs(f'./train_data/class_{i}', exist_ok=True)
            os.makedirs(f'./val_data/class_{i}', exist_ok=True)
        print("✅ Please add your images to train_data/class_0 to class_4 and val_data/class_0 to class_4")
        return None, None
    
    # Enhanced Configuration for GPU
    config = {
        'dataset': {
            'train_root': './train_data',
            'val_root': './val_data',
            'batch_size': 32 if torch.cuda.is_available() else 16,  # Larger batches for GPU
            'num_workers': 4 if torch.cuda.is_available() else 2,   # More workers for GPU
            'pin_memory': True if torch.cuda.is_available() else False
        },
        'training': {
            'epochs': 50,
            'lr': 0.001,
            'temperature': 3,
            'alpha': 0.7,
            'use_amp': True,  # Automatic Mixed Precision
            'accumulation_steps': 1
        },
        'models': {
            'num_classes': 5
        }
    }
    
    print("🚀 Training Configuration:")
    print(f"   Batch Size: {config['dataset']['batch_size']}")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Mixed Precision: {config['training']['use_amp']}")
    
    # Data transforms with GPU optimization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    try:
        train_dataset = FiveClassDataset(config['dataset']['train_root'], transform=train_transform)
        val_dataset = FiveClassDataset(config['dataset']['val_root'], transform=val_transform)
        
        if len(train_dataset) == 0:
            print("❌ No training images found! Please add images to train_data directories.")
            return None, None
            
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return None, None
    
    # DataLoaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=True if config['dataset']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=True if config['dataset']['num_workers'] > 0 else False
    )
    
    # Initialize models with DataParallel for multi-GPU
    student_model = MobileNetStudent(num_classes=config['models']['num_classes'])
    teacher_model = TeacherModel(num_classes=config['models']['num_classes'])
    
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
        student_model = nn.DataParallel(student_model)
        teacher_model = nn.DataParallel(teacher_model)
    
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    
    # Optimizers with weight decay
    student_optimizer = optim.AdamW(student_model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    teacher_optimizer = optim.AdamW(teacher_model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    
    # Learning rate schedulers
    student_scheduler = optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=config['training']['epochs'])
    teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=config['training']['epochs'])
    
    # Loss functions
    distillation_loss = DistillationLoss(
        temperature=config['training']['temperature'],
        alpha=config['training']['alpha']
    )
    ce_loss = nn.CrossEntropyLoss()
    
    # Mixed Precision Scaler
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    best_accuracy = 0.0
    training_history = []
    
    print("🎯 Starting Training with Knowledge Distillation...")
    
    for epoch in range(config['training']['epochs']):
        start_time = time.time()
        
        # Training phase
        student_model.train()
        teacher_model.train()
        
        train_student_loss = 0.0
        train_teacher_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # Teacher training with mixed precision
            teacher_optimizer.zero_grad()
            with autocast(enabled=config['training']['use_amp']):
                teacher_outputs = teacher_model(data)
                teacher_loss = ce_loss(teacher_outputs, targets)
            
            scaler.scale(teacher_loss).backward()
            scaler.step(teacher_optimizer)
            
            # Student training with distillation and mixed precision
            student_optimizer.zero_grad()
            with autocast(enabled=config['training']['use_amp']):
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                
                student_outputs = student_model(data)
                student_loss = distillation_loss(student_outputs, teacher_outputs, targets)
            
            scaler.scale(student_loss).backward()
            scaler.step(student_optimizer)
            scaler.update()
            
            train_student_loss += student_loss.item()
            train_teacher_loss += teacher_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Student Loss': f'{student_loss.item():.4f}',
                'Teacher Loss': f'{teacher_loss.item():.4f}'
            })
        
        # Update learning rates
        student_scheduler.step()
        teacher_scheduler.step()
        
        # Validation phase
        student_model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                with autocast(enabled=config['training']['use_amp']):
                    outputs = student_model(data)
                    loss = ce_loss(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_student_loss = train_student_loss / len(train_loader)
        avg_teacher_loss = train_teacher_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        epoch_time = time.time() - start_time
        
        print(f'✅ Epoch {epoch+1}/{config["training"]["epochs"]} - '
              f'Time: {epoch_time:.2f}s - '
              f'Student Loss: {avg_student_loss:.4f} - '
              f'Teacher Loss: {avg_teacher_loss:.4f} - '
              f'Val Loss: {avg_val_loss:.4f} - '
              f'Accuracy: {accuracy:.2f}%')
        
        training_history.append({
            'epoch': epoch + 1,
            'student_loss': avg_student_loss,
            'teacher_loss': avg_teacher_loss,
            'val_loss': avg_val_loss,
            'accuracy': accuracy
        })
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            
            # Remove DataParallel wrapper if present
            if isinstance(student_model, nn.DataParallel):
                model_to_save = student_model.module
            else:
                model_to_save = student_model
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': student_optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': avg_student_loss,
            }, 'best_mobilenet_student.pth')
            
            print(f'💾 New best model saved with accuracy: {accuracy:.2f}%')
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f'🎉 Training completed! Best accuracy: {best_accuracy:.2f}%')
    
    # Return the original model (without DataParallel)
    if isinstance(student_model, nn.DataParallel):
        student_model = student_model.module
    
    return student_model, train_dataset.classes

def convert_for_lambda_native(model, class_names):
    """Convert model to format suitable for Lambda Native with GPU optimizations"""
    
    device = setup_gpu()
    model.to(device)
    model.eval()
    
    # Save class names
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    # Convert to ONNX with GPU
    print("🔄 Converting model to ONNX...")
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            "mobilenet_model.onnx",
            export_params=True,
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            verbose=False
        )
        print("✅ Model converted to ONNX successfully!")
        
        # Verify ONNX model
        import onnx
        onnx_model = onnx.load("mobilenet_model.onnx")
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verified!")
        
    except Exception as e:
        print(f"❌ Error converting to ONNX: {e}")
        return
    
    # Convert to TensorFlow Lite with quantization (optional)
    try:
        print("🔄 Converting to TensorFlow Lite...")
        import onnxruntime as ort
        import tensorflow as tf
        
        # Create ONNX Runtime session with GPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession("mobilenet_model.onnx", providers=providers)
        
        # Convert using tf2onnx (simplified approach)
        import tf2onnx
        
        # For mobile deployment, we'll create a simple TFLite model
        # This is a simplified conversion - you might need to adjust based on your exact needs
        converter = tf.lite.TFLiteConverter.from_saved_model("mobilenet_model.onnx")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open('model_quantized.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print("✅ Model converted to TFLite successfully!")
        
    except ImportError as e:
        print(f"⚠️ TF conversion dependencies not available: {e}")
        print("📱 Using ONNX model for mobile deployment.")
    except Exception as e:
        print(f"⚠️ Error converting to TFLite: {e}")
        print("📱 Using ONNX model for mobile deployment.")

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        print("\n💾 GPU Memory Info:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

if __name__ == "__main__":
    print("🚀 Starting GPU-optimized MobileNet Training...")
    
    # Check GPU info
    check_gpu_memory()
    
    # Train model
    student_model, class_names = train_with_distillation()
    
    if student_model is not None:
        print("🔄 Converting model for Lambda Native deployment...")
        convert_for_lambda_native(student_model, class_names)
        print("🎉 Training and conversion completed!")
        
        # Final GPU memory check
        check_gpu_memory()
    else:
        print("❌ Training failed. Please check your data directories.")