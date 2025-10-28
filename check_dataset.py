# check_dataset.py
import os
import glob

def check_dataset_structure():
    base_dir = "."
    
    # Check training data
    train_path = os.path.join(base_dir, "train_data")
    if not os.path.exists(train_path):
        print("❌ train_data directory not found!")
        return False
    
    # Check validation data
    val_path = os.path.join(base_dir, "val_data")
    if not os.path.exists(val_path):
        print("❌ val_data directory not found!")
        return False
    
    # Check class directories
    for split in ["train_data", "val_data"]:
        print(f"\n📁 Checking {split}:")
        for class_idx in range(5):
            class_dir = os.path.join(base_dir, split, f"class_{class_idx}")
            if os.path.exists(class_dir):
                # Count images
                images = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                        glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                        glob.glob(os.path.join(class_dir, "*.png"))
                print(f"   class_{class_idx}: {len(images)} images")
            else:
                print(f"   ❌ class_{class_idx}: directory not found")
    
    return True

if __name__ == "__main__":
    if check_dataset_structure():
        print("\n✅ Dataset structure is correct!")
    else:
        print("\n❌ Please fix the dataset structure.")