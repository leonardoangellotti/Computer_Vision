import os
import shutil
import random
from PIL import Image

# Path to your current train folder (contains ONLY original images at the beginning)
train_dir = "/Users/leonardoangellotti/Desktop/universita/Comp_Vision/CNN/original_data/train"   

# New validation folder
val_dir = "/Users/leonardoangellotti/Desktop/universita/Comp_Vision/CNN/original_data/val"

# Create validation folder
os.makedirs(val_dir, exist_ok=True)

# Helper: check if a file name looks like an augmented one
AUG_TAGS = ["_flip", "_rot", "_crop"]

def is_augmented(filename: str) -> bool:
    base, _ = os.path.splitext(filename)
    return any(tag in base for tag in AUG_TAGS)

# 1) SPLIT ORIGINAL IMAGES INTO TRAIN / VAL
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"\nClass: {class_name}")

    # List **original** images only (ignore already-augmented ones)
    originals = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        and not is_augmented(f)
    ]

    if not originals:
        print("No original images found, skipping class.")
        continue

    random.shuffle(originals)

    n_total = len(originals)
    n_val = int(n_total * 0.15)  # 15% to validation

    val_images = originals[:n_val]
    train_originals = originals[n_val:]

    # Create corresponding class folder inside val/
    val_class_path = os.path.join(val_dir, class_name)
    os.makedirs(val_class_path, exist_ok=True)

    # Move validation images
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_path, img)
        shutil.move(src, dst)

    print(f"Moved {len(val_images)} original images to validation set (out of {n_total}).")

print("\nSplit completed")
