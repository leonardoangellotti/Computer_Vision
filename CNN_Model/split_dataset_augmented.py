import os
import shutil
import random
from PIL import Image

# Path to your current train folder (contains ONLY original images at the beginning)
train_dir = "/Users/leonardoangellotti/Desktop/universita/Comp_Vision/CNN/data/train"   

# New validation folder
val_dir = "/Users/leonardoangellotti/Desktop/universita/Comp_Vision/CNN/data/val"

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

    # 2) DATA AUGMENTATION ONLY ON REMAINING TRAIN ORIGINALS
    aug_count = 0

    for img_name in train_originals:
        img_path = os.path.join(class_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_name}: {e}")
            continue

        base_name, ext = os.path.splitext(img_name)

        # 1) Horizontal flip
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_name = f"{base_name}_flip{ext}"
        flipped.save(os.path.join(class_path, flip_name))
        aug_count += 1

        # 2) Random rotation (e.g. -15 to +15 degrees)
        angle = random.uniform(-15, 15)
        rotated = img.rotate(angle, expand=True)
        rot_name = f"{base_name}_rot{ext}"
        rotated.save(os.path.join(class_path, rot_name))
        aug_count += 1

        # 3) Center crop (e.g. 90% of original size)
        w, h = img.size
        crop_scale = 0.9
        new_w, new_h = int(w * crop_scale), int(h * crop_scale)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        cropped = img.crop((left, top, right, bottom))
        crop_name = f"{base_name}_crop{ext}"
        cropped.save(os.path.join(class_path, crop_name))
        aug_count += 1

    print(f"Generated {aug_count} augmented images for TRAIN in class {class_name}.")

print("\nSplit completed")