from PIL import Image
import os

img_dir = "data/train_images_512"
mask_dir = "data/train_masks_512"

# collect filenames without extension
images = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)}
masks = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)}

for base, img_file in images.items():
    if base not in masks:
        print("❌ Mask missing for:", img_file)
        continue

    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, masks[base])

    try:
        img = Image.open(img_path)
        mask = Image.open(mask_path)
    except Exception as e:
        print("❌ Error opening:", img_file, "->", e)
        continue

    if img.size != mask.size:
        print("⚠️ Size mismatch:", img_file, img.size, mask.size)
