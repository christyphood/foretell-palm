# debug_transform_check.py
import cv2
import os
import traceback

# change these to your folders / lists
img_dir = "data/train_images_512"
mask_dir = "data/train_masks_512"

# build lists using matching stems or however your dataset pairs are matched
images = {os.path.splitext(f)[0]: os.path.join(img_dir, f) for f in os.listdir(img_dir)}
masks  = {os.path.splitext(f)[0]: os.path.join(mask_dir, f) for f in os.listdir(mask_dir)}

# load your albumentations transform exactly as in train.py
from albumentations import Compose, HorizontalFlip, RandomCrop, Resize  # example
# replace below with your actual transform pipeline
transform = Compose([
    Resize(512, 512),
    HorizontalFlip(p=0.5),
])

for stem, img_path in images.items():
    mask_path = masks.get(stem)
    if mask_path is None:
        print("Missing mask for:", img_path)
        continue

    img = cv2.imread(img_path)
    img_rgb = img[:, :, ::-1] if img is not None else None
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    print("Checking:", os.path.basename(img_path))
    print("  img shape:", None if img_rgb is None else img_rgb.shape)
    print("  mask shape:", None if mask is None else mask.shape)

    if img_rgb is None or mask is None:
        print("  -> read error")
        continue

    # Convert mask to single channel if needed (same rules as above)
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[..., :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    try:
        _ = transform(image=img_rgb, mask=mask)  # this line can raise
    except Exception as e:
        print("  -> TRANSFORM ERROR for", os.path.basename(img_path))
        traceback.print_exc()
        # stop or continue depending on whether you want all failures
        # break
    else:
        print("  -> ok")
