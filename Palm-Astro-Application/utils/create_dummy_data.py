import cv2
import numpy as np
import os
import random

def create_dummy_data(output_dir="data", num_samples=50):
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print(f"Generating {num_samples} dummy samples in {output_dir}...")
    
    for i in range(num_samples):
        # Create a "palm" background (skin tone-ish)
        img_size = 512
        img = np.full((img_size, img_size, 3), (180, 200, 230), dtype=np.uint8) # BGR for skin
        
        # Add some noise/texture
        noise = np.random.randint(-20, 20, (img_size, img_size, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Initialize masks (3 channels: Life, Head, Heart)
        # 0: Background, 1: Life, 2: Head, 3: Heart
        # We'll save as class indices for simplicity in PNG, or separate channels.
        # Let's save as a single channel image where pixel val = class id.
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Draw Life Line (Curved, bottom-left to center)
        # Simple bezier-like curve approximation using polylines
        pts_life = np.array([[250, 450], [200, 350], [150, 250], [100, 200]], np.int32)
        pts_life = pts_life.reshape((-1, 1, 2))
        # Randomize slightly
        pts_life += np.random.randint(-20, 20, pts_life.shape)
        cv2.polylines(img, [pts_life], False, (50, 40, 40), 3, lineType=cv2.LINE_AA)
        cv2.polylines(mask, [pts_life], False, 1, 3, lineType=cv2.LINE_AA)
        
        # Draw Head Line (Center to Right)
        pts_head = np.array([[100, 200], [250, 220], [400, 250]], np.int32)
        pts_head = pts_head.reshape((-1, 1, 2))
        pts_head += np.random.randint(-20, 20, pts_head.shape)
        cv2.polylines(img, [pts_head], False, (50, 40, 40), 3, lineType=cv2.LINE_AA)
        cv2.polylines(mask, [pts_head], False, 2, 3, lineType=cv2.LINE_AA)
        
        # Draw Heart Line (Top, Left to Right)
        pts_heart = np.array([[100, 150], [250, 140], [450, 100]], np.int32)
        pts_heart = pts_heart.reshape((-1, 1, 2))
        pts_heart += np.random.randint(-20, 20, pts_heart.shape)
        cv2.polylines(img, [pts_heart], False, (50, 40, 40), 3, lineType=cv2.LINE_AA)
        cv2.polylines(mask, [pts_heart], False, 3, 3, lineType=cv2.LINE_AA)
        
        # Save
        cv2.imwrite(os.path.join(images_dir, f"palm_{i:03d}.jpg"), img)
        cv2.imwrite(os.path.join(masks_dir, f"palm_{i:03d}.png"), mask)

    print("Done.")

if __name__ == "__main__":
    create_dummy_data()
