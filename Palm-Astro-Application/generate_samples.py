"""
Simple script to generate a few sample palm images for testing
"""
import cv2
import numpy as np
import os

def create_sample_palms():
    """Create 5 sample palm images with different line patterns"""
    
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/masks", exist_ok=True)
    
    for i in range(5):
        # Create palm background
        img = np.full((512, 512, 3), (210, 180, 150), dtype=np.uint8)
        
        # Add texture
        noise = np.random.randint(-25, 25, (512, 512, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Create mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        # Vary the line positions based on sample index
        offset = i * 20
        
        # Life line
        pts = np.array([
            [250 + offset, 450],
            [200 + offset, 350],
            [150 + offset, 250],
            [120 + offset, 200]
        ], np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(img, [pts], False, (60, 50, 50), 4)
        cv2.polylines(mask, [pts], False, 1, 4)
        
        # Head line
        pts = np.array([
            [120 + offset, 200],
            [250 + offset, 220],
            [400 - offset, 250]
        ], np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(img, [pts], False, (60, 50, 50), 3)
        cv2.polylines(mask, [pts], False, 2, 3)
        
        # Heart line
        pts = np.array([
            [120 + offset, 150],
            [250 + offset, 140],
            [450 - offset, 110]
        ], np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(img, [pts], False, (60, 50, 50), 3)
        cv2.polylines(mask, [pts], False, 3, 3)
        
        # Save
        cv2.imwrite(f"data/images/sample_{i:02d}.jpg", img)
        cv2.imwrite(f"data/masks/sample_{i:02d}.png", mask)
    
    print(f"Created 5 sample palm images in data/ directory")

if __name__ == "__main__":
    create_sample_palms()
