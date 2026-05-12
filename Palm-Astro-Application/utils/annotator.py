"""
Simple Palm Line Annotator
--------------------------
Usage:
    python utils/annotator.py

Controls:
    1: Select Life Line (Red)
    2: Select Head Line (Green)
    3: Select Heart Line (Blue)
    
    Left Click: Draw points
    Right Click: Clear last point
    
    s: Save mask and move to next image
    c: Clear current line
    q: Quit
"""

import cv2
import numpy as np
import os
import sys

# Configuration
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")

# Colors (BGR)
COLORS = {
    1: (0, 0, 255),   # Life - Red
    2: (0, 255, 0),   # Head - Green
    3: (255, 0, 0)    # Heart - Blue
}

NAMES = {
    1: "Life Line (Red)",
    2: "Head Line (Green)",
    3: "Heart Line (Blue)"
}

class Annotator:
    def __init__(self):
        self.current_class = 1
        self.points = {1: [], 2: [], 3: []}
        self.drawing = False
        self.image = None
        self.display_image = None
        self.mask = None
        self.brush_size = 5
        
    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            return False
        
        # Resize for easier annotation if too large
        h, w = self.image.shape[:2]
        if h > 800:
            scale = 800 / h
            self.image = cv2.resize(self.image, (int(w*scale), int(h*scale)))
            
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.points = {1: [], 2: [], 3: []}
        self.update_display()
        return True

    def update_display(self):
        self.display_image = self.image.copy()
        
        # Draw existing lines
        for cls, pts in self.points.items():
            if len(pts) > 1:
                cv2.polylines(self.display_image, [np.array(pts)], False, COLORS[cls], 2)
                # Draw points
                for p in pts:
                    cv2.circle(self.display_image, p, 3, COLORS[cls], -1)
            elif len(pts) == 1:
                 cv2.circle(self.display_image, pts[0], 3, COLORS[cls], -1)
        
        # Overlay info
        text = f"Drawing: {NAMES[self.current_class]} | Press 1,2,3 to switch | 's' save | 'c' clear"
        cv2.putText(self.display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points[self.current_class].append((x, y))
            self.update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points[self.current_class]:
                self.points[self.current_class].pop()
                self.update_display()

    def save_mask(self, filename):
        # Create final mask
        final_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        for cls, pts in self.points.items():
            if len(pts) > 1:
                # Draw on mask with thickness
                cv2.polylines(final_mask, [np.array(pts)], False, cls, self.brush_size * 2)
        
        # Save
        save_path = os.path.join(MASKS_DIR, filename)
        cv2.imwrite(save_path, final_mask)
        print(f"Saved mask to {save_path}")

def main():
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: {IMAGES_DIR} does not exist.")
        return
        
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print("No images found in data/images")
        return
    
    annotator = Annotator()
    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", annotator.mouse_callback)
    
    print(f"Found {len(images)} images. Starting annotation...")
    print("Controls:\n 1: Life Line (Red)\n 2: Head Line (Green)\n 3: Heart Line (Blue)\n s: Save & Next\n q: Quit")
    
    for img_file in images:
        mask_path = os.path.join(MASKS_DIR, os.path.splitext(img_file)[0] + ".png")
        if os.path.exists(mask_path):
            print(f"Skipping {img_file} (already annotated)")
            continue
            
        print(f"Annotating: {img_file}")
        if not annotator.load_image(os.path.join(IMAGES_DIR, img_file)):
            continue
            
        while True:
            cv2.imshow("Annotator", annotator.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                return
            elif key == ord('s'):
                annotator.save_mask(os.path.splitext(img_file)[0] + ".png")
                break
            elif key == ord('1'):
                annotator.current_class = 1
                annotator.update_display()
            elif key == ord('2'):
                annotator.current_class = 2
                annotator.update_display()
            elif key == ord('3'):
                annotator.current_class = 3
                annotator.update_display()
            elif key == ord('c'):
                annotator.points[annotator.current_class] = []
                annotator.update_display()
                
    cv2.destroyAllWindows()
    print("Annotation complete!")

if __name__ == "__main__":
    main()
