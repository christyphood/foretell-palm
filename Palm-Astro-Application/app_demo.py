"""
Simplified Gradio app that doesn't require trained model
Demo-only version for testing the UI
"""
import gradio as gr
import numpy as np
from PIL import Image
import cv2

def create_demo_segmentation(image_np):
    """Create a simple demo segmentation without model"""
    h, w = image_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw fake palm lines for demo
    # Life line (Red - class 1)
    cv2.line(mask, (w//4, h//2), (w//4, 3*h//4), 1, max(3, h//100))
    
    # Head line (Green - class 2)
    cv2.line(mask, (w//4, h//2), (3*w//4, h//2), 2, max(3, h//100))
    
    # Heart line (Blue - class 3)
    cv2.line(mask, (w//4, h//3), (3*w//4, h//4), 3, max(3, h//100))
    
    return mask

def create_overlay(image, mask):
    """Create visualization overlay"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    overlay = image.copy()
    
    # Colors: Life=Red, Head=Green, Heart=Blue
    colors = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }
    
    for class_id, color in colors.items():
        class_mask = (mask == class_id)
        overlay[class_mask] = overlay[class_mask] * 0.5 + np.array(color) * 0.5
    
    return overlay.astype(np.uint8)

def process_palm_image(image):
    """Process palm image - demo version"""
    if image is None:
        return None, "Please upload an image", ""
    
    # Convert to numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Handle RGBA
    if len(image_np.shape) == 3 and image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Create demo mask
    mask = create_demo_segmentation(image_np)
    
    # Create overlay
    overlay_img = create_overlay(image_np, mask)
    
    # Demo features
    features_text = """
## 🎯 Demo Mode - Feature Extraction

### Line Lengths (approximate)
- **Life Line**: 156 pixels
- **Head Line**: 245 pixels  
- **Heart Line**: 198 pixels

### Curvature (Tortuosity)
- **Life Line**: 1.05 (Straight)
- **Head Line**: 1.12 (Slightly curved)
- **Heart Line**: 1.18 (Curved)

### Line Angles
- **Life Line**: 89.2° (Vertical)
- **Head Line**: 2.1° (Horizontal)
- **Heart Line**: 5.3°

### Intersections
- **Life-Head**: 1
- **Life-Heart**: 0
- **Head-Heart**: 0
"""
    
    classification_text = """
## 📊 Palm Analysis (Demo)

### Dominant Line
**Head Line** (Confidence: 68%)

### Palm Type
**Balanced**

### Career Shift Indicator
**Yes** (Confidence: 70%)

---
> ⚠️ **Note**: This is demo mode with simulated data.  
> Train a model on real palm images for actual predictions.
"""
    
    return overlay_img, features_text, classification_text

# Create Gradio interface
with gr.Blocks(title="Palm Line Analysis - Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖐️ Palm Line Segmentation & Analysis (Demo)")
    gr.Markdown("""
    Upload a palm image to see the demo segmentation and analysis.
    
    **Color Legend**:
    - 🔴 **Red**: Life Line
    - 🟢 **Green**: Head Line
    - 🔵 **Blue**: Heart Line
    
    > **Demo Mode**: Currently showing simulated segmentation. Train the model for real predictions.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="📤 Upload Palm Image", type="pil", height=400)
            analyze_btn = gr.Button("🔍 Analyze Palm", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="🎨 Segmentation Result", height=400)
    
    with gr.Row():
        with gr.Column():
            features_output = gr.Markdown(label="📐 Extracted Features")
        
        with gr.Column():
            classification_output = gr.Markdown(label="🏷️ Palm Classification")
    
    # Event handler
    analyze_btn.click(
        fn=process_palm_image,
        inputs=[input_image],
        outputs=[output_image, features_output, classification_output]
    )
    
    gr.Markdown("""
    ---
    ### About This Demo
    This is a demonstration interface for the Palm-Astro History project.  
    The system uses computer vision to segment palm lines and extract geometric features.
    
    **To enable real predictions**:
    1. Collect ~200 annotated palm images
    2. Train the U-Net model: `python train.py`
    3. The app will automatically use the trained model
    
    **Features**:
    - U-Net segmentation (ResNet18 encoder)
    - Geometric feature extraction
    - Rule-based classification
    - Interpretable predictions
    """)

if __name__ == "__main__":
    print("🚀 Launching Gradio app...")
    print("📍 URL: http://localhost:7860")
    demo.launch(share=False, server_port=7860)
