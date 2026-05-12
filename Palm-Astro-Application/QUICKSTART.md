# Quick Start Guide

## Testing the Demo (No Model Required)

You can test the UI immediately without training a model:

```bash
# Install minimal dependencies
pip install gradio pillow numpy opencv-python

# Run the demo app
python app_demo.py
```

This will launch the Gradio interface at http://localhost:7860

### What the Demo Does:
- ✅ Accepts palm image uploads
- ✅ Shows simulated segmentation overlay (Red=Life, Green=Head, Blue=Heart)
- ✅ Displays mock feature extraction results
- ✅ Shows example classifications
- ⚠️ Uses fake data (not trained model)

### Testing with Sample Image:
I've generated a sample palm image for you:
- Location: `data/images/test_palm.png`
- Upload this to the demo to see it in action

## Full Setup (With Model Training)

For actual palm line detection, follow the full setup:

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Prepare your dataset
# - Add palm images to data/images/
# - Add masks to data/masks/
# Or generate dummy data:
python utils\create_dummy_data.py

# 3. Train the model
python train.py

# 4. Run the full app (will use trained model)
python app.py
```

## File Guide

| File | Purpose | Dependencies |
|------|---------|--------------|
| `app_demo.py` | Demo UI (no model) | Gradio, PIL, NumPy, OpenCV |
| `app.py` | Full app (with model) | All in requirements.txt |
| `train.py` | Model training | PyTorch, segmentation-models-pytorch |
| `test_features.py` | Test feature extraction | NumPy, OpenCV, scipy |

## Next Steps

1. **Right Now**: Test `app_demo.py` to see the UI
2. **Today**: Collect/source real palm dataset (~200 images)
3. **Tomorrow**: Train model and test full pipeline
4. **End of Week**: Evaluate performance and refine

## Troubleshooting

**Gradio won't install?**
```bash
pip install --upgrade pip
pip install gradio
```

**Port 7860 in use?**
Edit `app_demo.py` and change:
```python
demo.launch(server_port=8080)  # Use different port
```

**Dependencies hanging?**
Try installing one at a time:
```bash
pip install opencv-python
pip install numpy
pip install pillow
pip install gradio
```
