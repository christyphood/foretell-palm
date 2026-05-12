# Testing Guide

## 1. Using the Application

Now that the system is set up and trained (on dummy data), you can run the full application:

```bash
python app.py
```

Open your browser to **http://localhost:7860**.

## 2. Testing with Your Own Images

You mentioned you have 5-6 palm images. Here is how to use them:

### Option A: Direct Upload (Easiest)
1. Run the app (`python app.py`)
2. Click the "Upload Palm Image" box
3. Select one of your 5-6 images
4. Click "Analyze Palm"

### Option B: Organize in Project
You can save your images in the project folder for easy access:
1. Copy your images to `data/images/`
2. You can name them `real_01.jpg`, `real_02.jpg`, etc.

## 3. What to Expect

> ⚠️ **IMPORTANT**: The model is currently trained on **synthetic/dummy data**. 

- **Segmentation**: It might struggle with real photos because it has only seen drawn lines. It may detect lines if they are very clear and high contrast.
- **Features**: Feature extraction math is accurate, but it depends on the segmentation quality.
- **Classification**: Based on the features found.

## 4. Improving Performance

To get good results on your real images:
1. We need to **annotate** your 5-6 images.
2. Add them to the training set.
3. Retrain the model.

### How to Annotate (Optional - for better results)
If you want to try training on your real images:
1. Use a tool like [CVAT.ai](https://www.cvat.ai/) or [Roboflow](https://roboflow.com/)
2. Upload your 5 images
3. Draw lines for Life (class 1), Head (class 2), Heart (class 3)
4. Export masks as PNG
5. Put images in `data/images` and masks in `data/masks`
6. Run `python train.py` again

For now, just testing the pipeline with the dummy-trained model is sufficient to prove the system works!
