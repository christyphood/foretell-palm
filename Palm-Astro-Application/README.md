# Palm-Astro History: Palm Line Segmentation & Analysis

A computer vision system that segments major palm lines from images and extracts interpretable geometric features for palm reading analysis.

## 🎯 Project Goal

Build an AI system that:
- Segments major palm lines (Life, Head, Heart) from Indian palm images
- Extracts geometric features (length, curvature, angles, intersections)
- Predicts interpretable palm attributes (line dominance, palm type)
- Demonstrates feature-based reasoning (not random outputs)

## 🏗️ Architecture

### Segmentation Model
- **Model**: U-Net with ResNet18 encoder
- **Input**: RGB palm images (512x512 or 256x256)
- **Output**: 4-class segmentation (Background, Life, Head, Heart lines)
- **Framework**: PyTorch + segmentation-models-pytorch

### Feature Extraction
Geometric features computed from segmentation masks:
- **Line Lengths**: Arc length of each palm line
- **Curvature**: Tortuosity ratio (arc length / straight-line distance)
- **Angles**: Orientation angles relative to horizontal
- **Intersections**: Count of line crossing points
- **Coverage**: Total palm line area ratio

### Classification
Rule-based and feature-driven classification:
- **Dominant Line**: Determined by relative line lengths
- **Palm Type**: Based on average curvature (Curved/Balanced/Straight)
- **Career Shift Indicator**: Synthetic label based on head line angle and intersections

## 📁 Project Structure

```
palm-astro/
├── data/
│   ├── images/          # Palm images
│   └── masks/           # Segmentation masks (PNG, class indices)
├── utils/
│   ├── __init__.py
│   ├── create_dummy_data.py    # Generate synthetic data for testing
│   └── feature_extraction.py   # Feature computation functions
├── results/
│   └── best_model.pth          # Trained model weights
├── app.py                       # Gradio web interface
├── train.py                     # Training script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── evaluation.md                # Performance metrics and analysis
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd abhiwan

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

**Option A: Use Dummy Data (for testing)**
```bash
python utils/create_dummy_data.py
```

**Option B: Use Real Dataset**
- Place palm images in `data/images/`
- Place corresponding masks in `data/masks/`
- Mask format: PNG with pixel values 0 (background), 1 (Life), 2 (Head), 3 (Heart)

### 3. Training

```bash
python train.py
```

This will:
- Load data from `data/` directory
- Train U-Net model for 10 epochs (configurable)
- Save best model to `results/best_model.pth`
- Display training and validation loss

### 4. Run the Demo

```bash
python app.py
```

This launches a Gradio web interface at `http://localhost:7860` where you can:
- Upload palm images
- View segmentation overlays
- See extracted features
- Get palm analysis predictions

## 📊 Dataset

### Current Status
The project includes a dummy data generator for testing purposes. For production use, we recommend:

### Recommended Datasets
1. **Roboflow Universe**: [PalmLinesDetection](https://universe.roboflow.com) - Pre-annotated palm line dataset
2. **Kaggle 11k Hands**: Can be annotated for palm lines
3. **Custom Dataset**: Collect ~200 Indian palm images and annotate using tools like LabelMe or CVAT

### Data Requirements
- **Quantity**: ~200 annotated images (minimum for training)
- **Format**: JPG/PNG images + corresponding masks
- **Resolution**: 512x512 recommended
- **Annotations**: Life, Head, and Heart lines labeled
- **Metadata**: Hand type (left/right), demographics (optional)

### Data Augmentation
Applied during training:
- Resize to 256x256
- Horizontal flip
- Rotation (±10°)
- Scale and shift
- Color jitter (optional)

## 🔬 Evaluation Metrics

### Target Metrics
| Metric | Target | Description |
|--------|--------|-------------|
| Mean IoU | ≥ 0.70 | Intersection over Union for segmentation |
| Dice Score | ≥ 0.70 | Per-line segmentation accuracy |
| Classification F1 | ≥ 0.65 | Palm type classification |
| Inference Time | < 1s | Per image on CPU |

See `evaluation.md` for detailed results and analysis.

## 🧠 Model Details

### Training Configuration
```python
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.001
OPTIMIZER = Adam
LOSS = CrossEntropyLoss
TRAIN_SPLIT = 80%
VALIDATION_SPLIT = 20%
```

### Preprocessing
- Resize to 256x256
- ImageNet normalization
- Spatial augmentations

### Inference Pipeline
1. Load image → Preprocess
2. Model prediction → Get class masks
3. Extract features from masks
4. Rule-based classification
5. Display results with confidence scores

## 📈 Example Output

**Input**: Palm image
**Output**:
- Segmentation overlay (colored lines)
- Feature values:
  - Life length: 234.5 px
  - Head curvature: 1.23
  - Heart angle: 15.3°
- Classification:
  - Dominant: Head Line (67%)
  - Type: Balanced
  - Career shift: Yes (70%)

## 🔒 Ethics & Privacy

### Ethical Considerations
- **Pseudoscience Disclaimer**: Palm reading is not scientifically validated. This project is for educational and technical demonstration purposes only.
- **No Medical Claims**: Results should not be used for medical, psychological, or life decisions.
- **Cultural Sensitivity**: Palm reading traditions vary across cultures. This system uses generic geometric features.

### Data Privacy
- **No Personal Data**: Do not collect or store identifiable information with palm images.
- **Local Processing**: All inference runs locally; no data sent to external servers.
- **User Consent**: Ensure informed consent when collecting palm images.
- **Anonymization**: Remove metadata (EXIF, location) from images before processing.

### Best Practices
- Clearly communicate that results are algorithmic, not mystical
- Do not use for discrimination or profiling
- Respect user privacy and data rights
- Use only for research, education, or entertainment

## 🛠️ Customization

### Changing Model
Edit `train.py` to use a different architecture:
```python
model = smp.DeepLabV3Plus(encoder_name="efficientnet-b0", ...)
```

### Adding Features
Edit `utils/feature_extraction.py` to compute new features:
```python
def get_line_thickness(mask):
    # Your implementation
    pass
```

### Modifying Classifications
Edit the `classify_palm()` function in `utils/feature_extraction.py`.

## 🐛 Troubleshooting

**Model not loading**: Ensure `results/best_model.pth` exists by running `train.py` first.

**Import errors**: Install all dependencies via `pip install -r requirements.txt`.

**CUDA out of memory**: Reduce `BATCH_SIZE` in `train.py`.

**Gradio not launching**: Check if port 7860 is available, or specify a different port:
```python
demo.launch(server_port=8080)
```

## 📚 References

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Gradio Documentation](https://gradio.app/docs)

## 👥 Contributing

Contributions welcome! Areas for improvement:
- Better datasets (especially Indian palm images)
- Advanced feature extraction (e.g., line texture, bifurcations)
- ML-based classification (replace rules with trained classifier)
- Multi-task learning (joint segmentation + classification)
- Mobile deployment

## 📄 License

This project is for educational purposes. Please ensure ethical use and comply with local regulations.

---

**Built for Palm-Astro History Assignment**  
*Computer Vision + Multimodal Inference*
