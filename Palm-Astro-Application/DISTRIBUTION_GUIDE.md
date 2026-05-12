# Distribution Guide

## What to Include When Sharing This Project

### ✅ MUST Include (Essential Files)

**Source Code:**
- `app.py` - Gradio application
- `train.py` - Training script
- `utils/` - All utility files
  - `feature_extraction.py`
  - `create_dummy_data.py`
  - `annotator.py`
  - `__init__.py`

**Documentation:**
- `README.md` - Main documentation
- `evaluation.md` - Evaluation framework
- `QUICKSTART.md` - Quick start guide
- `FINAL_STEPS.md` - Completion guide
- `requirements.txt` - Dependencies

**Helper Scripts:**
- `run_train.bat` - Training batch file
- `run_app.bat` - App batch file
- `generate_samples.py` - Sample generator

**Version Control:**
- `.gitignore` - Git ignore rules

---

### ⚠️ OPTIONAL (Can Include or Omit)

**Trained Model:**
- `results/best_model.pth` (45MB)
  - **Include if**: Recipient wants to test without training
  - **Omit if**: They will train their own model

**Data:**
- `data/` folder (~50-100MB with images)
  - **Omit**: Usually too large, privacy concerns
  - **Alternative**: Provide instructions to get dataset from Roboflow/Kaggle

---

### ❌ SHOULD NOT Include

**Large/Temporary Files:**
- `data/` folder (images and masks) - Privacy and size
- `results/` folder except README - Reproducibility
- `__pycache__/` - Python cache
- `*.pyc` files - Compiled Python
- Log files (`training_log.txt`)

**Personal Files:**
- IDE settings (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)

---

## Recommended Distribution Methods

### Method 1: GitHub Repository (Best Practice)

```bash
# Initialize git (if not done)
git init

# Add .gitignore (already created)
# This will automatically exclude data/ and results/

# Add files
git add .

# Commit
git commit -m "Initial commit: Palm-Astro History project"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/palm-astro.git
git push -u origin main
```

**Recipient gets:**
- Clean codebase
- Full documentation
- Can train their own model
- Can get data from external sources

---

### Method 2: ZIP Archive (Quick Share)

**Option A: Code Only (Small ~50KB)**
Include:
- All `.py` files
- All `.md` files
- `requirements.txt`
- `.gitignore`
- Batch files

**Option B: Code + Model (Medium ~50MB)**
Same as above, plus:
- `results/best_model.pth`

**Option C: Full Project (Large ~150MB+)**
Everything including data (not recommended)

---

## Creating a Clean Distribution ZIP

### For Code Only:
1. Create a new folder: `palm-astro-distribution/`
2. Copy these files:
   ```
   *.py (except mismatch.py, debug_*.py, check_imports.py)
   *.md
   *.bat
   requirements.txt
   .gitignore
   utils/*.py
   ```
3. ZIP the folder
4. Share

### For Code + Model:
1. Same as above
2. Also copy `results/best_model.pth`
3. ZIP and share

---

## What to Tell the Recipient

Include this in your email/message:

```
Hi,

I'm sharing my Palm-Astro History project for palm line segmentation.

TO RUN:
1. Install dependencies: pip install -r requirements.txt
2. (Optional) Train model: python train.py
3. Run app: python app.py
4. Open browser to http://localhost:7860

DATASET:
- I've omitted the training data for privacy/size
- You can generate dummy data with: python utils/create_dummy_data.py
- Or get real palm dataset from Roboflow/Kaggle

DOCUMENTATION:
- README.md - Full documentation
- QUICKSTART.md - Quick setup guide

Let me know if you have any questions!
```

---

## File Size Summary

| What to Share | Size | Files |
|---------------|------|-------|
| Code Only | ~50 KB | 15-20 files |
| Code + Model | ~50 MB | + best_model.pth |
| Code + Data | ~150 MB | + 14 images + masks |
| Everything | ~200 MB | All files |

---

## Recommendation

**For assignment submission:**
- Share **Code + Model** (50MB)
- Upload to GitHub (preferred) or ZIP file
- Include link to where they can get dataset

**For collaboration:**
- Share **Code Only** via GitHub
- Collaborators train their own models
- Share dataset separately via Drive/Dropbox

---

## Quick Commands

### Clean project (remove temp files):
```bash
# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Remove logs
rm training_log.txt
```

### Create distribution ZIP (Windows):
```bash
# Compress everything except data/results
# (Manually select files, or use a tool)
```

### Push to GitHub:
```bash
git init
git add .
git commit -m "Palm-Astro History project"
git remote add origin YOUR_REPO_URL
git push -u origin main
```

---

**Question to answer:**
Do you want to share **Code Only** or **Code + Model**?
