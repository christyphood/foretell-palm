"""
Quick test script to verify feature extraction works
"""
import numpy as np
import sys

# Add utils to path
sys.path.insert(0, '.')

try:
    from utils.feature_extraction import extract_palm_features, classify_palm
    
    # Create a simple test mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Draw simple lines
    # Life line (class 1)
    mask[100:200, 50] = 1
    
    # Head line (class 2)
    mask[120, 50:200] = 2
    
    # Heart line (class 3)
    mask[80, 50:180] = 3
    
    print("Testing feature extraction...")
    features = extract_palm_features(mask)
    
    print("\n=== FEATURES ===")
    for key, value in features.items():
        print(f"{key}: {value:.2f}")
    
    print("\n=== CLASSIFICATION ===")
    classification = classify_palm(features)
    for key, value in classification.items():
        print(f"{key}: {value}")
    
    print("\n✅ Feature extraction works!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("❌ Missing dependencies")
except Exception as e:
    print(f"Error: {e}")
    print("❌ Feature extraction failed")
