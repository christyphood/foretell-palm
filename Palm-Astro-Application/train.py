import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "train_images_512")
MASKS_DIR = os.path.join(DATA_DIR, "train_masks_512")
MODEL_SAVE_PATH = "results/best_model.pth"
BATCH_SIZE = 4
EPOCHS = 10 # Small for testing/dummy data
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PalmDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Only include images that have a corresponding mask
        self.ids = []
        for file in os.listdir(images_dir):
            if file.startswith('.'): continue
            
            name = os.path.splitext(file)[0]
            mask_name = name + ".png"
            if os.path.exists(os.path.join(masks_dir, mask_name)):
                self.ids.append(name)
        
        print(f"Found {len(self.ids)} valid samples (image + mask)")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path = os.path.join(self.images_dir, idx + ".jpg")
        mask_path = os.path.join(self.masks_dir, idx + ".png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, 0) # Read as grayscale
        
        # Mask is 0, 1, 2, 3. We need to one-hot encode it or keep as class indices?
        # SMP models usually take raw images. Loss function (CrossEntropy) takes class indices.
        # But we need to be careful about resizing.
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask.long()

def get_training_augmentation():
    train_transform = [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)

def train():
    if not os.path.exists(IMAGES_DIR):
        print("Data directory not found. Please ensure 'data/images' and 'data/masks' exist.")
        return

    # Dataset
    full_dataset = PalmDataset(IMAGES_DIR, MASKS_DIR, transform=get_training_augmentation())
    
    # Split (simple split)
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4, # Background, Life, Head, Heart
    )
    model.to(DEVICE)

    # Loss and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Starting training on {DEVICE}...")
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved!")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    train()
