import os
import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split

# Load images and masks
def load_data(image_dir, mask_dir):
    images = []
    masks = []
    
    for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):  # Adjust extension if needed
        mask_path = os.path.join(mask_dir, os.path.basename(img_path))
        
        if os.path.exists(mask_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            
            images.append(img)
            masks.append(mask)

    return np.array(images), np.array(masks)

# Normalize and split data
def preprocess_data(images, masks):
    images = images / 255.0  # Normalize to [0, 1]
    masks = masks / 255.0  # Assuming masks are binary (0 or 255)
    return train_test_split(images, masks, test_size=0.2, random_state=42)
