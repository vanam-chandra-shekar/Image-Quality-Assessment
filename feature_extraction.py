import numpy as np
import cv2

def extract_features(image_path):
    """Extracts 8 features from an image for IQA model."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))  # Resize to fixed dimensions

    # Example features - replace with actual ones used in training
    mean = np.mean(img)
    std_dev = np.std(img)
    contrast = img.max() - img.min()

    # Additional dummy features (ensure total is 8)
    feature_4 = np.median(img)
    feature_5 = np.percentile(img, 25)
    feature_6 = np.percentile(img, 75)
    feature_7 = np.var(img)
    feature_8 = np.mean(img[:100, :100])  # Mean of top-left corner

    features = np.array([mean, std_dev, contrast, feature_4, feature_5, feature_6, feature_7, feature_8])
    return features
