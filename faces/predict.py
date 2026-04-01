#!/usr/bin/env python3

import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

MODEL_PATH = "model/best_model.pth"
IMAGE_SIZE = 64


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def load_model(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FaceClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    return model, threshold


def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr[None, ...], dtype=torch.float32)


def predict(image_path, model, threshold, device):
    x = preprocess_image(image_path).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(x)).item()
    return probability >= threshold, probability


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")

    model, threshold = load_model(MODEL_PATH, device)
    image_path = sys.argv[1]

    is_member, confidence = predict(image_path, model, threshold, device)
    result = (
        "Yes - team member face detected"
        if is_member
        else "No - no team member face detected"
    )
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
