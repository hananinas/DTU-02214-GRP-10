#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from transfer_model import (
    configure_tensorflow_device,
    DEFAULT_TRANSFER_MODEL_PATH,
    load_transfer_model,
    predict_member_probability_from_pil_image,
)

BASE_DIR = os.path.dirname(__file__)
TORCH_MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
MODEL_PATH = (
    str(DEFAULT_TRANSFER_MODEL_PATH)
    if DEFAULT_TRANSFER_MODEL_PATH.exists()
    else TORCH_MODEL_PATH
)
IMAGE_SIZE = 64


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
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
    if path.endswith(".keras"):
        tensorflow_device = configure_tensorflow_device()
        model, metadata = load_transfer_model(path)
        return "keras", model, float(metadata.get("threshold", 0.5)), tensorflow_device

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        threshold = float(checkpoint.get("threshold", 0.5))
    else:
        state_dict = checkpoint
        meta_path = os.path.splitext(path)[0] + "_meta.npz"
        threshold = 0.5
        if os.path.exists(meta_path):
            try:
                meta = np.load(meta_path)
                threshold = float(meta["threshold"])
            except Exception:
                pass

    model = FaceClassifier()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return "torch", model, threshold, str(device)


def preprocess_image(image_path):
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr[None, ...], dtype=torch.float32)


def predict(image_path, backend, model, threshold, device):
    if backend == "keras":
        with Image.open(image_path) as image:
            probability = predict_member_probability_from_pil_image(image, model)
        return probability >= threshold, probability

    x = preprocess_image(image_path).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(x)).item()
    return probability >= threshold, probability


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    device = get_device()
    backend, model, threshold, runtime_device = load_model(MODEL_PATH, device)
    print(f"Using device: {runtime_device}")
    image_path = sys.argv[1]

    is_member, confidence = predict(image_path, backend, model, threshold, device)
    result = (
        "Yes - team member face detected"
        if is_member
        else "No - no team member face detected"
    )
    print(f"Backend: {backend}")
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
