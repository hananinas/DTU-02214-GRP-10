#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from huggingface_hub import login
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ID = "hananinas/faces2_dataset"
MODEL_SAVE_DIR = "model"
IMAGE_SIZE = 64
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def sigmoid_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))


def find_best_threshold(probs, labels):
    best_threshold = 0.5
    best_score = -1.0
    best_cm = None

    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_acc = 0.5 * (tpr + tnr)

        if balanced_acc > best_score or (
            np.isclose(balanced_acc, best_score) and threshold > best_threshold
        ):
            best_score = balanced_acc
            best_threshold = float(threshold)
            best_cm = cm

    return best_threshold, best_score, best_cm


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


def load_and_prepare_data(token: str):
    login(token=token, add_to_git_credential=False)
    ds = load_dataset(REPO_ID)

    def extract_images_labels(split):
        images = []
        labels = []
        for item in split:
            img = item["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            images.append(arr)
            labels.append(int(item["label"]))
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    print("Loading train split...")
    x_train, y_train = extract_images_labels(ds["train"])
    print("Loading test split...")
    x_test, y_test = extract_images_labels(ds["test"])

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print(f"Train member ratio: {y_train.mean():.2f}")
    print(f"Test member ratio: {y_test.mean():.2f}")

    return x_train, y_train, x_test, y_test


def train_and_evaluate(x_train, y_train, x_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FaceClassifier().to(device)
    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)

    if positive_count and positive_count < negative_count:
        pos_weight = torch.tensor(
            [negative_count / positive_count],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using positive class weighting: {pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using unweighted loss.")
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    best_val_score = 0.0
    best_threshold = 0.5
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)
                val_logits.append(outputs.cpu().numpy())
                val_labels.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_probs = sigmoid_numpy(np.concatenate(val_logits))
        val_targets = np.concatenate(val_labels).astype(int)
        epoch_threshold, val_score, _ = find_best_threshold(val_probs, val_targets)

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_train_loss:.4f} - "
            f"Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} - "
            f"Val Balanced Acc: {val_score:.4f} @ threshold {epoch_threshold:.2f}"
        )

        if val_score >= best_val_score:
            best_val_score = val_score
            best_threshold = epoch_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": int(IMAGE_SIZE),
                    "epoch": int(epoch),
                    "val_acc": float(val_acc),
                    "val_balanced_acc": float(val_score),
                    "threshold": float(best_threshold),
                },
                os.path.join(MODEL_SAVE_DIR, "best_model.pth"),
            )
            print(
                f"  -> Saved best model (val_balanced_acc: {val_score:.4f}, threshold: {best_threshold:.2f})"
            )

    try:
        best_checkpoint = torch.load(
            os.path.join(MODEL_SAVE_DIR, "best_model.pth"),
            map_location=device,
            weights_only=True,
        )
    except Exception:
        best_checkpoint = torch.load(
            os.path.join(MODEL_SAVE_DIR, "best_model.pth"),
            map_location=device,
            weights_only=False,
        )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_threshold = float(best_checkpoint.get("threshold", best_threshold))

    print(f"\nBest validation balanced accuracy: {best_val_score:.4f}")
    print(f"Best decision threshold: {best_threshold:.2f}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= best_threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Non-member  Member")
    print(f"Actual Non-mem   {cm[0][0]:>3}        {cm[0][1]:>3}")
    print(f"Actual Member    {cm[1][0]:>3}        {cm[1][1]:>3}")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["non-member", "member"]
    )
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "confusion_matrix.png"), dpi=150)
    print(f"Confusion matrix saved to {MODEL_SAVE_DIR}/confusion_matrix.png")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "training_history.png"), dpi=150)
    print(f"Training history saved to {MODEL_SAVE_DIR}/training_history.png")

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HF access token")
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_and_prepare_data(args.token)
    train_and_evaluate(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
