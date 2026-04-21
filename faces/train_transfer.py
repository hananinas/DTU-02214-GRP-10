#!/usr/bin/env python3

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from transfer_model import (
    DEFAULT_TRANSFER_MODEL_PATH,
    IMG_SIZE,
    POSITIVE_CLASS_NAME,
    build_transfer_model,
    resolve_negative_class_dir,
    save_transfer_metadata,
)

DATA_DIR = Path(__file__).parent / "data" / "classified_faces"


def collect_samples(dataset_dir: Path) -> tuple[list[str], np.ndarray]:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            "Missing dataset directory: "
            f"{dataset_dir}\n"
            "Build it first with: uv run build_member_non_member_dirs.py"
        )

    member_dir = dataset_dir / POSITIVE_CLASS_NAME

    try:
        negative_dir = resolve_negative_class_dir(dataset_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\n"
            "Build the dataset first with: uv run build_member_non_member_dirs.py"
        ) from exc

    if not member_dir.exists():
        raise FileNotFoundError(
            f"Missing member directory: {member_dir}\n"
            "Build the dataset first with: uv run build_member_non_member_dirs.py"
        )

    paths: list[str] = []
    labels: list[int] = []

    for image_path in sorted(member_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(image_path))
            labels.append(1)

    for image_path in sorted(negative_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(str(image_path))
            labels.append(0)

    if len(paths) < 10:
        raise ValueError(f"Not enough images found under {dataset_dir}")

    return paths, np.asarray(labels, dtype=np.float32)


def import_tf():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return tf, keras, preprocess_input


def load_image_array(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(IMG_SIZE)
        return np.asarray(image, dtype=np.float32)


def make_dataset(paths: Sequence[str], labels: np.ndarray, training: bool):
    tf, _, preprocess_input = import_tf()

    def generator():
        for image_path, label in zip(paths, labels):
            yield load_image_array(image_path), np.float32(label)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )

    if training:
        dataset = dataset.shuffle(len(paths), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda image, label: (preprocess_input(image), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(16).prefetch(tf.data.AUTOTUNE)


def find_best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.1, 0.9, 81):
        predictions = (probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(labels, predictions)
        if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def main():
    tf, keras, _ = import_tf()

    parser = argparse.ArgumentParser(description="Train the MobileNetV2 transfer-learning face classifier")
    parser.add_argument("--dataset-dir", type=Path, default=DATA_DIR, help=f"Dataset root (default: {DATA_DIR})")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_TRANSFER_MODEL_PATH,
        help=f"Where to save the trained model (default: {DEFAULT_TRANSFER_MODEL_PATH})",
    )
    parser.add_argument("--head-epochs", type=int, default=6, help="Epochs with a frozen backbone")
    parser.add_argument("--finetune-epochs", type=int, default=6, help="Epochs with the top MobileNetV2 layers unfrozen")
    args = parser.parse_args()

    paths, labels = collect_samples(args.dataset_dir)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels,
    )

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels, training=False)
    test_ds = make_dataset(test_paths, test_labels, training=False)

    model, base_model = build_transfer_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.model_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]

    print(f"Training on {len(train_paths)} images, validating on {len(val_paths)}, testing on {len(test_paths)}")
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs, callbacks=callbacks, verbose=1)

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs, callbacks=callbacks, verbose=1)

    best_model = keras.models.load_model(args.model_path)
    val_probabilities = best_model.predict(val_ds, verbose=0).reshape(-1)
    test_probabilities = best_model.predict(test_ds, verbose=0).reshape(-1)

    threshold, val_balanced_accuracy = find_best_threshold(val_probabilities, val_labels.astype(int))
    test_predictions = (test_probabilities >= threshold).astype(int)

    metadata = {
        "image_size": list(IMG_SIZE),
        "positive_class": POSITIVE_CLASS_NAME,
        "threshold": threshold,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "val_balanced_accuracy": float(val_balanced_accuracy),
        "test_accuracy": float(accuracy_score(test_labels, test_predictions)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_labels, test_predictions)),
        "tensorflow_version": tf.__version__,
    }
    save_transfer_metadata(args.model_path, metadata)

    print(f"Saved model to {args.model_path}")
    print(f"Saved metadata to {args.model_path.with_suffix('.json')}")
    print(f"Best validation threshold: {threshold:.2f}")
    print(f"Validation balanced accuracy: {val_balanced_accuracy:.4f}")
    print(f"Test accuracy: {metadata['test_accuracy']:.4f}")
    print(f"Test balanced accuracy: {metadata['test_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
