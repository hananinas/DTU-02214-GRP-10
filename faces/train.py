#!/usr/bin/env python3

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from transfer_model import (DEFAULT_TRANSFER_MODEL_PATH, IMG_SIZE,
                            POSITIVE_CLASS_NAME, build_transfer_model,
                            configure_tensorflow_device,
                            ensure_tensorflow_runtime_environment,
                            save_transfer_metadata)

DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 16


def collect_samples(data_dir: Path) -> tuple[list[str], np.ndarray]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    member_dir = data_dir / "member"
    non_member_dir = data_dir / "non_member"

    if not member_dir.exists():
        raise FileNotFoundError(f"Missing member directory: {member_dir}")
    if not non_member_dir.exists():
        raise FileNotFoundError(f"Missing non_member directory: {non_member_dir}")

    paths: list[str] = []
    labels: list[int] = []

    for image_path in sorted(member_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            paths.append(str(image_path))
            labels.append(1)

    for image_path in sorted(non_member_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            paths.append(str(image_path))
            labels.append(0)

    if len(paths) < 10:
        raise ValueError(f"Not enough images found under {data_dir}")

    return paths, np.asarray(labels, dtype=np.float32)


def import_tf():
    ensure_tensorflow_runtime_environment()
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return tf, keras, preprocess_input


def load_image_array(image_path: str | bytes) -> np.ndarray:
    if isinstance(image_path, bytes):
        image_path = image_path.decode("utf-8")

    with Image.open(image_path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize shortest side to target, then center crop (maintains aspect ratio)
        w, h = image.size
        target_w, target_h = IMG_SIZE

        if w < h:
            new_w = target_w
            new_h = int(h * (target_w / w))
        else:
            new_h = target_h
            new_w = int(w * (target_h / h))

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop
        w, h = image.size
        left = (w - target_w) // 2
        top = (h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))

        return np.asarray(image, dtype=np.float32)


def make_dataset(
    paths: Sequence[str],
    labels: np.ndarray,
    training: bool,
    seed: int,
    augment: bool = False,
):
    tf, _, preprocess_input = import_tf()

    def load_and_preprocess(image_path, label):
        image = tf.numpy_function(load_image_array, [image_path], tf.float32)
        image.set_shape((*IMG_SIZE, 3))
        return preprocess_input(image), label

    def augment_image(image, label):
        # Light augmentation: horizontal flip only
        image = tf.image.random_flip_left_right(image, seed=seed)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices(
        (list(paths), labels.astype(np.float32))
    )

    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training and augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def configure_reproducibility(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    tf, keras, _ = import_tf()
    keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    return tf, keras


def find_best_threshold(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.1, 0.9, 81):
        predictions = (probabilities >= threshold).astype(int)
        score = balanced_accuracy_score(labels, predictions)
        if score > best_score or (
            np.isclose(score, best_score) and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    total_count = positive_count + negative_count

    return {
        0: total_count / (2.0 * negative_count),
        1: total_count / (2.0 * positive_count),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the MobileNetV2 transfer-learning face classifier"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Data root with member/ and non_member/ subdirs (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_TRANSFER_MODEL_PATH,
        help=f"Where to save the trained model (default: {DEFAULT_TRANSFER_MODEL_PATH})",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=2,
        help="Epochs with a frozen backbone (default: 2 - optimized for convergence)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=2,
        help="Epochs with the top MobileNetV2 layers unfrozen (default: 2 - optimal for this dataset)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic runs"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting (class weights are ON by default to handle imbalance)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (augmentation is ON by default for better generalization)",
    )
    args = parser.parse_args()

    tf, keras = configure_reproducibility(args.seed)
    tensorflow_device = configure_tensorflow_device()

    print(f"Using TensorFlow device(s): {tensorflow_device}")

    paths, labels = collect_samples(args.data_dir)
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

    # Default: augmentation and class weights are ON (best configuration)
    use_augment = not args.no_augment
    use_class_weights = not args.no_class_weights

    train_ds = make_dataset(
        train_paths, train_labels, training=True, seed=args.seed, augment=use_augment
    )
    val_ds = make_dataset(val_paths, val_labels, training=False, seed=args.seed)
    test_ds = make_dataset(test_paths, test_labels, training=False, seed=args.seed)
    class_weight = compute_class_weights(train_labels) if use_class_weights else None

    model, base_model = build_transfer_model(dropout_rate=0.5)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            args.model_path, monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
        ),
    ]

    print(
        f"Training on {len(train_paths)} images, validating on {len(val_paths)}, testing on {len(test_paths)}"
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    best_model = keras.models.load_model(args.model_path)
    val_probabilities = best_model.predict(val_ds, verbose=0).reshape(-1)
    test_probabilities = best_model.predict(test_ds, verbose=0).reshape(-1)

    threshold, val_balanced_accuracy = find_best_threshold(
        val_probabilities, val_labels.astype(int)
    )
    test_predictions = (test_probabilities >= threshold).astype(int)
    test_labels_int = test_labels.astype(int)
    tn, fp, fn, tp = confusion_matrix(
        test_labels_int, test_predictions, labels=[0, 1]
    ).ravel()

    metadata = {
        "architecture": "MobileNetV2",
        "image_size": list(IMG_SIZE),
        "positive_class": POSITIVE_CLASS_NAME,
        "threshold": threshold,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "val_balanced_accuracy": float(val_balanced_accuracy),
        "test_accuracy": float(accuracy_score(test_labels_int, test_predictions)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(test_labels_int, test_predictions)
        ),
        "test_precision_member": float(
            precision_score(test_labels_int, test_predictions, zero_division=0)
        ),
        "test_recall_member": float(
            recall_score(test_labels_int, test_predictions, zero_division=0)
        ),
        "test_f1_member": float(
            f1_score(test_labels_int, test_predictions, zero_division=0)
        ),
        "confusion_matrix": {
            "true_non_member_pred_non_member": int(tn),
            "true_non_member_pred_member": int(fp),
            "true_member_pred_non_member": int(fn),
            "true_member_pred_member": int(tp),
        },
        "seed": args.seed,
        "use_class_weights": use_class_weights,
        "augment": use_augment,
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
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
