#!/usr/bin/env python3

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = (96, 96)
DEFAULT_MODEL_DIR = Path(__file__).parent / "model"
DEFAULT_TRANSFER_MODEL_PATH = DEFAULT_MODEL_DIR / "transfer_learning_model.keras"
DEFAULT_TRANSFER_METADATA_PATH = DEFAULT_MODEL_DIR / "transfer_learning_model.json"
POSITIVE_CLASS_NAME = "member"
NEGATIVE_CLASS_ALIASES = ("non_member", "non-member")


def import_keras_dependencies():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return tf, keras, layers, MobileNetV2, preprocess_input


def build_transfer_model(dropout_rate: float = 0.35):
    _, keras, layers, MobileNetV2, _ = import_keras_dependencies()

    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="member_probability")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def metadata_path_for_model(model_path: os.PathLike[str] | str) -> Path:
    path = Path(model_path)
    return path.with_suffix(".json")


def load_transfer_metadata(model_path: os.PathLike[str] | str) -> dict[str, object]:
    metadata_path = metadata_path_for_model(model_path)
    if not metadata_path.exists():
        return {
            "threshold": 0.5,
            "image_size": list(IMG_SIZE),
            "positive_class": POSITIVE_CLASS_NAME,
        }

    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_transfer_metadata(model_path: os.PathLike[str] | str, metadata: dict[str, object]):
    metadata_path = metadata_path_for_model(model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_transfer_model(model_path: os.PathLike[str] | str):
    _, keras, _, _, _ = import_keras_dependencies()
    model = keras.models.load_model(model_path)
    metadata = load_transfer_metadata(model_path)
    return model, metadata


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    _, _, _, _, preprocess_input = import_keras_dependencies()

    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    return preprocess_input(batch)


def predict_member_probability_from_pil_image(image: Image.Image, model) -> float:
    batch = preprocess_pil_image(image)
    probability = float(model.predict(batch, verbose=0).reshape(-1)[0])
    return probability


def resolve_negative_class_dir(root: Path) -> Path:
    for name in NEGATIVE_CLASS_ALIASES:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a non-member directory under {root}. Expected one of: {NEGATIVE_CLASS_ALIASES}"
    )


def main():
    print("transfer_model.py is a shared helper module.")
    print("Use one of these instead:")
    print("  uv run train_transfer.py")
    print("  uv run predict.py <image_path>")
    print("  uv run camera_app.py")


if __name__ == "__main__":
    main()
