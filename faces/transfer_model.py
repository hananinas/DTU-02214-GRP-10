#!/usr/bin/env python3

import json
import os
import site
import sys
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
    ensure_tensorflow_runtime_environment()
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return tf, keras, layers, MobileNetV2, preprocess_input


def _tensorflow_cuda_library_dirs() -> list[str]:
    library_dirs: list[str] = []

    for site_packages in site.getsitepackages():
        nvidia_root = Path(site_packages) / "nvidia"
        if not nvidia_root.exists():
            continue

        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                library_dirs.append(str(lib_dir))

    return library_dirs


def ensure_tensorflow_runtime_environment() -> None:
    if sys.platform != "linux":
        return

    library_dirs = _tensorflow_cuda_library_dirs()
    if not library_dirs:
        return

    current_paths = [
        path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path
    ]
    missing_paths = [path for path in library_dirs if path not in current_paths]

    if not missing_paths or os.environ.get("TF_CUDA_LIBPATH_READY") == "1":
        return

    updated_environment = os.environ.copy()
    updated_environment["LD_LIBRARY_PATH"] = ":".join(library_dirs + current_paths)
    updated_environment["TF_CUDA_LIBPATH_READY"] = "1"
    argv = getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv]
    os.execvpe(sys.executable, argv, updated_environment)


def configure_tensorflow_device() -> str:
    tf, _, _, _, _ = import_keras_dependencies()
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        return "CPU"

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # TensorFlow only allows this before the runtime is initialized.
            pass

    return ", ".join(gpu.name for gpu in gpus)


def build_transfer_model(dropout_rate: float = 0.35):
    configure_tensorflow_device()
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


def save_transfer_metadata(
    model_path: os.PathLike[str] | str, metadata: dict[str, object]
):
    metadata_path = metadata_path_for_model(model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_transfer_model(model_path: os.PathLike[str] | str):
    configure_tensorflow_device()
    _, keras, _, _, _ = import_keras_dependencies()
    model = keras.models.load_model(model_path)
    metadata = load_transfer_metadata(model_path)
    return model, metadata


def preprocess_pil_image(image: Image.Image) -> np.ndarray:
    _, _, _, _, preprocess_input = import_keras_dependencies()

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize shortest side to target, then center crop (maintains aspect ratio)
    w, h = image.size
    target_w, target_h = IMG_SIZE

    # Resize so that the shorter side matches target
    if w < h:
        new_w = target_w
        new_h = int(h * (target_w / w))
    else:
        new_h = target_h
        new_w = int(w * (target_h / h))

    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop to get exact target size
    w, h = image.size
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))

    array = np.asarray(image, dtype=np.float32)
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
