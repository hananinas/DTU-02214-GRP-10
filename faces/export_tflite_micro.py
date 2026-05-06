#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

from train_transfer import collect_samples, load_image_array
from transfer_model import (
    DEFAULT_TRANSFER_MODEL_PATH,
    IMG_SIZE,
    ensure_tensorflow_runtime_environment,
    load_transfer_metadata,
)


DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "esp32" / "main"


def import_tf():
    ensure_tensorflow_runtime_environment()
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    return tf, preprocess_input


def representative_dataset(data_dir: Path, limit: int):
    _, preprocess_input = import_tf()
    paths, _ = collect_samples(data_dir)

    for image_path in paths[:limit]:
        image = load_image_array(image_path)
        batch = np.expand_dims(image, axis=0).astype(np.float32)
        yield [preprocess_input(batch)]


def convert_model(
    model_path: Path, data_dir: Path, output_tflite: Path, representative_count: int
):
    tf, _ = import_tf()
    converter = tf.lite.TFLiteConverter.from_keras_model(
        tf.keras.models.load_model(model_path)
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        data_dir, representative_count
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    model_bytes = converter.convert()
    output_tflite.parent.mkdir(parents=True, exist_ok=True)
    output_tflite.write_bytes(model_bytes)
    return model_bytes


def format_c_array(data: bytes, symbol: str) -> str:
    lines = [f'#include "model.h"', "", f"const unsigned char {symbol}[] = {{"]
    for offset in range(0, len(data), 12):
        chunk = data[offset : offset + 12]
        values = ", ".join(f"0x{byte:02x}" for byte in chunk)
        lines.append(f"    {values},")
    lines.append("};")
    lines.append(f"const unsigned int {symbol}_len = {len(data)};")
    lines.append("")
    return "\n".join(lines)


def write_model_header(output_dir: Path, threshold: float, test_accuracy: float | None):
    width, height = IMG_SIZE
    accuracy = -1.0 if test_accuracy is None else test_accuracy
    header = f"""#pragma once

#include <stddef.h>

#define MODEL_INPUT_WIDTH {width}
#define MODEL_INPUT_HEIGHT {height}
#define MODEL_INPUT_CHANNELS 3
#define MEMBER_THRESHOLD {threshold:.8f}f
#define MODEL_TEST_ACCURACY {accuracy:.8f}f

extern const unsigned char model_binary[];
extern const unsigned int model_binary_len;
"""
    (output_dir / "model.h").write_text(header, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Convert the trained Keras face classifier to TFLite Micro C files"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_TRANSFER_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--representative-count", type=int, default=128)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Missing trained model: {args.model}. Run train_transfer.py first."
        )
    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Missing representative data directory: {args.data_dir}."
        )

    metadata = load_transfer_metadata(args.model)
    threshold = float(metadata.get("threshold", 0.5))
    test_accuracy = metadata.get("test_accuracy")
    if test_accuracy is not None:
        test_accuracy = float(test_accuracy)
    output_tflite = args.output_dir / "model.tflite"

    model_bytes = convert_model(
        args.model, args.data_dir, output_tflite, args.representative_count
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.c").write_text(
        format_c_array(model_bytes, "model_binary"), encoding="utf-8"
    )
    write_model_header(args.output_dir, threshold, test_accuracy)

    summary = {
        "model": str(args.model),
        "output_tflite": str(output_tflite),
        "output_c": str(args.output_dir / "model.c"),
        "output_h": str(args.output_dir / "model.h"),
        "bytes": len(model_bytes),
        "threshold": threshold,
        "test_accuracy": test_accuracy,
        "image_size": list(IMG_SIZE),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
