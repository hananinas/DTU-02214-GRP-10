#!/usr/bin/env python3

import argparse
import os
import pygame
import sys
import serial
import time
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from PIL import Image

from transfer_model import (
    DEFAULT_TRANSFER_MODEL_PATH,
    load_transfer_model,
    predict_member_probability_from_pil_image,
)

# Configuration constants
DEFAULT_PORT = "/dev/ttyACM0"
BAUD_RATE = 921600
SERIAL_TIMEOUT = 2.0
WIDTH = 320
HEIGHT = 240
DISPLAY_SCALE = 2
PREVIEW_WIDTH = WIDTH * DISPLAY_SCALE
PREVIEW_HEIGHT = HEIGHT * DISPLAY_SCALE
SIDEBAR_WIDTH = 360
WINDOW_WIDTH = PREVIEW_WIDTH + SIDEBAR_WIDTH
WINDOW_HEIGHT = PREVIEW_HEIGHT
FRAME_PREAMBLE = b"===FRAME===\n"
TORCH_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model.pth")
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
        model, metadata = load_transfer_model(path)
        return "keras", model, float(metadata.get("threshold", 0.5))

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FaceClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    return "torch", model, threshold


def preprocess_for_model(surface: pygame.Surface) -> torch.Tensor:
    rgb_bytes = pygame.image.tobytes(surface, "RGB")
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), rgb_bytes)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr[None, ...], dtype=torch.float32)


def predict(
    backend, model, threshold: float, surface: pygame.Surface, device
) -> tuple[bool, float]:
    if backend == "keras":
        rgb_bytes = pygame.image.tobytes(surface, "RGB")
        image = Image.frombytes("RGB", (WIDTH, HEIGHT), rgb_bytes)
        probability = predict_member_probability_from_pil_image(image, model)
        return probability >= threshold, probability

    x = preprocess_for_model(surface).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(x)).item()
    return probability >= threshold, probability


def capture_frame(serial_port: serial.Serial) -> pygame.Surface | None:
    chunk = serial_port.read_until(FRAME_PREAMBLE)
    if not chunk.endswith(FRAME_PREAMBLE):
        return None

    frame_rgb565 = serial_port.read(WIDTH * HEIGHT * 2)
    if len(frame_rgb565) != WIDTH * HEIGHT * 2:
        return None

    frame_rgb = bytearray(WIDTH * HEIGHT * 3)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            src_index = (y * WIDTH + x) * 2
            dst_index = (y * WIDTH + x) * 3
            byte1 = frame_rgb565[src_index]
            byte2 = frame_rgb565[src_index + 1]
            r8 = byte1 & 0xF8
            g8 = ((byte1 & 0x07) << 5) | ((byte2 & 0xE0) >> 3)
            b8 = (byte2 & 0x1F) << 3
            frame_rgb[dst_index] = r8
            frame_rgb[dst_index + 1] = g8
            frame_rgb[dst_index + 2] = b8

    return pygame.image.frombuffer(frame_rgb, (WIDTH, HEIGHT), "RGB")


def draw_text(screen, font, text, color, x, y):
    screen.blit(font.render(text, True, color), (x, y))


def draw_progress_bar(screen, rect, value, threshold, fill_color):
    x, y, width, height = rect
    pygame.draw.rect(screen, (28, 30, 36), rect, border_radius=8)
    fill_width = max(0, min(width, int(width * value)))
    if fill_width > 0:
        pygame.draw.rect(
            screen, fill_color, (x, y, fill_width, height), border_radius=8
        )

    threshold_x = x + int(width * threshold)
    pygame.draw.line(
        screen,
        (255, 255, 255),
        (threshold_x, y - 4),
        (threshold_x, y + height + 4),
        2,
    )


def main():
    parser = argparse.ArgumentParser(description="Face classifier with ESP32-S3 camera")
    parser.add_argument(
        "--port", default=DEFAULT_PORT, help=f"Serial port (default: {DEFAULT_PORT})"
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained model")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading model from {args.model}...")
    backend, model, threshold = load_model(args.model, device)
    print("Model loaded.")
    print(f"Using backend: {backend}")
    print(f"Using decision threshold: {threshold:.2f}")

    print(f"Opening serial port {args.port}...")
    try:
        serial_port = serial.Serial(args.port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2)
        serial_port.reset_input_buffer()
    except serial.SerialException as exc:
        print(f"Failed to open serial port {args.port}: {exc}", file=sys.stderr)
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Face Classifier - ESP32-S3")
    title_font = pygame.font.SysFont("monospace", 30, bold=True)
    status_font = pygame.font.SysFont("monospace", 24, bold=True)
    body_font = pygame.font.SysFont("monospace", 18)
    tiny_font = pygame.font.SysFont("monospace", 15)
    clock = pygame.time.Clock()

    print("Connection established. Sending 'S' to start streaming...")
    serial_port.write(b"S")
    time.sleep(0.5)

    frame_count = 0
    inference_interval = 5
    last_result = None
    last_confidence = None
    prediction_history = deque(maxlen=8)
    last_inference_ms = None
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            surface = capture_frame(serial_port)
            if surface is None:
                continue

            frame_count += 1
            if frame_count % inference_interval == 0:
                inference_start = time.perf_counter()
                is_member, confidence = predict(
                    backend, model, threshold, surface, device
                )
                last_inference_ms = (time.perf_counter() - inference_start) * 1000.0
                last_result = is_member
                last_confidence = confidence
                prediction_history.appendleft(("M" if is_member else "N", confidence))

            screen.fill((16, 18, 24))

            preview_surface = pygame.transform.scale(
                surface, (PREVIEW_WIDTH, PREVIEW_HEIGHT)
            )
            screen.blit(preview_surface, (0, 0))

            # Add a subtle frame around the preview.
            pygame.draw.rect(
                screen, (44, 48, 58), (0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT), 4
            )

            sidebar_x = PREVIEW_WIDTH
            pygame.draw.rect(
                screen, (22, 24, 30), (sidebar_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
            )

            draw_text(
                screen, title_font, "Face Monitor", (240, 240, 245), sidebar_x + 24, 20
            )
            draw_text(
                screen,
                tiny_font,
                "ESP32-S3 stream + live classifier",
                (150, 154, 166),
                sidebar_x + 24,
                58,
            )

            if last_result is not None:
                if last_result:
                    label_text = "MEMBER DETECTED"
                    color = (84, 214, 121)
                    panel_color = (20, 56, 33)
                else:
                    label_text = "NO MEMBER DETECTED"
                    color = (255, 117, 117)
                    panel_color = (64, 26, 26)

                pygame.draw.rect(
                    screen,
                    panel_color,
                    (sidebar_x + 20, 92, SIDEBAR_WIDTH - 40, 78),
                    border_radius=16,
                )
                draw_text(screen, status_font, label_text, color, sidebar_x + 36, 118)

                draw_text(
                    screen,
                    body_font,
                    f"Member score: {last_confidence:.2%}",
                    (232, 232, 236),
                    sidebar_x + 24,
                    206,
                )
                draw_text(
                    screen,
                    body_font,
                    f"Non-member score: {1.0 - last_confidence:.2%}",
                    (232, 232, 236),
                    sidebar_x + 24,
                    236,
                )
                draw_text(
                    screen,
                    body_font,
                    f"Threshold: {threshold:.2%}",
                    (232, 232, 236),
                    sidebar_x + 24,
                    266,
                )
                draw_text(
                    screen,
                    body_font,
                    f"Device: {device}",
                    (232, 232, 236),
                    sidebar_x + 24,
                    296,
                )
                draw_text(
                    screen,
                    body_font,
                    f"Inference every {inference_interval} frames",
                    (232, 232, 236),
                    sidebar_x + 24,
                    326,
                )
                if last_inference_ms is not None:
                    draw_text(
                        screen,
                        body_font,
                        f"Inference time: {last_inference_ms:.1f} ms",
                        (232, 232, 236),
                        sidebar_x + 24,
                        356,
                    )

                draw_text(
                    screen,
                    tiny_font,
                    "Member Probability",
                    (180, 184, 196),
                    sidebar_x + 24,
                    392,
                )
                draw_progress_bar(
                    screen,
                    (sidebar_x + 24, 418, SIDEBAR_WIDTH - 48, 24),
                    last_confidence,
                    threshold,
                    color,
                )

                draw_text(
                    screen,
                    tiny_font,
                    "Recent Decisions",
                    (180, 184, 196),
                    sidebar_x + 24,
                    468,
                )
                history_y = 496
                for marker, confidence in prediction_history:
                    chip_color = (84, 214, 121) if marker == "M" else (255, 117, 117)
                    pygame.draw.rect(
                        screen,
                        chip_color,
                        (sidebar_x + 24, history_y, 42, 24),
                        border_radius=12,
                    )
                    draw_text(
                        screen,
                        tiny_font,
                        marker,
                        (20, 20, 24),
                        sidebar_x + 39,
                        history_y + 3,
                    )
                    draw_text(
                        screen,
                        tiny_font,
                        f"{confidence:.1%}",
                        (220, 224, 232),
                        sidebar_x + 82,
                        history_y + 3,
                    )
                    history_y += 30

            else:
                pygame.draw.rect(
                    screen,
                    (36, 40, 50),
                    (sidebar_x + 20, 92, SIDEBAR_WIDTH - 40, 78),
                    border_radius=16,
                )
                draw_text(
                    screen,
                    status_font,
                    "Waiting For Inference",
                    (240, 195, 90),
                    sidebar_x + 36,
                    118,
                )
                draw_text(
                    screen,
                    body_font,
                    "Receiving frames from ESP32...",
                    (232, 232, 236),
                    sidebar_x + 24,
                    206,
                )

            fps = clock.get_fps()
            draw_text(
                screen,
                tiny_font,
                f"FPS: {fps:4.1f}",
                (240, 240, 245),
                16,
                14,
            )
            draw_text(
                screen,
                tiny_font,
                "Q / ESC to quit",
                (240, 240, 245),
                16,
                34,
            )

            pygame.display.flip()
            clock.tick(60)

    finally:
        serial_port.close()
        pygame.quit()


if __name__ == "__main__":
    main()
