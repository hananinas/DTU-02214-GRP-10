#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image

from download_faces import DATA_DIR, crop_faces


SOURCE_MEMBER_DIR = Path(DATA_DIR) / "face2_dataset"
OUTPUT_ROOT = Path(DATA_DIR) / "classified_faces"
MEMBER_DIR = OUTPUT_ROOT / "member"
NON_MEMBER_DIR = OUTPUT_ROOT / "non_member"
COCO_FACES_DIR = Path(DATA_DIR) / "faces"


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def reset_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def build_member_dir(source_dir: Path, member_dir: Path):
    copied = 0
    for image_path in sorted(source_dir.iterdir()):
        if not image_path.is_file() or not is_image_file(image_path):
            continue

        shutil.copy2(image_path, member_dir / image_path.name)
        copied += 1

    print(f"Copied {copied} member images to {member_dir}")


def is_large_enough_face(image_path: Path, min_size: int) -> bool:
    with Image.open(image_path) as img:
        width, height = img.size
    return width >= min_size and height >= min_size


def build_non_member_dir(
    source_dir: Path, non_member_dir: Path, limit: int, min_size: int
):
    copied = 0
    for image_path in sorted(source_dir.iterdir()):
        if copied >= limit:
            break
        if not image_path.is_file() or not is_image_file(image_path):
            continue
        if not is_large_enough_face(image_path, min_size):
            continue

        target_name = f"coco_{copied:05d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, non_member_dir / target_name)
        copied += 1

    print(f"Copied {copied} non-member images to {non_member_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build member/non_member face directories from local faces and COCO face crops"
    )
    parser.add_argument(
        "--non-member-count",
        type=int,
        default=500,
        help="Maximum number of COCO face crops to place in non_member",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=48,
        help="Minimum width and height for COCO face crops",
    )
    args = parser.parse_args()

    if not SOURCE_MEMBER_DIR.exists():
        raise FileNotFoundError(f"Missing source member dataset: {SOURCE_MEMBER_DIR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reset_directory(MEMBER_DIR)
    reset_directory(NON_MEMBER_DIR)

    build_member_dir(SOURCE_MEMBER_DIR, MEMBER_DIR)

    crop_faces(num_faces=args.non_member_count * 3)
    if not COCO_FACES_DIR.exists():
        raise FileNotFoundError(f"Missing COCO face crops: {COCO_FACES_DIR}")

    build_non_member_dir(
        COCO_FACES_DIR,
        NON_MEMBER_DIR,
        limit=args.non_member_count,
        min_size=args.min_face_size,
    )

    print(f"Built dataset under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
