#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

from PIL import Image

from download_faces import DATA_DIR, crop_faces


NON_MEMBER_DIR = Path(DATA_DIR) / "non_member"
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


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1 for child in path.iterdir() if child.is_file() and is_image_file(child)
    )


def is_large_enough_face(image_path: Path, min_size: int) -> bool:
    with Image.open(image_path) as img:
        width, height = img.size
    return width >= min_size and height >= min_size


def build_non_member_dir(
    source_dir: Path, non_member_dir: Path, limit: int, min_size: int
):
    existing_count = count_images(non_member_dir)
    if existing_count >= limit:
        print(
            f"Non-member directory already has {existing_count} images; target is {limit}"
        )
        return

    copied = 0
    next_index = existing_count
    for image_path in sorted(source_dir.iterdir()):
        if existing_count + copied >= limit:
            break
        if not image_path.is_file() or not is_image_file(image_path):
            continue
        if not is_large_enough_face(image_path, min_size):
            continue

        target_name = f"coco_{next_index:05d}{image_path.suffix.lower()}"
        target_path = non_member_dir / target_name
        if target_path.exists():
            next_index += 1
            continue

        shutil.copy2(image_path, target_path)
        copied += 1
        next_index += 1

    print(
        f"Copied {copied} non-member images to {non_member_dir} (total: {existing_count + copied})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Top up data/non_member from COCO face crops"
    )
    parser.add_argument(
        "--non-member-count",
        type=int,
        default=1000,
        help="Maximum number of COCO face crops to place in non_member",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=48,
        help="Minimum width and height for COCO face crops",
    )
    args = parser.parse_args()

    NON_MEMBER_DIR.mkdir(parents=True, exist_ok=True)

    missing_non_members = max(args.non_member_count - count_images(NON_MEMBER_DIR), 0)
    crop_faces(num_faces=max(missing_non_members * 3, 1))
    if not COCO_FACES_DIR.exists():
        raise FileNotFoundError(f"Missing COCO face crops: {COCO_FACES_DIR}")

    build_non_member_dir(
        COCO_FACES_DIR,
        NON_MEMBER_DIR,
        limit=args.non_member_count,
        min_size=args.min_face_size,
    )

    print(f"Updated non-member dataset under {NON_MEMBER_DIR}")


if __name__ == "__main__":
    main()
