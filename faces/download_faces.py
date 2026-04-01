#!/usr/bin/env python3

import os
import requests
import zipfile
import gdown

DATA_DIR = "data"
FACES_DIR = os.path.join(DATA_DIR, "faces")
FACES_URL = "https://drive.google.com/uc?id=1ZN3bI8uik0d_xcCBY3WdokkSboTnurry"
IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
ANNOTATION_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def download_file(url, dest):
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\rProgress: {downloaded / total * 100:.1f}%",
                        end="",
                        flush=True,
                    )
    print(f"\nSaved to {dest}")


def download_gdrive(url, dest):
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        return
    print(f"Downloading from Google Drive: {url}")
    gdown.download(url, dest, quiet=False)


def extract_zip(path, extract_to):
    print(f"Extracting {path}...")
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def resolve_faces_csv_dir(base_dir):
    candidates = [
        os.path.join(base_dir, "faces_csv"),
        os.path.join(base_dir, "train_face_labels"),
        os.path.join(base_dir, "yoloface_fast_predictions"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("Could not find extracted face annotation directory")


def parse_faces_csv(csv_path):
    faces = []
    if not os.path.exists(csv_path):
        return faces
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 6:
                try:
                    face = {
                        "image_path": parts[0],
                        "score": float(parts[1]),
                        "top": float(parts[2]),
                        "left": float(parts[3]),
                        "bottom": float(parts[4]),
                        "right": float(parts[5]),
                    }
                    if face["score"] >= 0.7:
                        faces.append(face)
                except ValueError:
                    continue
    return faces


def resolve_local_image_path(image_path):
    normalized = image_path.replace("\\", "/")

    if normalized.startswith("/data/coco/train2017/"):
        return os.path.join(DATA_DIR, "train2017", os.path.basename(normalized))

    if normalized.startswith("train2017/"):
        return os.path.join(DATA_DIR, normalized)

    return os.path.join(DATA_DIR, normalized.lstrip("/"))


def crop_faces(num_faces=10):
    os.makedirs(FACES_DIR, exist_ok=True)

    images_zip = os.path.join(DATA_DIR, "train2017.zip")
    images_dir = os.path.join(DATA_DIR, "train2017")

    if not os.path.exists(images_dir):
        download_file(IMAGES_URL, images_zip)
        extract_zip(images_zip, DATA_DIR)
        os.remove(images_zip)

    faces_zip = os.path.join(DATA_DIR, "faces_csv.zip")
    faces_csv_dir = os.path.join(DATA_DIR, "faces_csv")

    if not os.path.exists(faces_csv_dir):
        download_gdrive(FACES_URL, faces_zip)
        extract_zip(faces_zip, DATA_DIR)
        os.remove(faces_zip)
        extracted_dir = resolve_faces_csv_dir(DATA_DIR)
        if extracted_dir != faces_csv_dir:
            if os.path.exists(faces_csv_dir):
                raise FileExistsError(
                    f"Target directory already exists: {faces_csv_dir}"
                )
            os.rename(extracted_dir, faces_csv_dir)

    train_faces_dir = os.path.join(faces_csv_dir, "train2017")
    csv_root_dir = train_faces_dir if os.path.exists(train_faces_dir) else faces_csv_dir

    csv_files = [f for f in os.listdir(csv_root_dir) if f.endswith(".csv")]
    print(f"Found {len(csv_files)} CSV files with face annotations")

    faces_found = 0
    for csv_file in csv_files:
        if faces_found >= num_faces:
            break

        csv_path = os.path.join(csv_root_dir, csv_file)
        faces = parse_faces_csv(csv_path)

        if not faces:
            continue

        for face in faces:
            if faces_found >= num_faces:
                break

            img_path = face["image_path"]
            full_img_path = resolve_local_image_path(img_path)

            if not os.path.exists(full_img_path):
                continue

            from PIL import Image

            img = Image.open(full_img_path)

            left = int(face["left"])
            top = int(face["top"])
            right = int(face["right"])
            bottom = int(face["bottom"])

            left = max(0, left)
            top = max(0, top)
            right = min(img.width, right)
            bottom = min(img.height, bottom)

            if right > left and bottom > top:
                face_img = img.crop((left, top, right, bottom))
                face_filename = f"face_{faces_found:04d}.jpg"
                face_img.save(os.path.join(FACES_DIR, face_filename))
                faces_found += 1
                print(
                    f"[{faces_found}/{num_faces}] Saved {face_filename} (score: {face['score']:.2f})"
                )

    print(f"\nExtracted {faces_found} faces to {FACES_DIR}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    crop_faces(num_faces=2000)
