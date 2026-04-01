#!/usr/bin/env python3

from pathlib import Path
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import HfApi, login

DATA_DIR = Path(__file__).parent / "data"
CLASSIFIED_DIR = DATA_DIR / "classified_faces"
MEMBER_DIR = CLASSIFIED_DIR / "member"
NON_MEMBER_DIR = CLASSIFIED_DIR / "non_member"
LABELS_FILE = CLASSIFIED_DIR / "labels.txt"


def generate_labels_file() -> list[dict[str, object]]:
    rows = []

    for label_name, label_value, source_dir in [
        ("member", 1, MEMBER_DIR),
        ("non_member", 0, NON_MEMBER_DIR),
    ]:
        if not source_dir.exists():
            continue

        for image_path in sorted(source_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            rows.append(
                {
                    "image": str(image_path),
                    "label": label_value,
                    "filename": image_path.name,
                    "label_name": label_name,
                }
            )

    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w") as f:
        for row in rows:
            f.write(f"{row['filename']}: {row['label_name']}\n")

    return rows


def build_dataset() -> DatasetDict:
    rows = generate_labels_file()
    dataset_rows = [{"image": row["image"], "label": row["label"]} for row in rows]

    dataset = Dataset.from_dict(
        {
            "image": [r["image"] for r in dataset_rows],
            "label": [r["label"] for r in dataset_rows],
        }
    )
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.class_encode_column("label")

    split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    return DatasetDict({"train": split["train"], "test": split["test"]})


def upload_to_hf(dataset: DatasetDict, repo_id: str, token: str):
    login(token=token)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    dataset.push_to_hub(repo_id)
    api.upload_file(
        path_or_fileobj=str(LABELS_FILE),
        path_in_repo="labels.txt",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF dataset repo ID")
    parser.add_argument("--token", required=True, help="HF access token")
    args = parser.parse_args()

    dataset = build_dataset()
    print(f"Train: {len(dataset['train'])} images")
    print(f"Test: {len(dataset['test'])} images")
    print(
        f"Member ratio (train): {sum(dataset['train']['label']) / len(dataset['train']):.2f}"
    )
    print(
        f"Member ratio (test): {sum(dataset['test']['label']) / len(dataset['test']):.2f}"
    )

    upload_to_hf(dataset, args.repo_id, args.token)
