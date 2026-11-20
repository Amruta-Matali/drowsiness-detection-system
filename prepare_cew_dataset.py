
import os
import shutil
import random
from pathlib import Path
from PIL import Image

# Project paths
BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = (
    BASE_DIR
    / "external_datasets"
    / "CEW"
    / "dataset_B_Eye_Images"
)

OUTPUT_DIR = BASE_DIR / "cew_prepared"

IMAGE_SIZE = (128, 128)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Source folders and labels
categories = {
    "awake": [
        "openLeftEyes",
        "openRightEyes"
    ],
    "sleepy": [
        "closedLeftEyes",
        "closedRightEyes"
    ]
}

# Split percentages
SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

# Supported image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Check source directory
if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {SOURCE_DIR}"
    )

# Prepare output directories
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in SPLITS:
    for label in categories:
        (OUTPUT_DIR / split / label).mkdir(
            parents=True,
            exist_ok=True
        )

# Collect images
for label, folders in categories.items():

    images = []

    for folder_name in folders:

        folder_path = SOURCE_DIR / folder_name

        if not folder_path.exists():
            raise FileNotFoundError(folder_path)

        images.extend([
            path for path in folder_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ])

    random.shuffle(images)

    total = len(images)

    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_images = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    print(f"\nProcessing {label}: {total} images")

    for split, files in split_images.items():

        for index, image_path in enumerate(files):

            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(IMAGE_SIZE)

                    output_name = (
                        f"{label}_{index}_{image_path.stem}.jpg"
                    )

                    output_path = (
                        OUTPUT_DIR
                        / split
                        / label
                        / output_name
                    )

                    img.save(output_path, quality=95)

            except Exception as error:
                print(
                    f"Skipping {image_path.name}: {error}"
                )

        print(f"{split}: {len(files)} images")

print("\nCEW dataset preparation completed!")
print(f"Output location: {OUTPUT_DIR}")
