
import os

dataset_path = "dataset"

folders = ["train", "val", "test"]
categories = ["awake", "sleepy"]

for folder in folders:

    print("\n", folder.upper(), "DATASET")

    for category in categories:

        path = os.path.join(dataset_path, folder, category)

        images = os.listdir(path)

        print(category, ":", len(images), "images")
