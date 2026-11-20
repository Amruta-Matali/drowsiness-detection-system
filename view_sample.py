
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img

classes = ["awake", "sleepy"]

plt.figure(figsize=(10, 6))

for row, category in enumerate(classes):
    folder = os.path.join("dataset", "train", category)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    for col in range(3):
        image_path = os.path.join(folder, images[col])

        img = load_img(image_path)

        plt.subplot(2, 3, row * 3 + col + 1)
        plt.imshow(img)
        plt.title(category)
        plt.axis("off")

plt.tight_layout()
plt.show()

