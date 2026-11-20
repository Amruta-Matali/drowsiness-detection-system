
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model("best_eye_model.keras")

IMAGE_SIZE = (128, 128)

# Test original dataset images
for category in ["awake", "sleepy"]:

    folder = os.path.join("dataset", "test", category)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print("\nActual Class:", category.upper())
    print("------------------------------")

    for filename in images[:5]:

        image_path = os.path.join(folder, filename)

        img = load_img(
            image_path,
            target_size=IMAGE_SIZE,
            color_mode="rgb"
        )

        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(
            img_array,
            verbose=0
        )[0][0]

        predicted_class = (
            "SLEEPY" if prediction >= 0.5 else "AWAKE"
        )

        print("Image:", filename)
        print("Prediction:", predicted_class)
        print("Sleepy probability:", round(float(prediction), 4))
        print()
