
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
TEST_PATH = "cew_prepared/test"

# Load CEW test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    class_names=["awake", "sleepy"],
    shuffle=False
)

# Normalize images
normalization = tf.keras.layers.Rescaling(1.0 / 255)

test_data = test_data.map(
    lambda x, y: (normalization(x), y)
)

# Get actual labels
y_true = []
for images, labels in test_data:
    y_true.extend(labels.numpy().flatten())

y_true = [int(label) for label in y_true]

# Evaluate both models
model_paths = [
    "best_eye_model.keras",
    "best_eye_model_cew_finetuned.keras"
]

for model_path in model_paths:

    print("\n" + "=" * 55)
    print("Evaluating:", model_path)
    print("=" * 55)

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(test_data, verbose=1)

    # Sigmoid probability >= 0.5 means sleepy
    y_pred = (predictions.flatten() >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Awake", "Sleepy"]
        )
    )

print("\nBoth model evaluations completed!")
