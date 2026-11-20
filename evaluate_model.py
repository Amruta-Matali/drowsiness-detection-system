
import tensorflow as tf

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

# Normalize test images
normalization_layer = tf.keras.layers.Rescaling(1./255)

test_data = test_data.map(
    lambda x, y: (normalization_layer(x), y)
)

# Load best saved model
model = tf.keras.models.load_model("best_eye_model.keras")

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Evaluation Results")
print("-----------------------")
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))

