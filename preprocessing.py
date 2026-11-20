
import tensorflow as tf

# Dataset paths
train_path = "dataset/train"
val_path = "dataset/val"
test_path = "dataset/test"

# Image parameters
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=True,
    seed=42
)

# Load validation dataset
val_data = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

# Load testing dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_data = train_data.map(
    lambda x, y: (normalization_layer(x), y)
)

val_data = val_data.map(
    lambda x, y: (normalization_layer(x), y)
)

test_data = test_data.map(
    lambda x, y: (normalization_layer(x), y)
)

# Display class names
print("Class Names:", train_data.class_names if hasattr(train_data, "class_names") else ["awake", "sleepy"])

print("Image Size:", IMAGE_SIZE)
print("Batch Size:", BATCH_SIZE)

print("Preprocessing Completed Successfully!")



