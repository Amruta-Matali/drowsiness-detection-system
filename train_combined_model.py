
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

# CEW dataset folders
TRAIN_PATH = "cew_prepared/train"
VAL_PATH = "cew_prepared/val"

# Load CEW training data
train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    class_names=["awake", "sleepy"],
    shuffle=True,
    seed=SEED
)

# Load CEW validation data
val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    class_names=["awake", "sleepy"],
    shuffle=False
)

# Normalize images
normalization = tf.keras.layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_data = val_data.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Load our original MRL-trained model
model = tf.keras.models.load_model("best_eye_model.keras")

# Use a low learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Stop if validation performance stops improving
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Save the best fine-tuned model separately
checkpoint = ModelCheckpoint(
    "best_eye_model_cew_finetuned.keras",
    monitor="val_loss",
    save_best_only=True
)

# Fine-tune
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print("\nCEW fine-tuning completed!")
