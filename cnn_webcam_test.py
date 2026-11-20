
import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from collections import deque

# -------------------------------
# 1. Load Haar Cascade Classifiers
# -------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if face_cascade.empty() or eye_cascade.empty():
    print("Error loading Haar Cascade files!")
    exit()

# -------------------------------
# 2. Load CNN Model
# -------------------------------

model = tf.keras.models.load_model("best_eye_model.keras")

IMAGE_SIZE = (128, 128)

print("CNN Model Loaded Successfully!")

# -------------------------------
# 3. Screenshot Folder
# -------------------------------

os.makedirs("captured_images", exist_ok=True)

# -------------------------------
# 4. Webcam Initialization
# -------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot access webcam!")
    exit()

# -------------------------------
# 5. Prediction Smoothing
# -------------------------------

left_predictions = deque(maxlen=5)
right_predictions = deque(maxlen=5)

sleepy_confirmed = False

print("Webcam Starting...")

print("\nControls:")
print("S - Capture Screenshot")
print("Q - Quit Application")


# -------------------------------
# 6. Function to Predict Eye
# -------------------------------

def predict_eye(region, eye_box):

    ex, ey, ew, eh = eye_box

    eye_img = region[ey:ey + eh, ex:ex + ew]

    eye_img = cv2.resize(eye_img, IMAGE_SIZE)

    eye_img = cv2.cvtColor(
        eye_img, cv2.COLOR_GRAY2RGB
    )

    eye_img = eye_img.astype("float32") / 255.0

    eye_img = np.expand_dims(eye_img, axis=0)

    prediction = model.predict(
        eye_img, verbose=0
    )[0][0]

    return float(prediction)


# -------------------------------
# 7. Main Webcam Loop
# -------------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame!")
        break

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    left_status = "Not Detected"
    right_status = "Not Detected"

    left_value = None
    right_value = None

    # ---------------------------
    # 8. Face Detection
    # ---------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        upper_face = gray[
            y:y + int(h * 0.60),
            x:x + w
        ]

        height, width = upper_face.shape

        left_region = upper_face[:, :width // 2]

        right_region = upper_face[:, width // 2:]

        regions = [
            ("Left", left_region, x, y),
            ("Right", right_region, x + width // 2, y)
        ]

        # -----------------------
        # 9. Detect Both Eyes
        # -----------------------

        for side, region, offset_x, offset_y in regions:

            eyes = eye_cascade.detectMultiScale(
                region,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )

            if len(eyes) > 0:

                ex, ey, ew, eh = max(
                    eyes,
                    key=lambda e: e[2] * e[3]
                )

                prediction = predict_eye(
                    region,
                    (ex, ey, ew, eh)
                )

                if side == "Left":

                    left_predictions.append(prediction)

                    left_value = float(
                        np.mean(left_predictions)
                    )

                    color = (
                        (0, 0, 255)
                        if left_value >= 0.5
                        else (0, 255, 0)
                    )

                    left_status = (
                        "Sleepy"
                        if left_value >= 0.5
                        else "Awake"
                    )

                else:

                    right_predictions.append(prediction)

                    right_value = float(
                        np.mean(right_predictions)
                    )

                    color = (
                        (0, 0, 255)
                        if right_value >= 0.5
                        else (0, 255, 0)
                    )

                    right_status = (
                        "Sleepy"
                        if right_value >= 0.5
                        else "Awake"
                    )

                # Draw eye rectangle

                cv2.rectangle(
                    frame,
                    (offset_x + ex, offset_y + ey),
                    (offset_x + ex + ew, offset_y + ey + eh),
                    color,
                    2
                )

    # -------------------------------
    # 10. Overall Drowsiness Status
    # -------------------------------

    available_values = [
        value for value in [left_value, right_value]
        if value is not None
    ]

    # Require at least one eye prediction.
    # An undetected eye is not treated as sleepy.

    if available_values:

        sleepy_count = sum(
            value >= 0.5 for value in available_values
        )

        if sleepy_count > 0:

            status = "SLEEPY EYE DETECTED"
            status_color = (0, 0, 255)

            if not sleepy_confirmed:

                filename = datetime.now().strftime(
                    "captured_images/sleepy_%Y%m%d_%H%M%S_%f.jpg"
                )

                cv2.imwrite(filename, frame)

                print("Sleepy screenshot saved:", filename)

            sleepy_confirmed = True

        else:

            status = "AWAKE"
            status_color = (0, 255, 0)

            sleepy_confirmed = False

    else:

        status = "EYES NOT DETECTED"
        status_color = (0, 255, 255)

        sleepy_confirmed = False

    # -------------------------------
    # 11. Display Eye Predictions
    # -------------------------------

    cv2.putText(
        frame,
        "Left Eye: " + left_status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Right Eye: " + right_status,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    cv2.putText(
        frame,
        "S: Screenshot | Q: Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # -------------------------------
    # 12. Display Webcam
    # -------------------------------

    cv2.imshow(
        "Drowsiness Detection - CNN",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Manual Screenshot

    if key == ord('s'):

        filename = datetime.now().strftime(
            "captured_images/manual_%Y%m%d_%H%M%S_%f.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Manual screenshot saved:", filename)

    # Quit

    if key == ord('q'):
        break


# -------------------------------
# 13. Release Resources
# -------------------------------

camera.release()

cv2.destroyAllWindows()

print("\nWebcam Closed Successfully!")
