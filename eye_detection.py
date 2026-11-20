
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot access webcam!")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame, (x, y), (x+w, y+h), (255, 0, 0), 2
        )

        # Upper face region
        upper_face = gray[
            y:y + int(h * 0.60),
            x:x+w
        ]

        height, width = upper_face.shape

        # Divide upper face into two regions
        left_region = upper_face[:, :width//2]
        right_region = upper_face[:, width//2:]

        regions = [
            (left_region, x, y),
            (right_region, x + width//2, y)
        ]

        total_eyes = 0

        for region, offset_x, offset_y in regions:

            eyes = eye_cascade.detectMultiScale(
                region,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )

            if len(eyes) > 0:
                # Select largest detection in each region
                ex, ey, ew, eh = max(
                    eyes, key=lambda e: e[2] * e[3]
                )

                cv2.rectangle(
                    frame,
                    (offset_x + ex, offset_y + ey),
                    (offset_x + ex + ew, offset_y + ey + eh),
                    (0, 255, 0),
                    2
                )

                total_eyes += 1

        cv2.putText(
            frame,
            "Eyes Detected: " + str(total_eyes),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
