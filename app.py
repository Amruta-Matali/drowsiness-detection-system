
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import threading

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="👁️",
    layout="wide"
)

# -------------------------------
# Load CNN Model
# -------------------------------

@st.cache_resource
def load_cnn_model():
    return  tf.keras.models.load_model( "best_eye_model_cew_finetuned.keras")

model = load_cnn_model()

IMAGE_SIZE = (128, 128)

# -------------------------------
# Load Haar Cascades
# -------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# -------------------------------
# Sidebar Navigation
# -------------------------------

st.sidebar.title("Drowsiness Detection")

page = st.sidebar.radio(
    "Navigation",
    ["HOME", "IMAGE", "VIDEO", "CAMERA"]
)

# -------------------------------
# Eye Prediction Function
# -------------------------------

def predict_eye(eye_img):

    eye_img = cv2.resize(
        eye_img,
        IMAGE_SIZE
    )

    eye_img = cv2.cvtColor(
        eye_img,
        cv2.COLOR_GRAY2RGB
    )

    eye_img = eye_img.astype("float32") / 255.0

    eye_img = np.expand_dims(
        eye_img,
        axis=0
    )

    prediction = model.predict(
        eye_img,
        verbose=0
    )[0][0]

    if prediction >= 0.5:
        return "Sleepy", float(prediction)

    return "Awake", float(prediction)

# -------------------------------
# Webcam Video Processor
# -------------------------------

class DrowsinessProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "Waiting for camera"

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        sleepy_detected = False
        eyes_detected = 0

        for (x, y, w, h) in faces:

            cv2.rectangle(
                img,
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

            regions = [
                ("Left Eye", upper_face[:, :width // 2], x, y),
                (
                    "Right Eye",
                    upper_face[:, width // 2:],
                    x + width // 2,
                    y
                )
            ]

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

                    eye_img = region[
                        ey:ey + eh,
                        ex:ex + ew
                    ]

                    label, confidence = predict_eye(eye_img)

                    eyes_detected += 1

                    if label == "Sleepy":
                        color = (0, 0, 255)
                        sleepy_detected = True
                    else:
                        color = (0, 255, 0)

                    cv2.rectangle(
                        img,
                        (offset_x + ex, offset_y + ey),
                        (offset_x + ex + ew, offset_y + ey + eh),
                        color,
                        2
                    )

                    cv2.putText(
                        img,
                        f"{side}: {label}",
                        (offset_x + ex, max(20, offset_y + ey - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        if len(faces) == 0:
            status = "NO FACE DETECTED"
            status_color = (0, 255, 255)

        elif eyes_detected == 0:
            status = "NO EYES DETECTED"
            status_color = (0, 255, 255)

        elif sleepy_detected:
            status = "SLEEPY"
            status_color = (0, 0, 255)

        else:
            status = "AWAKE"
            status_color = (0, 255, 0)

        cv2.putText(
            img,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )

        with self.lock:
            self.status = status

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# -------------------------------
# HOME PAGE
# -------------------------------


if page == "HOME":

    st.title("Drowsiness Detection System")

    st.markdown(
        """
        ### AI-Powered Eye State Monitoring

        This project uses Deep Learning and Computer Vision
        to identify whether a person's eyes are **Awake** or
        **Sleepy** from images, videos, and live webcam input.

        The system combines a Convolutional Neural Network (CNN)
        with OpenCV-based eye detection.
        """
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        """
        The objective is to develop an intelligent monitoring
        system that detects signs of drowsiness by analyzing
        eye states and identifying closed or sleepy eyes.
        """
    )

    st.subheader("Technologies Used")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Language", "Python")
    col2.metric("Deep Learning", "CNN")
    col3.metric("Computer Vision", "OpenCV")
    col4.metric("Interface", "Streamlit")

    st.divider()

    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Input", "128 × 128")
    col2.metric("Classification", "2 Classes")
    col3.metric("CEW Test Accuracy", "76.41%")

    st.caption(
        "The reported accuracy is from the preliminary CEW "
        "test evaluation. Results may vary on real-world images."
    )

    st.write("**Classification Categories:**")
    st.write("🟢 Awake – Eyes are open.")
    st.write("🔴 Sleepy – Eyes are closed or classified as sleepy.")

    st.divider()

    st.subheader("Detection Modes")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### IMAGE")
        st.write("Upload an image and detect eye states.")

    with col2:
        st.markdown("### VIDEO")
        st.write("Analyze uploaded videos frame by frame.")

    with col3:
        st.markdown("### CAMERA")
        st.write("Perform real-time detection using a webcam.")

    st.divider()


# -------------------------------
# IMAGE DETECTION
# -------------------------------


# -------------------------------
# IMAGE DETECTION
# -------------------------------

elif page == "IMAGE":

    st.title("Image-Based Drowsiness Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        st.subheader("Uploaded Image")

        st.image(
            image_array,
            caption="Input Image",
            width=450
        )

        if st.button("Detect Eye State"):

            with st.spinner("Analyzing image..."):

                frame = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(100, 100)
                )

                result_frame = frame.copy()
                detected_eyes = []

                for (x, y, w, h) in faces:

                    cv2.rectangle(
                        result_frame,
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
                        ("Left Eye", left_region, x, y),
                        ("Right Eye", right_region, x + width // 2, y)
                    ]

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

                            eye_img = region[
                                ey:ey + eh,
                                ex:ex + ew
                            ]

                            label, confidence = predict_eye(eye_img)

                            color = (
                                (0, 0, 255)
                                if label == "Sleepy"
                                else (0, 255, 0)
                            )

                            cv2.rectangle(
                                result_frame,
                                (offset_x + ex, offset_y + ey),
                                (offset_x + ex + ew, offset_y + ey + eh),
                                color,
                                2
                            )

                            detected_eyes.append(label)

                # -----------------------
                # DISPLAY RESULT
                # -----------------------

                st.divider()
                st.subheader("Detection Result")

                result_rgb = cv2.cvtColor(
                    result_frame,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    result_rgb,
                    caption="Processed Image",
                    width=450
                )

                if len(faces) == 0:

                    st.warning(
                        "No face detected. Trying direct eye-image classification."
                    )

                    full_eye = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY
                    )

                    label, confidence = predict_eye(full_eye)

                    if label == "Sleepy":
                        st.error("🔴 SLEEPY")
                    else:
                        st.success("🟢 AWAKE")

                elif len(detected_eyes) == 0:

                    st.warning("No eyes detected. Please try another image.")

                else:

                    if "Sleepy" in detected_eyes:
                        st.error("🔴 SLEEPY – Drowsiness detected.")
                    else:
                        st.success("🟢 AWAKE – No drowsiness detected.")

# -------------------------------
# VIDEO PAGE
# -------------------------------

elif page == "VIDEO":

    st.title("🎥 Video-Based Drowsiness Detection")

    st.write(
        "Upload a video to detect eye states frame by frame."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video, width=550)

        if st.button("Start Video Detection"):

            with st.spinner("Processing video... Please wait."):

                # Save uploaded video temporarily
                input_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                input_temp.write(uploaded_video.getvalue())
                input_temp.close()

                # Open video
                cap = cv2.VideoCapture(input_temp.name)

                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps <= 0:
                    fps = 25.0

                frame_width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                frame_height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                # Create output video
                output_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_temp.close()

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    output_temp.name,
                    fourcc,
                    fps,
                    (frame_width, frame_height)
                )

                progress_bar = st.progress(0)

                frame_count = 0
                sleepy_frames = 0
                awake_frames = 0
                no_face_frames = 0

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    gray = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY
                    )

                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(100, 100)
                    )

                    frame_sleepy = False

                    for (x, y, w, h) in faces:

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (255, 0, 0),
                            2
                        )

                        # Upper face region
                        upper_face = gray[
                            y:y + int(h * 0.60),
                            x:x + w
                        ]

                        height, width = upper_face.shape

                        left_region = upper_face[
                            :, :width // 2
                        ]

                        right_region = upper_face[
                            :, width // 2:
                        ]

                        regions = [
                            (
                                "Left Eye",
                                left_region,
                                x,
                                y
                            ),
                            (
                                "Right Eye",
                                right_region,
                                x + width // 2,
                                y
                            )
                        ]

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

                                eye_img = region[
                                    ey:ey + eh,
                                    ex:ex + ew
                                ]

                                label, confidence = predict_eye(
                                    eye_img
                                )

                                if label == "Sleepy":

                                    color = (0, 0, 255)
                                    frame_sleepy = True

                                else:

                                    color = (0, 255, 0)

                                cv2.rectangle(
                                    frame,
                                    (
                                        offset_x + ex,
                                        offset_y + ey
                                    ),
                                    (
                                        offset_x + ex + ew,
                                        offset_y + ey + eh
                                    ),
                                    color,
                                    2
                                )

                                cv2.putText(
                                    frame,
                                    f"{side}: {label}",
                                    (
                                        offset_x + ex,
                                        max(20, offset_y + ey - 8)
                                    ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2
                                )

                    # Overall frame status
                    if frame_sleepy:

                        sleepy_frames += 1

                        status = "SLEEPY"
                        status_color = (0, 0, 255)

                    elif len(faces) > 0:

                        awake_frames += 1

                        status = "AWAKE"
                        status_color = (0, 255, 0)

                    else:

                        no_face_frames += 1

                        status = "NO FACE"
                        status_color = (0, 255, 255)

                    cv2.putText(
                        frame,
                        status,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        status_color,
                        2
                    )

                    writer.write(frame)

                    frame_count += 1

                    if total_frames > 0:
                        progress_bar.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                # Release video resources
                cap.release()
                writer.release()

                progress_bar.progress(1.0)

                st.success("Video processing completed!")

                # Display processed video
                st.subheader("Processed Video")

                st.video(output_temp.name, width=550)

                # Display summary
                st.subheader("Detection Summary")

                st.write("Total Frames Processed:", frame_count)

                st.write(
                    "Frames with at least one sleepy eye:",
                    sleepy_frames
                )

                if frame_count > 0:

                    sleepy_percentage = (
                        sleepy_frames / frame_count
                    ) * 100

                    st.write(
                        "Sleepy Frame Percentage:",
                        f"{sleepy_percentage:.2f}%"
                    )

                # Download processed video
                with open(output_temp.name, "rb") as video_file:

                    st.download_button(
                        label="Download Processed Video",
                        data=video_file,
                        file_name="drowsiness_detection_result.mp4",
                        mime="video/mp4"
                    )

                # Remove temporary input file
                import os
                os.remove(input_temp.name)



# -------------------------------
# CAMERA PAGE
# -------------------------------


# -------------------------------
# CAMERA PAGE
# -------------------------------

elif page == "CAMERA":

    st.title("📷 Live Webcam Drowsiness Detection")

    st.write(
        "Monitor eye states in real time using your webcam "
        "and the trained CNN model."
    )

    st.info(
        "Click START to activate your webcam. "
        "Allow camera permission when your browser asks."
    )

    st.divider()

    # Center the webcam display
    left_col, center_col, right_col = st.columns([1, 3, 1])

    with center_col:

        ctx = webrtc_streamer(
            key="drowsiness-camera",
            video_processor_factory=DrowsinessProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

    st.divider()

    st.subheader("Live Detection Status")

    if ctx.video_processor:

        with ctx.video_processor.lock:
            current_status = ctx.video_processor.status

        if current_status == "AWAKE":

            st.success("AWAKE - Eyes detected as open.")

        elif current_status == "SLEEPY":

            st.error("SLEEPY - Drowsiness detected.")

        elif current_status == "NO FACE DETECTED":

            st.warning("No face detected. Adjust your position.")

        elif current_status == "NO EYES DETECTED":

            st.warning(" No eyes detected. Please face the camera.")

        else:

            st.info("Waiting for camera detection...")

    else:

        st.info("Start the webcam to begin live detection.")

    st.caption(
        "Note: This is an educational demonstration, "
        "not a certified driver safety system."
    )


