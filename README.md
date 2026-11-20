# AI-Based Driver Drowsiness Detection System

An AI-powered computer vision application that detects eye states and identifies possible drowsiness using a Convolutional Neural Network (CNN), OpenCV, and Streamlit.

## Project Overview

Driver drowsiness can reduce attention and reaction time. This project demonstrates an image-processing and deep-learning approach to identifying whether a person's eyes appear **Awake** or **Sleepy**.

The application supports image uploads, video analysis, and real-time webcam detection through an interactive Streamlit interface.

## Features

- **Home:** Project overview, model information, and detection modes.
- **Image Detection:** Upload an image and detect eye states.
- **Video Detection:** Analyse an uploaded video and display detection statistics.
- **Live Webcam Detection:** Detect eye states in real time using a webcam.
- **CNN Classification:** Classify detected eye regions as Awake or Sleepy.
- **Visual Detection:** Display face and eye detection results using OpenCV.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| TensorFlow / Keras | CNN model training and prediction |
| OpenCV | Face and eye detection |
| NumPy | Image processing |
| Streamlit | Interactive web application |
| Streamlit-WebRTC | Real-time webcam streaming |
| Pillow | Image handling |

## Dataset Information

This project uses two eye-image datasets:

1. **MRL Eye Dataset:** Used to train the initial CNN model.
2. **Closed Eyes in the Wild (CEW):** Used to fine-tune the initial model and improve performance on visible-light eye images.

The datasets were organised into two classes:

- `awake`
- `sleepy`

The CEW dataset was divided into training, validation, and testing subsets.

Note: Download and prepare the datasets separately. They are excluded from this repository.

## Model Architecture

A Convolutional Neural Network (CNN) is used for binary eye-state classification.

**Model configuration:**
- Input image size: 128 × 128 pixels
- Input channels: 3 (RGB)
- Classes: Awake and Sleepy
- Convolutional layers with MaxPooling
- Global Average Pooling
- Dense layer with Dropout
- Sigmoid output layer

The initial model was trained using the MRL dataset and subsequently fine-tuned using CEW eye images.

**Final model:** `best_eye_model_cew_finetuned.keras`

## Model Evaluation

The fine-tuned model was evaluated using the CEW test dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | 76.41% |
| Awake Precision | 75% |
| Awake Recall | 81% |
| Sleepy Precision | 78% |
| Sleepy Recall | 72% |

The fine-tuned model improved test accuracy compared with the original MRL-trained model.

These results are preliminary. The CEW split was performed at image level, so images from the same individuals may appear in multiple subsets. The results should not be interpreted as proof of generalisation to unseen drivers or real-world driving conditions.

## Project Structure

```text
drowsiness_detection/
│
├── app.py
├── best_eye_model.keras
├── best_eye_model_cew_finetuned.keras
├── requirements.txt
├── .gitignore
├── README.md
│
├── dataset/
├── cew_prepared/
├── external_datasets/
├── captured_images/
│
├── train_model.py
├── train_combined_model.py
├── evaluate_model.py
├── evaluate_cew_model.py
├── prepare_cew_dataset.py
├── preprocessing.py
├── eye_detection.py
│
└── Other testing and utility scripts
```

Dataset folders and captured images are excluded from Git tracking.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd drowsiness_detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use

1. Open the Streamlit application.
2. Select a detection mode from the sidebar.
3. For image detection, upload an image.
4. For video detection, upload a video for analysis.
5. For webcam detection, allow camera access and start the webcam.
6. Observe the detected eye state and analysis results.

## Limitations

- Haar Cascade detection may not consistently detect eyes under all lighting conditions or face angles.
- Model performance may vary across different cameras, environments, and individuals.
- The model can produce false positive and false negative predictions.
- The CEW test evaluation is preliminary and has possible subject overlap between subsets.

## Future Improvements

- Improve eye detection under different lighting conditions.
- Explore more diverse datasets.
- Improve model generalisation to real-world webcam images.
- Add temporal analysis to identify sustained eye closure.
- Explore more advanced face and eye detection techniques.

## Disclaimer

This project is developed for educational and portfolio demonstration purposes only. It is not a certified driver-monitoring or vehicle safety system and must not be used as the sole basis for driving-safety decisions.
