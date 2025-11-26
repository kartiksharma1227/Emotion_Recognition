# Emotion Recognition from Gait and Pose

A machine learning project that classifies human emotions (Angry, Happy, Neutral, Sad) from body pose features extracted using MediaPipe. The system uses a Random Forest classifier trained on gait and pose data to predict emotions in real-time from video input.

## Overview

This project analyzes body movement patterns and pose landmarks to recognize emotional states. It extracts 48 features from 16 key body landmarks (nose, shoulders, elbows, wrists, hips, knees, ankles, feet) and uses machine learning to classify emotions.

## Features

- **Pose Feature Extraction**: Extracts 48 features (x, y, z coordinates) from 16 body landmarks using MediaPipe
- **Multi-class Classification**: Classifies 4 emotion categories (Angry, Happy, Neutral, Sad)
- **Real-time Video Processing**: Processes video frames and displays emotion predictions
- **High Accuracy**: Achieves ~89% classification accuracy on test data

## Project Structure

```
emotion-recognition-project/
├── Emotion_Recognition.ipynb       # Main notebook with complete pipeline
├── config.py                       # Configuration and constants
├── data_loader.py                  # H5 data loading utilities
├── feature_extractor.py            # MediaPipe pose feature extraction
├── train_model.py                  # Model training and evaluation
├── inference.py                    # Video inference and webcam support
├── main.py                         # Main entry point (CLI)
├── requirements.txt                # Python dependencies
├── data/                           # Training data (H5 files)
│   └── *.h5                        # Pose feature datasets
├── testing.mp4                     # Test video for inference
├── emotion_classifier_rf.pkl       # Trained Random Forest model (generated)
├── scaler.pkl                      # Feature scaler (generated)
└── README.md                       # This file
```

### Module Descriptions

- **`config.py`**: Central configuration file with emotion labels, landmark indices, file paths, and hyperparameters
- **`data_loader.py`**: Functions to load and process H5 files containing pose features
- **`feature_extractor.py`**: MediaPipe-based pose landmark detection and feature extraction
- **`train_model.py`**: Random Forest classifier training, evaluation, and model persistence
- **`inference.py`**: Video processing and real-time emotion prediction with webcam support
- **`main.py`**: Command-line interface to orchestrate training and inference workflows

## Requirements

### Python Packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision torchaudio
pip install mediapipe opencv-python h5py
pip install scikit-learn joblib numpy pandas
```

### System Requirements

- Python 3.8+
- CUDA support (optional, for GPU acceleration)
- Webcam or video file for inference

## Installation

### Option 1: Google Colab (Recommended)

1. Upload `Emotion_Recognition.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload your `data/` folder and `testing.mp4`
3. Run all cells sequentially

### Option 2: Local Environment

```bash
# Clone or download the project
cd emotion-recognition-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install torch torchvision torchaudio
pip install mediapipe opencv-python h5py scikit-learn joblib numpy pandas
```

## Usage

### Using Modular Python Code (Recommended)

#### 1. Prepare Data

Place your H5 files in the `data/` folder. Each H5 file should contain datasets with:

- Shape: `(n_samples, 48)` - 48 features per sample
- Names containing emotion labels: `Angry`, `Happy`, `Neutral`, `Sad` (or `_an_`, `_ha_`, `_nu_`, `_sa_`)

#### 2. Train the Model

```bash
# Train model and save to disk
python main.py --mode train
```

This will:

- Load all H5 files from `data/` folder
- Split data into train/test sets
- Train Random Forest classifier
- Evaluate performance
- Save `emotion_classifier_rf.pkl` and `scaler.pkl`

#### 3. Run Inference on Video

```bash
# Process video file
python main.py --mode inference --video testing.mp4

# Save output video
python main.py --mode inference --video input.mp4 --output result.mp4

# Disable display (useful for servers)
python main.py --mode inference --video input.mp4 --no-display
```

#### 4. Real-time Webcam Detection

```bash
# Use webcam for live emotion detection
python main.py --mode webcam
```

#### 5. Full Pipeline (Train + Inference)

```bash
# Train and run inference in one command
python main.py --mode all --video testing.mp4
```

### Using Jupyter Notebook

#### 1. Train the Model

Run cells 1-7 in the notebook:

```python
# Cell 1: Install dependencies (if needed)
# Cell 2: Load H5 data
# Cell 3: Split and scale data
# Cell 4: Train Random Forest classifier
# Cell 5: Save model
# Cell 7: Save scaler
```

#### 2. Run Inference

**For Colab:**

```python
# Cell 10: Run as-is with cv2_imshow
```

**For Local:**

Modify cell 10 to use standard OpenCV:

```python
import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load('emotion_classifier_rf.pkl')
scaler = joblib.load('scaler.pkl')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_features(results):
    features = []
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
                features.extend([lm.x, lm.y, lm.z])
    if len(features) != 48:
        return None
    return np.array(features).reshape(1, -1)

video_path = 'testing.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    features = extract_features(results)

    if features is not None:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        # Map prediction to emotion label
        emotions = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad'}
        emotion_label = emotions.get(prediction, 'Unknown')

        cv2.putText(frame, f'Emotion: {emotion_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("Video processing done.")
```

## Model Details

### Architecture

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Features**: 48 (16 landmarks × 3 coordinates)
- **Classes**: 4 emotions

### Landmarks Used (MediaPipe Pose)

```
0:  Nose
11-12: Shoulders
13-14: Elbows
15-16: Wrists
23-24: Hips
25-26: Knees
27-28: Ankles
29-30: Heels
31-32: Foot indices
```

### Performance Metrics

- **Accuracy**: ~89%
- **Classes**: Balanced classification across 4 emotions

## Emotion Labels

| Label | Emotion | Code   |
| ----- | ------- | ------ |
| 0     | Angry   | `_an_` |
| 1     | Happy   | `_ha_` |
| 2     | Neutral | `_nu_` |
| 3     | Sad     | `_sa_` |

## Troubleshooting

### "No data read from H5 files!"

- Ensure `.h5` files are in the `data/` folder
- Check that datasets have shape `(n, 48)`
- Verify dataset names contain emotion labels

### "cv2_imshow not found"

- Replace with `cv2.imshow()` for local environments
- `cv2_imshow` is Colab-specific

### Low Accuracy

- Ensure proper data preprocessing
- Check that scaler is fitted on training data
- Verify feature extraction matches training pipeline

## Dataset

This project uses the **ELMD (Emotion Locomotion Motion Dataset)** or similar gait-based emotion datasets. Features should be pre-extracted as 48-dimensional vectors representing body pose landmarks.

## Code Organization

The codebase follows a modular architecture for easy maintenance and extension:

1. **Configuration (`config.py`)**: All constants, paths, and hyperparameters in one place
2. **Data Pipeline (`data_loader.py`)**: Reusable functions for loading H5 files
3. **Feature Extraction (`feature_extractor.py`)**: MediaPipe wrapper with context manager support
4. **Model Training (`train_model.py`)**: Classifier class with train/evaluate/save methods
5. **Inference (`inference.py`)**: Video and webcam processing with progress tracking
6. **CLI (`main.py`)**: Command-line interface with argument parsing

### Key Features of Modular Code

- **Type Hints**: All functions have type annotations for clarity
- **Docstrings**: Comprehensive documentation for all classes and functions
- **Error Handling**: Proper validation and error messages
- **Context Managers**: Automatic resource cleanup (e.g., MediaPipe pose)
- **Progress Tracking**: Real-time feedback during long operations
- **Flexible CLI**: Easy-to-use command-line interface

## Quick Start Examples

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python main.py --mode train

# Test on video
python main.py --mode inference --video testing.mp4

# Live webcam
python main.py --mode webcam
```


## Acknowledgments

- MediaPipe for pose estimation
- Scikit-learn for machine learning tools
- ELMD dataset contributors
