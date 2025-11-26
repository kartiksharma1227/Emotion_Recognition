"""
Configuration file containing constants, paths, and emotion labels.
"""

import os

# Emotion labels mapping
EMOTION_LABELS = {
    0: 'Angry',
    1: 'Happy',
    2: 'Neutral',
    3: 'Sad'
}

# Emotion keywords for dataset name matching
EMOTION_KEYWORDS = {
    0: ['Angry', '_an_'],
    1: ['Happy', '_ha_'],
    2: ['Neutral', '_nu_'],
    3: ['Sad', '_sa_']
}

# MediaPipe pose landmark indices to use for feature extraction
# 0: Nose, 11-12: Shoulders, 13-14: Elbows, 15-16: Wrists,
# 23-24: Hips, 25-26: Knees, 27-28: Ankles, 29-30: Heels, 31-32: Foot indices
POSE_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# Number of features per sample (16 landmarks * 3 coordinates: x, y, z)
NUM_FEATURES = len(POSE_LANDMARK_INDICES) * 3  # 48 features

# File paths
DATA_DIR = './data'
MODEL_SAVE_PATH = 'emotion_classifier_rf.pkl'
SCALER_SAVE_PATH = 'scaler.pkl'
TEST_VIDEO_PATH = 'testing.mp4'

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = None

# MediaPipe configuration
MIN_DETECTION_CONFIDENCE = 0.5
STATIC_IMAGE_MODE = False

# Video processing
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_COLOR = (0, 255, 0)  # Green
FONT_THICKNESS = 2
TEXT_POSITION = (10, 30)
