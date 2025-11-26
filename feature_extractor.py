"""
Feature extraction module using MediaPipe for pose landmark detection.
"""

import numpy as np
import mediapipe as mp
from typing import Optional
from config import POSE_LANDMARK_INDICES, NUM_FEATURES, MIN_DETECTION_CONFIDENCE, STATIC_IMAGE_MODE


class PoseFeatureExtractor:
    """
    Extracts pose features from video frames using MediaPipe Pose.
    """
    
    def __init__(
        self,
        static_image_mode: bool = STATIC_IMAGE_MODE,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE
    ):
        """
        Initialize the MediaPipe Pose detector.
        
        Args:
            static_image_mode (bool): Whether to treat images as static (False for video)
            min_detection_confidence (float): Minimum confidence for pose detection
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence
        )
    
    def extract_features_from_landmarks(self, pose_results) -> Optional[np.ndarray]:
        """
        Extract 48 features from MediaPipe pose landmarks.
        
        Features consist of x, y, z coordinates for 16 specific body landmarks:
        - Nose (0)
        - Shoulders (11, 12)
        - Elbows (13, 14)
        - Wrists (15, 16)
        - Hips (23, 24)
        - Knees (25, 26)
        - Ankles (27, 28)
        - Heels (29, 30)
        - Foot indices (31, 32)
        
        Args:
            pose_results: MediaPipe pose detection results
            
        Returns:
            Optional[np.ndarray]: Array of shape (1, 48) containing features, or None if landmarks not detected
        """
        features = []
        
        if pose_results.pose_landmarks:
            for landmark_index, landmark in enumerate(pose_results.pose_landmarks.landmark):
                if landmark_index in POSE_LANDMARK_INDICES:
                    features.extend([landmark.x, landmark.y, landmark.z])
        
        # Validate feature count
        if len(features) != NUM_FEATURES:
            return None
        
        return np.array(features).reshape(1, -1)
    
    def process_frame(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single RGB frame and extract pose features.
        
        Args:
            rgb_frame (np.ndarray): RGB image frame
            
        Returns:
            Optional[np.ndarray]: Extracted features of shape (1, 48), or None if pose not detected
        """
        pose_results = self.pose.process(rgb_frame)
        return self.extract_features_from_landmarks(pose_results)
    
    def close(self):
        """
        Release MediaPipe resources.
        """
        self.pose.close()
    
    def __enter__(self):
        """
        Context manager entry.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - automatically closes resources.
        """
        self.close()


if __name__ == "__main__":
    # Test the feature extractor
    print("Testing PoseFeatureExtractor...")
    
    with PoseFeatureExtractor() as extractor:
        # Create a dummy RGB frame (480x640x3)
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        features = extractor.process_frame(dummy_frame)
        
        if features is not None:
            print(f"✓ Feature extraction successful!")
            print(f"Feature shape: {features.shape}")
            print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
        else:
            print("✗ No pose detected in dummy frame (expected for random data)")
