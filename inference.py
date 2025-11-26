"""
Inference module for real-time emotion prediction from video.
"""

import cv2
import numpy as np
from typing import Optional
from feature_extractor import PoseFeatureExtractor
from train_model import EmotionClassifier
from config import (
    EMOTION_LABELS,
    TEST_VIDEO_PATH,
    MODEL_SAVE_PATH,
    SCALER_SAVE_PATH,
    FONT,
    FONT_SCALE,
    FONT_COLOR,
    FONT_THICKNESS,
    TEXT_POSITION
)


class EmotionVideoProcessor:
    """
    Process video frames to detect and classify emotions in real-time.
    """
    
    def __init__(
        self,
        model_path: str = MODEL_SAVE_PATH,
        scaler_path: str = SCALER_SAVE_PATH
    ):
        """
        Initialize the video processor with a trained model.
        
        Args:
            model_path (str): Path to the trained model
            scaler_path (str): Path to the saved scaler
        """
        # Load trained model
        self.classifier = EmotionClassifier()
        self.classifier.load_model(model_path, scaler_path)
        
        # Initialize feature extractor
        self.feature_extractor = PoseFeatureExtractor()
        
        print("✓ EmotionVideoProcessor initialized")
    
    def predict_emotion(self, features: np.ndarray) -> tuple[int, str]:
        """
        Predict emotion from extracted features.
        
        Args:
            features (np.ndarray): Feature vector of shape (1, 48)
            
        Returns:
            tuple[int, str]: Predicted emotion label and emotion name
        """
        emotion_label = self.classifier.predict(features)
        emotion_name = EMOTION_LABELS.get(emotion_label, 'Unknown')
        return emotion_label, emotion_name
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        emotion_name: str,
        emotion_label: int
    ) -> np.ndarray:
        """
        Add emotion prediction text overlay to the frame.
        
        Args:
            frame (np.ndarray): Video frame (BGR format)
            emotion_name (str): Name of the predicted emotion
            emotion_label (int): Numeric label of the emotion
            
        Returns:
            np.ndarray: Annotated frame
        """
        text = f'Emotion: {emotion_name} ({emotion_label})'
        
        cv2.putText(
            frame,
            text,
            TEXT_POSITION,
            FONT,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS
        )
        
        return frame
    
    def process_video(
        self,
        video_path: str = TEST_VIDEO_PATH,
        display: bool = True,
        save_output: Optional[str] = None
    ) -> None:
        """
        Process a video file and predict emotions frame by frame.
        
        Args:
            video_path (str): Path to the input video file
            display (bool): Whether to display the video (True for local, False for Colab)
            save_output (Optional[str]): Path to save output video (None to skip saving)
        """
        # Open video capture
        video_capture = cv2.VideoCapture(video_path)
        
        if not video_capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Initialize video writer if saving output
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"Saving output to: {save_output}")
        
        frame_count = 0
        detected_count = 0
        
        try:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract pose features
                features = self.feature_extractor.process_frame(rgb_frame)
                
                if features is not None:
                    detected_count += 1
                    
                    # Predict emotion
                    emotion_label, emotion_name = self.predict_emotion(features)
                    
                    # Annotate frame
                    frame = self.annotate_frame(frame, emotion_name, emotion_label)
                else:
                    # No pose detected
                    cv2.putText(
                        frame,
                        'No pose detected',
                        TEXT_POSITION,
                        FONT,
                        FONT_SCALE,
                        (0, 0, 255),  # Red color
                        FONT_THICKNESS
                    )
                
                # Save frame if output path provided
                if video_writer:
                    video_writer.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Emotion Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nVideo processing interrupted by user")
                        break
                
                # Progress indicator
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
        
        finally:
            # Clean up
            video_capture.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            self.feature_extractor.close()
        
        print(f"\n\n✓ Video processing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Poses detected: {detected_count} ({detected_count/frame_count*100:.1f}%)")
    
    def process_webcam(self) -> None:
        """
        Process live webcam feed for real-time emotion detection.
        """
        # Open webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            raise RuntimeError("Failed to open webcam")
        
        print("\nStarting webcam emotion detection...")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract pose features
                features = self.feature_extractor.process_frame(rgb_frame)
                
                if features is not None:
                    # Predict emotion
                    emotion_label, emotion_name = self.predict_emotion(features)
                    
                    # Annotate frame
                    frame = self.annotate_frame(frame, emotion_name, emotion_label)
                else:
                    cv2.putText(
                        frame,
                        'No pose detected',
                        TEXT_POSITION,
                        FONT,
                        FONT_SCALE,
                        (0, 0, 255),
                        FONT_THICKNESS
                    )
                
                # Display frame
                cv2.imshow('Webcam Emotion Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Clean up
            video_capture.release()
            cv2.destroyAllWindows()
            self.feature_extractor.close()
        
        print("\n✓ Webcam processing stopped")


def run_inference_on_video(
    video_path: str = TEST_VIDEO_PATH,
    display: bool = True,
    save_output: Optional[str] = None
) -> None:
    """
    Convenience function to run inference on a video file.
    
    Args:
        video_path (str): Path to the input video file
        display (bool): Whether to display the video during processing
        save_output (Optional[str]): Path to save output video (None to skip)
    """
    processor = EmotionVideoProcessor()
    processor.process_video(video_path, display, save_output)


def run_inference_on_webcam() -> None:
    """
    Convenience function to run inference on webcam feed.
    """
    processor = EmotionVideoProcessor()
    processor.process_webcam()


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use main.py to run inference.")
