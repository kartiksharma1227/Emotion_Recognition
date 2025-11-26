"""
Main script to run emotion recognition training and inference.

Usage:
    Train model:        python main.py --mode train
    Run inference:      python main.py --mode inference --video path/to/video.mp4
    Use webcam:         python main.py --mode webcam
    Full pipeline:      python main.py --mode all
"""

import argparse
import os
import sys
from data_loader import load_all_h5_files
from train_model import train_emotion_classifier, EmotionClassifier
from inference import run_inference_on_video, run_inference_on_webcam
from config import (
    DATA_DIR,
    MODEL_SAVE_PATH,
    SCALER_SAVE_PATH,
    TEST_VIDEO_PATH
)


def check_data_exists():
    """
    Check if data directory and H5 files exist.
    
    Returns:
        bool: True if data exists, False otherwise
    """
    if not os.path.exists(DATA_DIR):
        print(f"✗ Error: Data directory '{DATA_DIR}' not found!")
        print(f"  Please create the directory and add H5 files.")
        return False
    
    h5_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]
    if not h5_files:
        print(f"✗ Error: No H5 files found in '{DATA_DIR}'!")
        print(f"  Please add H5 files with emotion data.")
        return False
    
    return True


def check_model_exists():
    """
    Check if trained model files exist.
    
    Returns:
        bool: True if model files exist, False otherwise
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"✗ Model not found: {MODEL_SAVE_PATH}")
        return False
    
    if not os.path.exists(SCALER_SAVE_PATH):
        print(f"✗ Scaler not found: {SCALER_SAVE_PATH}")
        return False
    
    return True


def train_pipeline():
    """
    Execute the complete training pipeline.
    """
    print("\n" + "="*60)
    print("TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Check if data exists
    if not check_data_exists():
        sys.exit(1)
    
    # Load data
    print("Step 1: Loading data...")
    X, y = load_all_h5_files()
    
    # Train model
    print("\nStep 2: Training model...")
    classifier = train_emotion_classifier(X, y)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60 + "\n")


def inference_pipeline(video_path: str = TEST_VIDEO_PATH, display: bool = True, save_output: str = None):
    """
    Execute the inference pipeline on a video file.
    
    Args:
        video_path (str): Path to input video
        display (bool): Whether to display video during processing
        save_output (str): Path to save output video (optional)
    """
    print("\n" + "="*60)
    print("INFERENCE PIPELINE")
    print("="*60 + "\n")
    
    # Check if model exists
    if not check_model_exists():
        print("\n✗ Trained model not found!")
        print("  Please run training first: python main.py --mode train")
        sys.exit(1)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"✗ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Run inference
    run_inference_on_video(video_path, display, save_output)
    
    print("\n" + "="*60)
    print("✓ INFERENCE COMPLETE!")
    print("="*60 + "\n")


def webcam_pipeline():
    """
    Execute real-time webcam emotion detection.
    """
    print("\n" + "="*60)
    print("WEBCAM EMOTION DETECTION")
    print("="*60 + "\n")
    
    # Check if model exists
    if not check_model_exists():
        print("\n✗ Trained model not found!")
        print("  Please run training first: python main.py --mode train")
        sys.exit(1)
    
    # Run webcam inference
    run_inference_on_webcam()
    
    print("\n" + "="*60)
    print("✓ WEBCAM DETECTION STOPPED!")
    print("="*60 + "\n")


def full_pipeline(video_path: str = TEST_VIDEO_PATH):
    """
    Execute both training and inference pipelines.
    
    Args:
        video_path (str): Path to test video
    """
    print("\n" + "="*60)
    print("FULL PIPELINE: TRAINING + INFERENCE")
    print("="*60 + "\n")
    
    # Train model
    train_pipeline()
    
    # Run inference
    inference_pipeline(video_path)
    
    print("\n" + "="*60)
    print("✓ FULL PIPELINE COMPLETE!")
    print("="*60 + "\n")


def main():
    """
    Main entry point for the emotion recognition system.
    """
    parser = argparse.ArgumentParser(
        description='Emotion Recognition from Pose Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model:              python main.py --mode train
  Run inference on video:   python main.py --mode inference --video testing.mp4
  Run webcam detection:     python main.py --mode webcam
  Full pipeline:            python main.py --mode all
  Save output video:        python main.py --mode inference --video input.mp4 --output result.mp4
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference', 'webcam', 'all'],
        default='train',
        help='Execution mode: train, inference, webcam, or all (default: train)'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        default=TEST_VIDEO_PATH,
        help=f'Path to input video file for inference (default: {TEST_VIDEO_PATH})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output video with predictions (optional)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display during inference (useful for servers/Colab)'
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    try:
        if args.mode == 'train':
            train_pipeline()
        
        elif args.mode == 'inference':
            inference_pipeline(
                video_path=args.video,
                display=not args.no_display,
                save_output=args.output
            )
        
        elif args.mode == 'webcam':
            if args.no_display:
                print("✗ Error: Webcam mode requires display!")
                sys.exit(1)
            webcam_pipeline()
        
        elif args.mode == 'all':
            full_pipeline(video_path=args.video)
    
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
