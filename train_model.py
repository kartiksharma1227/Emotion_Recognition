"""
Model training module for emotion classification using Random Forest.
"""

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, Dict, Any
from config import (
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MODEL_SAVE_PATH,
    SCALER_SAVE_PATH,
    EMOTION_LABELS
)


class EmotionClassifier:
    """
    Random Forest classifier for emotion recognition from pose features.
    """
    
    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        max_depth: int = MAX_DEPTH,
        random_state: int = RANDOM_STATE
    ):
        """
        Initialize the emotion classifier.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees (None for unlimited)
            random_state (int): Random seed for reproducibility
        """
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_data(
        self,
        feature_matrix: np.ndarray,
        label_vector: np.ndarray,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split and scale the dataset into training and testing sets.
        
        Args:
            feature_matrix (np.ndarray): Feature matrix of shape (n_samples, 48)
            label_vector (np.ndarray): Label vector of shape (n_samples,)
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for splitting
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                X_train, X_test, y_train, y_test (all scaled)
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            label_vector,
            test_size=test_size,
            random_state=random_state,
            stratify=label_vector
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: X_train={X_train_scaled.shape}, y_train={y_train.shape}")
        print(f"Testing set: X_test={X_test_scaled.shape}, y_test={y_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Train the Random Forest classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        print("\nTraining Random Forest classifier...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        print("✓ Training complete!")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate the trained classifier on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(
            y_test,
            y_pred,
            target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())]
        )
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Model Evaluation Results")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:\n{class_report}")
        print(f"\nConfusion Matrix:\n{conf_matrix}")
        print(f"{'='*60}\n")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred
        }
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict emotion for a single sample.
        
        Args:
            features (np.ndarray): Feature vector of shape (1, 48)
            
        Returns:
            int: Predicted emotion label (0-3)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        features_scaled = self.scaler.transform(features)
        prediction = self.classifier.predict(features_scaled)[0]
        return prediction
    
    def save_model(
        self,
        model_path: str = MODEL_SAVE_PATH,
        scaler_path: str = SCALER_SAVE_PATH
    ) -> None:
        """
        Save the trained model and scaler to disk.
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
    
    def load_model(
        self,
        model_path: str = MODEL_SAVE_PATH,
        scaler_path: str = SCALER_SAVE_PATH
    ) -> None:
        """
        Load a trained model and scaler from disk.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
        """
        self.classifier = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Scaler loaded from: {scaler_path}")


def train_emotion_classifier(
    feature_matrix: np.ndarray,
    label_vector: np.ndarray
) -> EmotionClassifier:
    """
    Complete training pipeline for emotion classifier.
    
    Args:
        feature_matrix (np.ndarray): Feature matrix of shape (n_samples, 48)
        label_vector (np.ndarray): Label vector of shape (n_samples,)
        
    Returns:
        EmotionClassifier: Trained classifier instance
    """
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(
        feature_matrix,
        label_vector
    )
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate model
    classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save_model()
    
    return classifier


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use main.py to train the model.")
