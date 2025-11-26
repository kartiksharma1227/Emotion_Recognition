"""
Data loading module for reading and processing H5 files containing pose features.
"""

import h5py
import numpy as np
import glob
from typing import Tuple, List
from config import EMOTION_KEYWORDS, NUM_FEATURES, DATA_DIR


def infer_emotion_label_from_name(dataset_name: str) -> int:
    """
    Infer emotion label from dataset name based on emotion keywords.
    
    Args:
        dataset_name (str): Name of the dataset in H5 file
        
    Returns:
        int: Emotion label (0=Angry, 1=Happy, 2=Neutral, 3=Sad, -1=Unknown)
    """
    for label, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in dataset_name:
                return label
    return -1  # Unknown emotion


def load_h5_file(file_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load pose features and labels from a single H5 file.
    
    Args:
        file_path (str): Path to the H5 file
        
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Lists of feature arrays and label arrays
    """
    samples = []
    labels = []
    
    print(f"Processing {file_path}")
    
    with h5py.File(file_path, 'r') as h5_file:
        for dataset_key in h5_file.keys():
            data = h5_file[dataset_key][()]
            
            # Skip scalar or empty datasets
            if np.isscalar(data) or data.size == 0:
                print(f"Skipping scalar or empty dataset {file_path}::{dataset_key}")
                continue
            
            # Handle 1D arrays
            if data.ndim == 1:
                if data.size == NUM_FEATURES:
                    data = data.reshape(1, NUM_FEATURES)
                else:
                    print(f"Skipping 1D dataset with unexpected length: {file_path}::{dataset_key}, length={data.size}")
                    continue
            
            # Validate dimensions
            if data.ndim != 2 or data.shape[1] != NUM_FEATURES:
                print(f"Skipping dataset with unexpected dims {file_path}::{dataset_key} shape={data.shape}")
                continue
            
            # Infer emotion label
            emotion_label = infer_emotion_label_from_name(dataset_key)
            if emotion_label == -1:
                print(f"Skipping dataset with unknown label: {file_path}::{dataset_key}")
                continue
            
            # Store samples and labels
            num_samples = data.shape[0]
            samples.append(data.astype('float32'))
            labels.append(np.full(num_samples, emotion_label, dtype='int64'))
    
    return samples, labels


def load_all_h5_files(data_directory: str = DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all H5 files from the data directory and combine them into feature and label arrays.
    
    Args:
        data_directory (str): Path to directory containing H5 files
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X (n_samples, 48) and label vector y (n_samples,)
        
    Raises:
        RuntimeError: If no valid data is found in H5 files
    """
    h5_file_paths = glob.glob(f"{data_directory}/*.h5")
    
    if not h5_file_paths:
        raise RuntimeError(f"No H5 files found in {data_directory}")
    
    all_samples = []
    all_labels = []
    
    for file_path in h5_file_paths:
        samples, labels = load_h5_file(file_path)
        all_samples.extend(samples)
        all_labels.extend(labels)
    
    if len(all_samples) == 0:
        raise RuntimeError("No data read from H5 files!")
    
    # Combine all samples and labels
    feature_matrix = np.vstack(all_samples)
    label_vector = np.concatenate(all_labels)
    
    print(f"\nFinal shapes: X={feature_matrix.shape}, y={label_vector.shape}")
    print(f"Total samples: {len(label_vector)}")
    print(f"Unique labels: {np.unique(label_vector)}")
    
    return feature_matrix, label_vector


if __name__ == "__main__":
    # Test the data loader
    try:
        X, y = load_all_h5_files()
        print("\n✓ Data loading successful!")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label vector shape: {y.shape}")
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
