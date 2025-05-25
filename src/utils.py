"""
Utility functions for Hazardous Fuel Treatment Prediction
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to save
    filepath : str
        Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
    
    Returns:
    --------
    sklearn model
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_results(results_dict, filepath):
    """
    Save results dictionary to JSON file.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results
    filepath : str
        Path to save the results
    """
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {filepath}")


def create_submission(predictions, test_ids, filepath):
    """
    Create submission file for predictions.
    
    Parameters:
    -----------
    predictions : array-like
        Model predictions
    test_ids : array-like
        Test record IDs
    filepath : str
        Path to save submission file
    """
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'Predicted_GIS_ACRES': predictions
    })
    
    submission_df.to_csv(filepath, index=False)
    print(f"Submission saved to {filepath}")


def get_project_info():
    """
    Get project information.
    
    Returns:
    --------
    dict
        Project information
    """
    return {
        'project_name': 'Hazardous Fuel Treatment Prediction',
        'team_members': [
            'Gnanasudharsan Ashokumar',
            'Meghana Rao',
            'Meena Periasamy',
            'Nirmalkumar Thirupallikrishnan Kesavan'
        ],
        'date': datetime.now().strftime('%Y-%m-%d'),
        'version': '1.0.0'
    }


def print_data_summary(df):
    """
    Print comprehensive data summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("=== Data Summary ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    print(df.columns.tolist())
    
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing.sort_values(ascending=False))
    else:
        print("No missing values")
    
    print("\n=== Data Types ===")
    print(df.dtypes.value_counts())
    
    print("\n=== Numerical Features Summary ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(df[numerical_cols].describe())


def calculate_metrics_summary(metrics_list):
    """
    Calculate summary statistics for multiple model runs.
    
    Parameters:
    -----------
    metrics_list : list of dicts
        List of metrics dictionaries
    
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {}
    
    # Get all metric names
    metric_names = list(metrics_list[0].keys())
    
    for metric in metric_names:
        values = [m[metric] for m in metrics_list]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return summary


def format_time(seconds):
    """
    Format time in seconds to readable string.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
    
    Returns:
    --------
    str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def check_environment():
    """
    Check if required packages are installed.
    
    Returns:
    --------
    dict
        Environment check results
    """
    packages = {
        'pandas': None,
        'numpy': None,
        'sklearn': None,
        'xgboost': None,
        'matplotlib': None,
        'seaborn': None,
        'shap': None
    }
    
    for package in packages:
        try:
            module = __import__(package)
            packages[package] = module.__version__
        except ImportError:
            packages[package] = 'Not installed'
    
    return packages
