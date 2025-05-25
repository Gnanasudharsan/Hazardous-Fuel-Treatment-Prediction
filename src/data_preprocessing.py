"""
Data preprocessing functions for Hazardous Fuel Treatment Prediction
"""

import pandas as pd
import numpy as np


def load_and_clean_data(file_path, verbose=True):
    """
    Load and clean the hazardous fuel treatment dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    verbose : bool
        Whether to print progress messages
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset ready for analysis
    """
    # Load data
    if verbose:
        print("Loading dataset...")
    df = pd.read_csv(file_path, low_memory=False)
    
    if verbose:
        print(f"Initial shape: {df.shape}")
    
    # Drop columns with >70% missing values
    threshold = 0.7 * len(df)
    df_clean = df.dropna(thresh=threshold, axis=1)
    
    if verbose:
        print(f"After dropping high-missing columns: {df_clean.shape}")
    
    # Drop rows where target is missing or zero
    df_clean = df_clean[df_clean['GIS_ACRES'].notnull()]
    df_clean = df_clean[df_clean['GIS_ACRES'] > 0]
    
    if verbose:
        print(f"After cleaning target variable: {df_clean.shape}")
    
    return df_clean


def drop_unnecessary_columns(df):
    """
    Drop identifier and metadata columns that are not useful for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with unnecessary columns removed
    """
    columns_to_drop = [
        'OBJECTID', 'SUID', 'FACTS_ID', 'UK', 'CRC_VALUE', 'ETL_MODIFIED',
        'UK_HAZ', 'CRC_HAZ', 'ETL_MODIFIED_DATE_HAZ', 'EDW_INSERT_DATE',
        'ACT_CREATED_DATE', 'ACT_MODIFIED_DATE', 'DATE_PLANNED',
        'DATE_AWARDED', 'DATE_COMPLETED', 'REV_DATE', 'NEPA_PROJECT_ID',
        'NEPA_DOC_NAME', 'IMPLEMENTATION_PROJECT', 'IMPLEMENTATION_PROJECT_NBR',
        'IMPLEMENTATION_PROJECT_TYPE', 'SHAPEAREA', 'SHAPELEN', 'SUBUNIT',
        'KEYPOINT', 'STAGE', 'STAGE_VALUE'
    ]
    
    return df.drop(columns=columns_to_drop, errors='ignore')


def select_modeling_features(df, selected_columns=None):
    """
    Select important columns for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    selected_columns : list, optional
        List of columns to select. If None, uses default selection.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with selected columns
    """
    if selected_columns is None:
        selected_columns = [
            'GIS_ACRES',
            'ASU_NBR_UNITS',
            'COST_PER_UOM',
            'STATE_ABBR',
            'TREATMENT_TYPE',
            'METHOD',
            'FISCAL_YEAR_PLANNED',
            'OWNERSHIP_CODE',
            'LAND_SUITABILITY_CLASS_CODE',
            'PRODUCTIVITY_CLASS_CODE'
        ]
    
    # Select columns and drop rows with missing values
    df_model = df[selected_columns].dropna()
    
    # Ensure target is positive
    df_model = df_model[df_model['GIS_ACRES'] > 0]
    
    return df_model


def add_log_target(df, target_col='GIS_ACRES'):
    """
    Add log-transformed target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added log target column
    """
    df = df.copy()
    df[f'LOG_{target_col}'] = np.log1p(df[target_col])
    return df


def remove_outliers(df, target_col='GIS_ACRES', percentile=99):
    """
    Remove outliers based on percentile.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Column to check for outliers
    percentile : float
        Percentile threshold (default 99)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    threshold = df[target_col].quantile(percentile / 100)
    return df[df[target_col] <= threshold]


def prepare_features_target(df, target_col='GIS_ACRES', 
                          cat_cols=None, exclude_cols=None):
    """
    Prepare features and target for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    cat_cols : list, optional
        List of categorical columns
    exclude_cols : list, optional
        Additional columns to exclude from features
    
    Returns:
    --------
    tuple
        (X, y, cat_cols, num_cols) - features, target, categorical columns, numerical columns
    """
    if cat_cols is None:
        cat_cols = ['STATE_ABBR', 'TREATMENT_TYPE', 'METHOD', 'OWNERSHIP_CODE']
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Separate features and target
    exclude_cols = exclude_cols + [target_col]
    if f'LOG_{target_col}' in df.columns:
        exclude_cols.append(f'LOG_{target_col}')
    
    X = df.drop(columns=exclude_cols)
    y = df[target_col]
    
    # Identify numerical columns
    num_cols = [col for col in X.columns if col not in cat_cols]
    
    return X, y, cat_cols, num_cols
