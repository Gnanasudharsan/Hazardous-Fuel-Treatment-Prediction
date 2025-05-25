"""
Modeling functions for Hazardous Fuel Treatment Prediction
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb


def create_preprocessor(num_cols, cat_cols):
    """
    Create preprocessing pipeline for numerical and categorical features.
    
    Parameters:
    -----------
    num_cols : list
        List of numerical column names
    cat_cols : list
        List of categorical column names
    
    Returns:
    --------
    ColumnTransformer
        Preprocessing pipeline
    """
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    
    return preprocessor


def evaluate_model(y_true, y_pred, verbose=True):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    dict
        Dictionary with metrics (rmse, mae, r2)
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    if verbose:
        print("Model Evaluation:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_random_forest(X_train, X_test, y_train, y_test, 
                       num_cols, cat_cols, n_estimators=100, 
                       max_depth=15, random_state=42):
    """
    Train Random Forest model.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    num_cols : list
        Numerical column names
    cat_cols : list
        Categorical column names
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (model, metrics_dict)
    """
    # Create preprocessor
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    # Create pipeline
    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    # Train model
    rf_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = rf_model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    return rf_model, metrics


def train_xgboost(X_train, X_test, y_train, y_test, 
                  num_cols, cat_cols, param_grid=None, 
                  cv=5, random_state=42):
    """
    Train XGBoost model with grid search.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    num_cols : list
        Numerical column names
    cat_cols : list
        Categorical column names
    param_grid : dict, optional
        Parameter grid for grid search
    cv : int
        Cross-validation folds
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (best_model, metrics_dict, best_params)
    """
    # Create preprocessor
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    # Create pipeline
    xgb_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=random_state
        ))
    ])
    
    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [4, 6, 8],
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__subsample': [0.8, 1.0]
        }
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    return best_model, metrics, grid_search.best_params_


def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract feature importance from tree-based model.
    
    Parameters:
    -----------
    model : Pipeline
        Trained model pipeline
    feature_names : list
        Feature names after preprocessing
    top_n : int
        Number of top features to return
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature names and importance scores
    """
    import pandas as pd
    
    # Get feature importances
    importances = model.named_steps['regressor'].feature_importances_
    
    # Create dataframe
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort and return top features
    return imp_df.sort_values(by='Importance', ascending=False).head(top_n)


def perform_clustering(df, features, n_clusters=3, random_state=42):
    """
    Perform K-means clustering with PCA visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of feature columns to use
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (cluster_labels, pca_components, kmeans_model, pca_model)
    """
    # Prepare data
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_pca)
    
    return clusters, X_pca, kmeans, pca


def cross_validate_model(model, X, y, cv=5, scoring='r2'):
    """
    Perform cross-validation on a model.
    
    Parameters:
    -----------
    model : sklearn model or pipeline
        Model to evaluate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Number of folds
    scoring : str
        Scoring metric
    
    Returns:
    --------
    dict
        Cross-validation results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
        'cv': cv
    }


def create_ensemble_predictions(models, X_test, weights=None):
    """
    Create ensemble predictions from multiple models.
    
    Parameters:
    -----------
    models : list
        List of trained models
    X_test : pd.DataFrame
        Test features
    weights : list, optional
        Model weights for averaging
    
    Returns:
    --------
    array
        Ensemble predictions
    """
    predictions = []
    
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    if weights is None:
        # Simple average
        ensemble_pred = np.mean(predictions, axis=0)
    else:
        # Weighted average
        weights = np.array(weights)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    return ensemble_pred
