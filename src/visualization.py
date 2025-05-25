"""
Visualization functions for Hazardous Fuel Treatment Prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def create_distribution_plots(df, target_col='GIS_ACRES', figsize=(14, 5)):
    """
    Create distribution plots for the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Target column name
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Raw distribution
    sns.histplot(df[target_col], bins=50, kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {target_col}")
    axes[0].set_xlabel("Acres Treated")
    axes[0].set_ylabel("Frequency")
    
    # Log distribution
    if f'LOG_{target_col}' in df.columns:
        log_col = f'LOG_{target_col}'
    else:
        log_col = np.log1p(df[target_col])
    
    sns.histplot(log_col, bins=50, kde=True, ax=axes[1])
    axes[1].set_title(f"Log-Transformed Distribution of {target_col}")
    axes[1].set_xlabel("Log(Acres Treated)")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()


def create_correlation_heatmap(df, numerical_cols=None, figsize=(10, 6)):
    """
    Create correlation heatmap for numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerical_cols : list, optional
        List of numerical columns. If None, uses default selection
    figsize : tuple
        Figure size
    """
    if numerical_cols is None:
        numerical_cols = ['GIS_ACRES', 'ASU_NBR_UNITS', 'COST_PER_UOM',
                         'FISCAL_YEAR_PLANNED', 'LAND_SUITABILITY_CLASS_CODE',
                         'PRODUCTIVITY_CLASS_CODE']
    
    # Filter columns that exist in dataframe
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    plt.figure(figsize=figsize)
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", 
                fmt='.3f', square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def create_categorical_boxplots(df, cat_col, target_col='GIS_ACRES', 
                               figsize=(12, 5), rotation=90):
    """
    Create boxplot for categorical variable vs target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    cat_col : str
        Categorical column name
    target_col : str
        Target column name
    figsize : tuple
        Figure size
    rotation : int
        X-axis label rotation
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=cat_col, y=target_col)
    plt.xticks(rotation=rotation)
    plt.title(f"{target_col} by {cat_col}")
    plt.tight_layout()
    plt.show()


def create_yearly_trend(df, year_col='FISCAL_YEAR_PLANNED', 
                       target_col='GIS_ACRES', figsize=(10, 5)):
    """
    Create yearly trend plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    year_col : str
        Year column name
    target_col : str
        Target column name
    figsize : tuple
        Figure size
    """
    yearly_avg = df.groupby(year_col)[target_col].mean()
    
    plt.figure(figsize=figsize)
    sns.lineplot(x=yearly_avg.index, y=yearly_avg.values)
    plt.title(f"Average {target_col} Over Years")
    plt.xlabel("Fiscal Year")
    plt.ylabel(f"Average {target_col}")
    plt.tight_layout()
    plt.show()


def create_pairplot(df, columns, corner=True):
    """
    Create pairplot for selected columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of columns to include
    corner : bool
        Whether to show only lower triangle
    """
    # Filter columns that exist in dataframe
    columns = [col for col in columns if col in df.columns]
    
    g = sns.pairplot(df[columns], corner=corner)
    g.fig.suptitle("Pairwise Relationships", y=1.02)
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=15, figsize=(10, 6)):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importances : list
        List of importance scores
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    """
    # Create dataframe and sort
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=imp_df, x='Importance', y='Feature')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", 
                           figsize=(8, 5)):
    """
    Create actual vs predicted scatter plot.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    
    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(models, metrics_dict, figsize=(15, 5)):
    """
    Plot comparison of multiple models.
    
    Parameters:
    -----------
    models : list
        List of model names
    metrics_dict : dict
        Dictionary with keys 'r2', 'rmse', 'mae' and values as lists
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = ['skyblue', 'lightgreen', 'coral', 'gold'][:len(models)]
    
    # R² Score
    axes[0].bar(models, metrics_dict['r2'], color=colors)
    axes[0].set_title("R² Score")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # RMSE
    axes[1].bar(models, metrics_dict['rmse'], color=colors)
    axes[1].set_title("RMSE")
    axes[1].set_ylabel("Score")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAE
    axes[2].bar(models, metrics_dict['mae'], color=colors)
    axes[2].set_title("MAE")
    axes[2].set_ylabel("Score")
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Model Performance Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_cluster_distribution(df, cluster_col='Cluster', 
                            feature_col='TREATMENT_TYPE', figsize=(12, 5)):
    """
    Plot distribution of features across clusters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with cluster labels
    cluster_col : str
        Cluster column name
    feature_col : str
        Feature column to analyze
    figsize : tuple
        Figure size
    """
    crosstab = pd.crosstab(df[feature_col], df[cluster_col])
    
    crosstab.plot(kind='bar', stacked=True, figsize=figsize)
    plt.title(f"{feature_col} by {cluster_col}")
    plt.xlabel(feature_col)
    plt.ylabel("Number of Projects")
    plt.xticks(rotation=90)
    plt.legend(title=cluster_col)
    plt.tight_layout()
    plt.show()
