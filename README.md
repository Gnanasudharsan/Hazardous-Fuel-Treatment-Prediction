# Hazardous Fuel Treatment Prediction

## Project Overview
This project develops machine learning models to predict the number of acres treated in hazardous fuel reduction activities by the U.S. Forest Service. Using features such as treatment type, method, planning year, land suitability, and cost inputs, we built predictive models that achieve over 94% accuracy.

## Team Members
- **Gnanasudharsan Ashokumar** - ashokumar.g@northeastern.edu
- **Meghana Rao** - rao.meg@northeastern.edu
- **Meena Periasamy** - periasamy.m@northeastern.edu
- **Nirmalkumar Thirupallikrishnan Kesavan** - thirupallikrishnan.n@northeastern.edu

## Dataset
The dataset used is the **Hazardous Fuel Treatment Reduction – Polygon Feature Layer** from the U.S. Forest Service through Data.gov. It contains detailed information about fuel treatment activities across U.S. federal lands.

- **Source**: [Data.gov - Hazardous Fuel Treatment Reduction](https://catalog.data.gov/dataset/hazardous-fuel-treatment-reduction-polygon-feature-layer-9c557)
- **Size**: 166,500 rows × 82 columns
- **Target Variable**: GIS_ACRES (treated area size)

## Key Features
- **Numerical**: ASU_NBR_UNITS, COST_PER_UOM, FISCAL_YEAR_PLANNED, LAND_SUITABILITY_CLASS_CODE, PRODUCTIVITY_CLASS_CODE
- **Categorical**: STATE_ABBR, TREATMENT_TYPE, METHOD, OWNERSHIP_CODE

## Project Structure
```
├── data/                    # Dataset directory (add your CSV file here)
├── notebooks/              # Jupyter notebooks with complete analysis
├── src/                    # Source code modules
│   ├── data_preprocessing.py
│   ├── visualization.py
│   ├── modeling.py
│   └── utils.py
├── figures/                # Generated plots and visualizations
├── docs/                   # Project documentation and reports
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gnanasudharsan/Hazardous-Fuel-Treatment-Prediction.git
cd hazardous-fuel-treatment-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download the CSV file from the [Data.gov link](https://catalog.data.gov/dataset/hazardous-fuel-treatment-reduction-polygon-feature-layer-9c557)
   - Place it in the `data/` directory as `Hazardous_Fuel_Treatment_Reduction__Polygon_(Feature_Layer)(1).csv`

## Usage

### Running the Complete Analysis
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/hazardous_fuel_treatment_analysis.ipynb
```

### Using Individual Modules
```python
# Data preprocessing
from src.data_preprocessing import load_and_clean_data
df_clean = load_and_clean_data('path/to/dataset.csv')

# Visualization
from src.visualization import create_distribution_plots
create_distribution_plots(df_clean)

# Modeling
from src.modeling import train_random_forest, train_xgboost
rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
```

## Methodology

### 1. Data Preprocessing
- Removed columns with >70% missing values
- Dropped rows with missing or zero target values
- Selected relevant features for modeling
- Applied log transformation to handle skewed distribution

### 2. Exploratory Data Analysis
- Distribution analysis of treatment areas
- Correlation analysis between features
- Geographic and categorical breakdowns
- Temporal trend analysis

### 3. Clustering Analysis
- K-Means clustering (k=3) to identify project patterns
- PCA for dimensionality reduction and visualization

### 4. Predictive Modeling
- **Random Forest**: 100 trees, max depth 15
- **XGBoost**: Grid search optimized hyperparameters
- Train-test split: 80/20
- Feature preprocessing: StandardScaler for numeric, OneHotEncoder for categorical

## Results

### Model Performance
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Random Forest | 0.9421 | 8.07 | 2.45 |
| XGBoost | 0.9414 | 8.11 | 2.56 |

### Key Findings
1. **Most Important Features**:
   - ASU_NBR_UNITS (number of units treated)
   - FISCAL_YEAR_PLANNED
   - COST_PER_UOM
   - Land classification codes

2. **Treatment Patterns**:
   - Most treatments are small-scale (<50 acres)
   - Large outliers exist, especially for broadcast burns
   - Geographic variations in treatment scale

3. **Cluster Analysis** revealed three distinct project types based on scale and resource intensity

## Visualizations
The project includes various visualizations:
- Distribution plots (raw and log-transformed)
- Correlation heatmaps
- Box plots by state, treatment type, and method
- Temporal trends
- Cluster analysis plots
- Feature importance charts
- Actual vs. predicted scatter plots

## Future Work
- Incorporate temporal features for time-series analysis
- Add weather and climate data
- Develop region-specific models
- Create a web application for real-time predictions
- Investigate ensemble methods combining multiple algorithms

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- shap

See `requirements.txt` for complete list with versions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- U.S. Forest Service for providing the dataset
- Data.gov for hosting the data
- Course instructors and teammates for guidance and collaboration
