"""
Configuration file for Bank Term Deposit Prediction Project.
Contains all constants, paths, and hyperparameters.
"""
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data Configuration
DATA_CONFIG = {
    'file_name': 'bank_data.csv',
    'target_column': 'y',
    'separator': ';',
    'test_size': 0.2,
    'random_state': 42
}

# Feature Configuration
CATEGORICAL_COLUMNS = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome'
]

NUMERICAL_COLUMNS = [
    'age', 'duration', 'campaign', 'previous', 'emp.var.rate',
    'cons.price.idx', 'cons.conf.idx'
]

# Highly correlated features to potentially drop
CORRELATED_FEATURES = ['euribor3m', 'nr.employed']

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'pdays_threshold': 999,  # Value indicating no previous contact
    'drop_correlated': True,
    'correlation_threshold': 0.9
}

# Model Configuration
MODEL_CONFIG = {
    'naive_bayes': {
        'model_type': 'GaussianNB',
        'params': {}
    },
    'decision_tree_gini': {
        'model_type': 'DecisionTreeClassifier',
        'params': {
            'criterion': 'gini',
            'random_state': 42
        }
    },
    'decision_tree_entropy': {
        'model_type': 'DecisionTreeClassifier',
        'params': {
            'criterion': 'entropy',
            'random_state': 42
        }
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'whitegrid',
    'color_palette': ['red', 'blue'],
    'subscription_colors': ['red', 'blue']
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'max_columns_display': 30,
    'correlation_method': 'pearson',
    'top_correlated_features': 3
}