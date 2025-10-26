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
    },
    'random_forest': {
        'model_type': 'RandomForestClassifier',
        'params': {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'xgboost': {
        'model_type': 'XGBClassifier',
        'params': {
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    },
    'lightgbm': {
        'model_type': 'LGBMClassifier',
        'params': {
            'random_state': 42,
            'verbose': -1
        }
    },
    'svm': {
        'model_type': 'SVC',
        'params': {
            'random_state': 42,
            'probability': True
        }
    },
    'logistic_regression': {
        'model_type': 'LogisticRegression',
        'params': {
            'random_state': 42,
            'max_iter': 1000
        }
    }
}

# Hyperparameter Tuning Configuration
TUNING_CONFIG = {
    'enable_tuning': True,
    'tuning_method': 'optuna',  # 'grid', 'random', or 'optuna'
    'n_trials': 20,  # Reduced for faster execution
    'cv_folds': 3,   # Reduced for faster execution
    'scoring': 'roc_auc',  # Changed to roc_auc which works better with imbalanced data
    'n_jobs': -1,
    'random_state': 42
}

# Hyperparameter Search Spaces
HYPERPARAMETER_SPACES = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'lightgbm': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']
    }
}

# Threshold Optimization Configuration
THRESHOLD_CONFIG = {
    'enable_threshold_optimization': True,
    'threshold_range': (0.1, 0.9),
    'threshold_step': 0.01,
    'optimization_metric': 'f1'
}

# Model Stacking Configuration
STACKING_CONFIG = {
    'enable_stacking': True,
    'stacking_method': 'voting',  # 'voting' or 'blending'
    'meta_model': 'logistic_regression',
    'cv_folds': 5,
    'use_proba': True
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