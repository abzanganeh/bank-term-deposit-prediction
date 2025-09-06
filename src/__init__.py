"""
Bank Term Deposit Prediction Package
A comprehensive machine learning project for predicting customer subscription to term deposits.
"""

__version__ = "1.0.0"
__author__ = "Alireza Barzin Zanganeh"
__description__ = "Machine learning pipeline for bank marketing campaign prediction"

# Package imports
import config
import data_processing
import feature_engineering
import model_training
import evaluation
import main


__all__ = [
    'config',
    'data_processing',
    'feature_engineering',
    'model_training',
    'evaluation',
    'main'
]