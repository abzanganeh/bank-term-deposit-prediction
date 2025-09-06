"""
Feature engineering module for Bank Term Deposit Prediction.
Handles feature creation, transformation, and preprocessing.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_CONFIG, FEATURE_CONFIG, CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS, CORRELATED_FEATURES
)

class BankFeatureEngineer:
    """Feature engineering class for bank marketing dataset."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.categorical_columns = None
        self.numerical_columns = None

    def create_pdays_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary feature for pdays == 999 (no previous contact)."""
        df_copy = df.copy()

        if 'pdays' in df_copy.columns:
            # Create binary feature indicating no previous contact
            df_copy['pdays999'] = (df_copy['pdays'] == FEATURE_CONFIG['pdays_threshold']).astype(int)
            print(f"Created 'pdays999' feature:")
            print(f"Value counts: {df_copy['pdays999'].value_counts().to_dict()}")
        else:
            print("Warning: 'pdays' column not found in dataset")

        return df_copy

    def drop_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop highly correlated features to reduce multicollinearity."""
        df_copy = df.copy()

        if FEATURE_CONFIG['drop_correlated']:
            columns_to_drop = []

            # Drop original pdays column if pdays999 was created
            if 'pdays999' in df_copy.columns and 'pdays' in df_copy.columns:
                columns_to_drop.append('pdays')

            # Add highly correlated features that exist in the dataset
            for feature in CORRELATED_FEATURES:
                if feature in df_copy.columns:
                    columns_to_drop.append(feature)

            if columns_to_drop:
                print(f"Dropping correlated features: {columns_to_drop}")
                df_copy = df_copy.drop(columns=columns_to_drop)
                print(f"Dataset shape after dropping features: {df_copy.shape}")
            else:
                print("No correlated features to drop")

        return df_copy

    def identify_feature_types(self, df: pd.DataFrame, target_column: str) -> Tuple[list, list]:
        """Identify categorical and numerical columns in the dataset."""
        feature_columns = [col for col in df.columns if col != target_column]

        categorical_cols = []
        numerical_cols = []

        for col in feature_columns:
            if df[col].dtype in ['object', 'bool'] or col.endswith('999'):
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() <= 10 and df[col].dtype == 'int64':
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)

        print(f"Identified {len(categorical_cols)} categorical, {len(numerical_cols)} numerical features")

        self.categorical_columns = categorical_cols
        self.numerical_columns = numerical_cols

        return categorical_cols, numerical_cols

    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """Convert categorical features to dummy variables."""
        if not categorical_cols:
            return pd.DataFrame()

        categorical_df = df[categorical_cols]
        encoded_df = pd.get_dummies(categorical_df, drop_first=True)

        print(f"Categorical encoding: {len(categorical_cols)} → {len(encoded_df.columns)} features")
        return encoded_df

    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create train-test split for the dataset."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state'],
            stratify=y
        )

        print("=== TRAIN-TEST SPLIT ===")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Main feature engineering pipeline."""
        print("="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)

        # Step 1: Create new features
        df_engineered = self.create_pdays_feature(df)

        # Step 2: Drop correlated features
        df_engineered = self.drop_correlated_features(df_engineered)

        # Step 3: Separate features and target
        target_column = DATA_CONFIG['target_column']
        X = df_engineered.drop(columns=[target_column])
        y = df_engineered[target_column]

        # Step 4: Identify feature types
        categorical_cols, numerical_cols = self.identify_feature_types(X, target_column)

        # Step 5: Encode categorical features
        X_categorical = self.encode_categorical_features(X, categorical_cols)

        # Step 6: Handle numerical features
        if numerical_cols:
            X_numerical = X[numerical_cols]
            X_final = pd.concat([X_categorical, X_numerical], axis=1)
        else:
            X_final = X_categorical

        print(f"Final feature matrix shape: {X_final.shape}")
        return X_final, y

def main_feature_engineering(df: pd.DataFrame, apply_scaling: bool = False) -> Dict[str, Any]:
    """Main function to execute feature engineering pipeline."""
    print("="*60)
    print("BANK TERM DEPOSIT PREDICTION - FEATURE ENGINEERING")
    print("="*60)

    engineer = BankFeatureEngineer()
    X, y = engineer.engineer_features(df)
    split_data = engineer.create_train_test_split(X, y)

    results = {
        'X': X,
        'y': y,
        'categorical_columns': engineer.categorical_columns or [],
        'numerical_columns': engineer.numerical_columns or [],
        'feature_names': list(X.columns),
        **split_data
    }

    print(f"Feature engineering completed! Total features: {len(results['feature_names'])}")
    return results