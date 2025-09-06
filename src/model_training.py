"""
Model training module for Bank Term Deposit Prediction.
Handles model definition, training, and persistence.
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Change this line - remove the dot
from config import MODEL_CONFIG, MODELS_DIR

class BankModelTrainer:
    """Model training class for bank marketing prediction."""

    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.trained_models = {}
        self.training_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models from configuration."""
        print("Initializing models...")

        for model_name, config in MODEL_CONFIG.items():
            model_type = config['model_type']
            params = config['params']

            if model_type == 'GaussianNB':
                self.models[model_name] = GaussianNB(**params)
            elif model_type == 'DecisionTreeClassifier':
                self.models[model_name] = DecisionTreeClassifier(**params)
            else:
                print(f"Warning: Unknown model type {model_type}")
                continue

            print(f"  ✓ {model_name}: {model_type}")

        return self.models

    def train_single_model(self, model_name: str, model: Any,
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train a single model and evaluate it."""
        print(f"\nTraining {model_name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        pos_label = 'yes' if 'yes' in y_test.unique() else y_test.unique()[1]
        precision = precision_score(y_test, y_pred, pos_label=pos_label, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=pos_label, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=pos_label, average='binary')

        # Generate reports
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        results = {
            'model': model,
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }

        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        return results

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train all configured models."""
        print("="*60)
        print("TRAINING ALL MODELS")
        print("="*60)

        self.initialize_models()

        for model_name, model in self.models.items():
            try:
                results = self.train_single_model(
                    model_name, model, X_train, y_train, X_test, y_test
                )
                self.training_results[model_name] = results
                self.trained_models[model_name] = model
            except Exception as e:
                print(f"Error training {model_name}: {e}")

        return self.training_results

    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all trained models."""
        if not self.training_results:
            return pd.DataFrame()

        comparison_data = []
        for model_name, results in self.training_results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

        print("\nMODEL COMPARISON:")
        print(comparison_df.round(4))

        return comparison_df

    def save_models(self) -> Dict[str, str]:
        """Save all trained models to disk."""
        saved_paths = {}

        print(f"\nSaving models to {MODELS_DIR}...")

        for model_name, model in self.trained_models.items():
            filename = f"{model_name}_model_{self.timestamp}.pkl"
            filepath = MODELS_DIR / filename

            joblib.dump(model, filepath)
            saved_paths[model_name] = str(filepath)
            print(f"  ✓ {model_name} → {filename}")

        return saved_paths

def main_model_training(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to execute model training pipeline."""
    print("="*60)
    print("BANK TERM DEPOSIT PREDICTION - MODEL TRAINING")
    print("="*60)

    trainer = BankModelTrainer()

    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']

    training_results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    comparison_df = trainer.compare_models()
    saved_paths = trainer.save_models()

    return {
        'trainer': trainer,
        'training_results': training_results,
        'model_comparison': comparison_df,
        'saved_model_paths': saved_paths,
        'trained_models': trainer.trained_models
    }