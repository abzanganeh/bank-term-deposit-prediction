"""
Model evaluation module for Bank Term Deposit Prediction.
Handles model evaluation, metrics calculation, and visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Change this line - remove the dot
from config import VIZ_CONFIG, RESULTS_DIR

class BankModelEvaluator:
    """Model evaluation class for bank marketing prediction."""

    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        self.y_test = None

    def evaluate_single_model(self, model_name: str, y_true: pd.Series,
                             y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate a single model with comprehensive metrics."""

        accuracy = accuracy_score(y_true, y_pred)
        pos_label = 'yes' if 'yes' in y_true.unique() else y_true.unique()[1]
        precision = precision_score(y_true, y_pred, pos_label=pos_label, average='binary')
        recall = recall_score(y_true, y_pred, pos_label=pos_label, average='binary')
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, average='binary')

        class_report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }

        return results

    def evaluate_multiple_models(self, training_results: Dict[str, Any],
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate multiple models and compare their performance."""

        self.y_test = y_test
        evaluation_results = {}

        print("="*60)
        print("MODEL EVALUATION")
        print("="*60)

        for model_name, results in training_results.items():
            if 'model' not in results:
                continue

            print(f"Evaluating {model_name}...")

            model = results['model']
            y_pred = model.predict(X_test)

            eval_results = self.evaluate_single_model(model_name, y_test, y_pred)
            evaluation_results[model_name] = eval_results

            print(f"  F1-Score: {eval_results['f1_score']:.4f}")

        self.evaluation_results = evaluation_results
        return evaluation_results

    def create_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of all model performances."""
        if not self.evaluation_results:
            return pd.DataFrame()

        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

        print("\nMODEL PERFORMANCE COMPARISON:")
        print(comparison_df.round(4))

        return comparison_df

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        if not self.evaluation_results:
            return

        n_models = len(self.evaluation_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            conf_matrix = results['confusion_matrix']
            unique_labels = sorted(self.y_test.unique())

            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[idx],
                cbar=False,
                xticklabels=unique_labels,
                yticklabels=unique_labels
            )

            axes[idx].set_title(f'{model_name.replace("_", " ").title()}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices saved to {RESULTS_DIR}/confusion_matrices.png")

    def identify_best_model(self) -> str:
        """Identify the best performing model based on F1-score."""
        if not self.evaluation_results:
            return None

        best_model = max(
            self.evaluation_results.items(),
            key=lambda x: x[1]['f1_score']
        )

        best_name = best_model[0]
        best_f1 = best_model[1]['f1_score']

        print(f"\nBEST MODEL: {best_name} (F1-Score: {best_f1:.4f})")
        return best_name

    def create_evaluation_summary(self) -> Dict[str, Any]:
        """Create comprehensive evaluation summary."""
        comparison_df = self.create_comparison_table()
        self.plot_confusion_matrices()
        best_model = self.identify_best_model()

        return {
            'comparison_table': comparison_df,
            'best_model': best_model,
            'evaluation_results': self.evaluation_results
        }

def main_evaluation(training_results: Dict[str, Any],
                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Main function to execute model evaluation pipeline."""
    print("="*60)
    print("BANK TERM DEPOSIT PREDICTION - MODEL EVALUATION")
    print("="*60)

    evaluator = BankModelEvaluator()
    evaluation_results = evaluator.evaluate_multiple_models(training_results, X_test, y_test)
    summary = evaluator.create_evaluation_summary()

    print(f"\nModel evaluation completed! Models evaluated: {len(evaluation_results)}")
    return {
        'evaluator': evaluator,
        'evaluation_results': evaluation_results,
        'summary': summary
    }