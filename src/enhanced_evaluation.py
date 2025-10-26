"""
Model evaluation with advanced metrics and visualizations.
Provides detailed analysis of model performance including:
- ROC curves and AUC scores
- Precision-Recall curves for imbalanced data
- Threshold optimization analysis
- Model calibration assessment
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve,
    average_precision_score, log_loss
)
from sklearn.calibration import calibration_curve
import joblib

from config import VIZ_CONFIG, RESULTS_DIR, THRESHOLD_CONFIG

class EnhancedModelEvaluator:
    """
    Evaluates trained models using multiple metrics and creates visualizations.
    Helps compare models and understand their strengths and weaknesses.
    """

    def __init__(self):
        """Set up evaluator with empty result containers."""
        self.evaluation_results = {}
        self.threshold_analysis = {}
        self.calibration_analysis = {}

    def evaluate_single_model(self, model_name: str, model: Any,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             y_pred: np.ndarray = None, y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate performance metrics for a single model.
        Returns accuracy, precision, recall, F1-score, ROC-AUC, and more.
        """
        print(f"\n[EVAL] Evaluating {model_name}...")

        # Get predictions if not provided
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        if y_pred_proba is None and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='yes')
        recall = recall_score(y_test, y_pred, pos_label='yes')
        f1 = f1_score(y_test, y_pred, pos_label='yes')
        
        # Advanced metrics - convert string labels to binary for probability-based metrics
        roc_auc = None
        avg_precision = None
        logloss = None
        
        if y_pred_proba is not None:
            y_test_binary = (y_test == 'yes').astype(int)
            roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
            avg_precision = average_precision_score(y_test_binary, y_pred_proba)
            logloss = log_loss(y_test_binary, y_pred_proba)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'log_loss': logloss,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        avg_precision_str = f"{avg_precision:.4f}" if avg_precision is not None else "N/A"
        
        print(f"  ROC-AUC: {roc_auc_str}")
        print(f"  Average Precision: {avg_precision_str}")

        return results

    def optimize_threshold_detailed(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Test different probability thresholds to find the best one.
        Returns metrics for each threshold and identifies optimal values.
        """
        print(f"\n[THRESHOLD] Detailed threshold optimization...")

        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        metrics = []

        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            y_pred_thresh = ['yes' if x == 1 else 'no' for x in y_pred_thresh]
            
            accuracy = accuracy_score(y_true, y_pred_thresh)
            precision = precision_score(y_true, y_pred_thresh, pos_label='yes', zero_division=0)
            recall = recall_score(y_true, y_pred_thresh, pos_label='yes', zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, pos_label='yes', zero_division=0)
            
            metrics.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

        metrics_df = pd.DataFrame(metrics)
        
        # Find best threshold for different metrics
        best_f1_idx = metrics_df['f1_score'].idxmax()
        best_precision_idx = metrics_df['precision'].idxmax()
        best_recall_idx = metrics_df['recall'].idxmax()
        best_accuracy_idx = metrics_df['accuracy'].idxmax()

        best_thresholds = {
            'f1_score': {
                'threshold': metrics_df.loc[best_f1_idx, 'threshold'],
                'value': metrics_df.loc[best_f1_idx, 'f1_score']
            },
            'precision': {
                'threshold': metrics_df.loc[best_precision_idx, 'threshold'],
                'value': metrics_df.loc[best_precision_idx, 'precision']
            },
            'recall': {
                'threshold': metrics_df.loc[best_recall_idx, 'threshold'],
                'value': metrics_df.loc[best_recall_idx, 'recall']
            },
            'accuracy': {
                'threshold': metrics_df.loc[best_accuracy_idx, 'threshold'],
                'value': metrics_df.loc[best_accuracy_idx, 'accuracy']
            }
        }

        print(f"  Best F1 threshold: {best_thresholds['f1_score']['threshold']:.3f} (F1: {best_thresholds['f1_score']['value']:.4f})")
        print(f"  Best Precision threshold: {best_thresholds['precision']['threshold']:.3f} (Precision: {best_thresholds['precision']['value']:.4f})")
        print(f"  Best Recall threshold: {best_thresholds['recall']['threshold']:.3f} (Recall: {best_thresholds['recall']['value']:.4f})")

        return {
            'metrics_df': metrics_df,
            'best_thresholds': best_thresholds
        }

    def create_evaluation_visualizations(self, evaluation_results: Dict[str, Any],
                                       X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Generate all evaluation plots and save them to the results directory.
        Creates ROC curves, PR curves, confusion matrices, and more.
        """
        print(f"\n[VISUAL] Creating evaluation visualizations...")

        # 1. Model Comparison Bar Chart
        self._plot_model_comparison(evaluation_results)

        # 2. ROC Curves
        self._plot_roc_curves(evaluation_results, y_test)

        # 3. Precision-Recall Curves
        self._plot_precision_recall_curves(evaluation_results, y_test)

        # 4. Confusion Matrices
        self._plot_confusion_matrices(evaluation_results, y_test)

        # 5. Threshold Optimization Plots
        self._plot_threshold_optimization(evaluation_results)

        # 6. Calibration Plots
        self._plot_calibration_curves(evaluation_results, y_test)

    def _plot_model_comparison(self, evaluation_results: Dict[str, Any]) -> None:
        """Plot model comparison bar chart."""
        models = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for model_name, results in evaluation_results.items():
            if results and 'accuracy' in results:
                model_data = {'Model': model_name.replace('_', ' ').title()}
                for metric in metrics:
                    if metric in results and results[metric] is not None:
                        model_data[metric.title()] = results[metric]
                models.append(model_data)

        if not models:
            return

        df = pd.DataFrame(models)
        df = df.set_index('Model')

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if metric in df.columns:
                df[metric].plot(kind='bar', ax=axes[i], color='skyblue', alpha=0.7)
                axes[i].set_title(f'{metric.title()} Comparison')
                axes[i].set_ylabel(metric.title())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)

        # Hide empty subplot
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "enhanced_model_comparison.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"  [OK] Model comparison plot saved")

    def _plot_roc_curves(self, evaluation_results: Dict[str, Any], y_test: pd.Series) -> None:
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        # Convert string labels to binary once
        y_test_binary = (y_test == 'yes').astype(int)
        
        for model_name, results in evaluation_results.items():
            if results and results.get('probabilities') is not None:
                fpr, tpr, _ = roc_curve(y_test_binary, results['probabilities'])
                roc_auc = results.get('roc_auc', 0)
                plt.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "roc_curves_comparison.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"  [OK] ROC curves plot saved")

    def _plot_precision_recall_curves(self, evaluation_results: Dict[str, Any], y_test: pd.Series) -> None:
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(10, 8))
        
        # Convert string labels to binary once
        y_test_binary = (y_test == 'yes').astype(int)
        
        for model_name, results in evaluation_results.items():
            if results and results.get('probabilities') is not None:
                precision, recall, _ = precision_recall_curve(y_test_binary, results['probabilities'])
                avg_precision = results.get('average_precision', 0)
                plt.plot(recall, precision, label=f"{model_name.replace('_', ' ').title()} (AP = {avg_precision:.3f})")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "precision_recall_curves.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"  [OK] Precision-Recall curves plot saved")

    def _plot_confusion_matrices(self, evaluation_results: Dict[str, Any], y_test: pd.Series) -> None:
        """Plot confusion matrices for all models."""
        n_models = len([r for r in evaluation_results.values() if r and 'confusion_matrix' in r])
        if n_models == 0:
            return

        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        model_idx = 0
        for model_name, results in evaluation_results.items():
            if results and 'confusion_matrix' in results:
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[model_idx])
                axes[model_idx].set_title(f'{model_name.replace("_", " ").title()}')
                axes[model_idx].set_xlabel('Predicted')
                axes[model_idx].set_ylabel('Actual')
                model_idx += 1

        # Hide empty subplots
        for i in range(model_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "enhanced_confusion_matrices.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"  [OK] Confusion matrices plot saved")

    def _plot_threshold_optimization(self, evaluation_results: Dict[str, Any]) -> None:
        """Plot threshold optimization results."""
        threshold_data = []
        for model_name, results in evaluation_results.items():
            if results and 'threshold_analysis' in results:
                threshold_df = results['threshold_analysis']['metrics_df']
                threshold_df['model'] = model_name
                threshold_data.append(threshold_df)

        if not threshold_data:
            return

        combined_df = pd.concat(threshold_data, ignore_index=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            for model in combined_df['model'].unique():
                model_data = combined_df[combined_df['model'] == model]
                ax.plot(model_data['threshold'], model_data[metric], label=model.replace('_', ' ').title())
            
            ax.set_xlabel('Threshold')
            ax.set_ylabel(metric.title())
            ax.set_title(f'{metric.title()} vs Threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "threshold_optimization.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"  [OK] Threshold optimization plot saved")

    def _plot_calibration_curves(self, evaluation_results: Dict[str, Any], y_test: pd.Series) -> None:
        """Plot calibration curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in evaluation_results.items():
            if results and results.get('probabilities') is not None:
                y_proba = results['probabilities']
                y_binary = (y_test == 'yes').astype(int)
                
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, y_proba, n_bins=10
                )
                
                plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                        label=f"{model_name.replace('_', ' ').title()}")

        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "calibration_curves.png", dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        print(f"  [OK] Calibration curves plot saved")

    def generate_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary."""
        print(f"\n[SUMMARY] Generating evaluation summary...")

        # Find best model for each metric
        best_models = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics:
            best_score = 0
            best_model = None
            for model_name, results in evaluation_results.items():
                if results and metric in results and results[metric] is not None:
                    if results[metric] > best_score:
                        best_score = results[metric]
                        best_model = model_name
            
            if best_model:
                best_models[metric] = {
                    'model': best_model,
                    'score': best_score
                }

        # Overall best model (based on F1-score)
        overall_best = best_models.get('f1_score', {}).get('model', 'N/A')

        summary = {
            'total_models': len([r for r in evaluation_results.values() if r]),
            'best_models': best_models,
            'overall_best_model': overall_best,
            'evaluation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        print(f"  Total models evaluated: {summary['total_models']}")
        print(f"  Overall best model: {overall_best}")
        
        for metric, info in best_models.items():
            print(f"  Best {metric}: {info['model']} ({info['score']:.4f})")

        return summary

def main_enhanced_evaluation(training_results: Dict[str, Any],
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate all trained models and create comparison visualizations.
    Returns evaluation metrics and identifies the best performing models.
    """
    print("="*80)
    print("ENHANCED MODEL EVALUATION")
    print("="*80)

    evaluator = EnhancedModelEvaluator()
    evaluation_results = {}

    # Evaluate all models
    for model_name, results in training_results.items():
        if results and 'model' in results:
            model_eval = evaluator.evaluate_single_model(
                model_name, results['model'], X_test, y_test,
                results.get('predictions'), results.get('probabilities')
            )
            evaluation_results[model_name] = model_eval

            # Add threshold optimization for models with probabilities
            if model_eval.get('probabilities') is not None:
                threshold_analysis = evaluator.optimize_threshold_detailed(
                    y_test, model_eval['probabilities']
                )
                evaluation_results[model_name]['threshold_analysis'] = threshold_analysis

    # Create visualizations
    evaluator.create_evaluation_visualizations(evaluation_results, X_test, y_test)

    # Generate summary
    summary = evaluator.generate_evaluation_summary(evaluation_results)

    return {
        'evaluation_results': evaluation_results,
        'summary': summary,
        'evaluator': evaluator
    }

if __name__ == "__main__":
    # This would be called from the main pipeline
    pass
