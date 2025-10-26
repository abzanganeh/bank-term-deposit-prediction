"""
Enhanced model training for Bank Term Deposit Prediction.
This module extends the basic models with advanced machine learning techniques:
- Multiple algorithms (Random Forest, XGBoost, LightGBM, SVM, Logistic Regression)
- Automated hyperparameter tuning using Optuna
- Threshold optimization for better predictions
- Model stacking for ensemble predictions
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import LabelEncoder

# Advanced ML libraries
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

from config import (
    MODEL_CONFIG, TUNING_CONFIG, HYPERPARAMETER_SPACES,
    THRESHOLD_CONFIG, STACKING_CONFIG, MODELS_DIR
)

class EnhancedBankModelTrainer:
    """
    Trains multiple ML models with hyperparameter optimization.
    Combines individual models into an ensemble for improved performance.
    """

    def __init__(self):
        """Set up the trainer with empty containers for models and results."""
        self.models = {}
        self.trained_models = {}
        self.tuned_models = {}
        self.training_results = {}
        self.tuning_results = {}
        self.threshold_results = {}
        self.stacking_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_thresholds = {}
        self.label_encoder = LabelEncoder()
        self.needs_encoding = {}

    def initialize_models(self) -> Dict[str, Any]:
        """Create instances of all configured models."""
        print("Initializing models...")

        for model_name, config in MODEL_CONFIG.items():
            model_type = config['model_type']
            params = config['params']

            try:
                if model_type == 'GaussianNB':
                    self.models[model_name] = GaussianNB(**params)
                elif model_type == 'DecisionTreeClassifier':
                    self.models[model_name] = DecisionTreeClassifier(**params)
                elif model_type == 'RandomForestClassifier':
                    self.models[model_name] = RandomForestClassifier(**params)
                elif model_type == 'XGBClassifier':
                    if XGBOOST_AVAILABLE:
                        self.models[model_name] = XGBClassifier(**params)
                    else:
                        print(f"  [WARNING] Skipping {model_name}: XGBoost not available")
                        continue
                elif model_type == 'LGBMClassifier':
                    if LIGHTGBM_AVAILABLE:
                        self.models[model_name] = LGBMClassifier(**params)
                    else:
                        print(f"  [WARNING] Skipping {model_name}: LightGBM not available")
                        continue
                elif model_type == 'SVC':
                    self.models[model_name] = SVC(**params)
                elif model_type == 'LogisticRegression':
                    self.models[model_name] = LogisticRegression(**params)
                else:
                    print(f"Warning: Unknown model type {model_type}")
                    continue

                print(f"  [OK] {model_name}: {model_type}")
            except Exception as e:
                print(f"  ❌ Error initializing {model_name}: {e}")
                continue

        return self.models

    def train_single_model(self, model_name: str, model: Any,
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train a model and calculate performance metrics.
        Handles label encoding for XGBoost and LightGBM automatically.
        """
        print(f"\nTraining {model_name}...", flush=True)
        
        import time
        start_time = time.time()

        try:
            # Check if model needs encoded labels (XGBoost, LightGBM)
            model_type = type(model).__name__
            needs_encoding = model_type in ['XGBClassifier', 'LGBMClassifier']
            
            if needs_encoding:
                # Encode labels for XGBoost and LightGBM
                y_train_encoded = self.label_encoder.fit_transform(y_train)
                y_test_encoded = self.label_encoder.transform(y_test)
                self.needs_encoding[model_name] = True
                
                # Train the model with encoded labels
                model.fit(X_train, y_train_encoded)
                
                # Make predictions and decode
                y_pred_encoded = model.predict(X_test)
                y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            else:
                self.needs_encoding[model_name] = False
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
            y_pred_proba = None
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            pos_label = 'yes' if 'yes' in y_test.unique() else y_test.unique()[1]
            precision = precision_score(y_test, y_pred, pos_label=pos_label, average='binary')
            recall = recall_score(y_test, y_pred, pos_label=pos_label, average='binary')
            f1 = f1_score(y_test, y_pred, pos_label=pos_label, average='binary')
            
            # Calculate ROC-AUC if probabilities are available
            roc_auc = None
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y_test, y_pred_proba)

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
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix
            }

            elapsed = time.time() - start_time
            roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
            print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc_str} (trained in {elapsed:.1f}s)")
            return results

        except Exception as e:
            print(f"  [ERROR] Error training {model_name}: {e}")
            return None

    def hyperparameter_tuning(self, model_name: str, model: Any,
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Find optimal hyperparameters using Bayesian optimization.
        Tests different parameter combinations and selects the best based on cross-validation.
        """
        if not TUNING_CONFIG['enable_tuning'] or model_name not in HYPERPARAMETER_SPACES:
            return None

        print(f"\n[TUNING] Hyperparameter optimization for {model_name}...")

        if not OPTUNA_AVAILABLE:
            print("  [WARNING] Optuna not available, skipping tuning")
            return None

        def objective(trial):
            # Get parameter space for this model
            param_space = HYPERPARAMETER_SPACES[model_name]
            
            # Sample parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(x, (int, float)) for x in param_values):
                        if all(isinstance(x, int) for x in param_values):
                            params[param_name] = trial.suggest_categorical(param_name, param_values)
                        else:
                            params[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    params[param_name] = param_values

            # Create model with sampled parameters
            model_type = MODEL_CONFIG[model_name]['model_type']
            if model_type == 'RandomForestClassifier':
                tuned_model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif model_type == 'XGBClassifier' and XGBOOST_AVAILABLE:
                tuned_model = XGBClassifier(**params, random_state=42, eval_metric='logloss')
            elif model_type == 'LGBMClassifier' and LIGHTGBM_AVAILABLE:
                tuned_model = LGBMClassifier(**params, random_state=42, verbose=-1)
            elif model_type == 'SVC':
                tuned_model = SVC(**params, random_state=42, probability=True)
            elif model_type == 'LogisticRegression':
                tuned_model = LogisticRegression(**params, random_state=42, max_iter=1000)
            else:
                return 0.0

            # Cross-validation
            try:
                # Check if model needs encoded labels
                model_type_name = type(tuned_model).__name__
                
                # Use appropriate scoring based on model type
                scoring_metric = TUNING_CONFIG['scoring']
                
                if model_type_name in ['XGBClassifier', 'LGBMClassifier']:
                    # Encode labels for cross-validation
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    cv_scores = cross_val_score(
                        tuned_model, X_train, y_train_encoded,
                        cv=StratifiedKFold(n_splits=TUNING_CONFIG['cv_folds'], shuffle=True, random_state=42),
                        scoring=scoring_metric,
                        n_jobs=1
                    )
                else:
                    # For string labels, ensure we use metrics that work with them
                    # ROC-AUC works with both string and numeric labels
                    cv_scores = cross_val_score(
                        tuned_model, X_train, y_train,
                        cv=StratifiedKFold(n_splits=TUNING_CONFIG['cv_folds'], shuffle=True, random_state=42),
                        scoring=scoring_metric,
                        n_jobs=1
                    )
                
                mean_score = cv_scores.mean()
                # Return 0 if NaN to avoid optuna errors
                if np.isnan(mean_score):
                    print(f"Warning: NaN score in trial, returning 0.0")
                    return 0.0
                return mean_score
            except Exception as e:
                print(f"Error in trial: {str(e)[:100]}")
                return 0.0

        # Run optimization with progress callback
        study = optuna.create_study(direction='maximize')
        
        def progress_callback(study, trial):
            """Show progress during optimization."""
            completed = len([t for t in study.trials if t.state.name == 'COMPLETE'])
            total = TUNING_CONFIG['n_trials']
            best_value = study.best_value if study.best_value else 0.0
            
            print(f"  Trial {completed}/{total} complete | Best {TUNING_CONFIG['scoring']}: {best_value:.4f}", flush=True)
        
        print(f"  Running {TUNING_CONFIG['n_trials']} optimization trials (this may take a few minutes)...")
        import time
        start_time = time.time()
        
        study.optimize(objective, n_trials=TUNING_CONFIG['n_trials'], callbacks=[progress_callback])
        
        elapsed_time = time.time() - start_time
        print(f"  Optimization completed in {elapsed_time:.1f} seconds")

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value

        if best_score > 0:
            print(f"  Best {TUNING_CONFIG['scoring']}: {best_score:.4f}")
            print(f"  Best parameters: {best_params}")
        else:
            print(f"  [WARNING] Tuning did not improve score (best: {best_score:.4f})")
            print(f"  Using parameters: {best_params}")

        # Train model with best parameters
        model_type = MODEL_CONFIG[model_name]['model_type']
        if model_type == 'RandomForestClassifier':
            best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        elif model_type == 'XGBClassifier' and XGBOOST_AVAILABLE:
            best_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        elif model_type == 'LGBMClassifier' and LIGHTGBM_AVAILABLE:
            best_model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        elif model_type == 'SVC':
            best_model = SVC(**best_params, random_state=42, probability=True)
        elif model_type == 'LogisticRegression':
            best_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
        else:
            return None

        # Train with appropriate label encoding
        if model_type in ['XGBClassifier', 'LGBMClassifier']:
            # Encode labels for XGBoost and LightGBM
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            best_model.fit(X_train, y_train_encoded)
        else:
            best_model.fit(X_train, y_train)

        return {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }

    def optimize_threshold(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Find the best probability threshold for classification.
        Default threshold is 0.5, but other values may give better results.
        """
        if not THRESHOLD_CONFIG['enable_threshold_optimization']:
            return 0.5

        if not hasattr(model, 'predict_proba'):
            return 0.5

        print(f"\n[THRESHOLD] Optimizing decision threshold...", flush=True)

        # Get prediction probabilities
        y_proba = model.predict_proba(X_val)[:, 1]

        # Test different thresholds
        thresholds = np.arange(
            THRESHOLD_CONFIG['threshold_range'][0],
            THRESHOLD_CONFIG['threshold_range'][1],
            THRESHOLD_CONFIG['threshold_step']
        )

        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            y_pred_thresh = ['yes' if x == 1 else 'no' for x in y_pred_thresh]
            
            if THRESHOLD_CONFIG['optimization_metric'] == 'f1':
                score = f1_score(y_val, y_pred_thresh, pos_label='yes')
            elif THRESHOLD_CONFIG['optimization_metric'] == 'precision':
                score = precision_score(y_val, y_pred_thresh, pos_label='yes')
            elif THRESHOLD_CONFIG['optimization_metric'] == 'recall':
                score = recall_score(y_val, y_pred_thresh, pos_label='yes')
            else:
                score = f1_score(y_val, y_pred_thresh, pos_label='yes')

            if score > best_score:
                best_score = score
                best_threshold = threshold

        print(f"  Best threshold: {best_threshold:.3f} (Score: {best_score:.4f})")
        return best_threshold

    def create_model_stacking(self, trained_models: Dict[str, Any],
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Combine multiple models into a voting ensemble.
        The ensemble makes predictions by averaging probabilities from all base models.
        """
        if not STACKING_CONFIG['enable_stacking']:
            return None

        print(f"\n[STACKING] Creating ensemble model...", flush=True)

        # Select models for stacking (exclude models without predict_proba)
        stacking_models = []
        for name, results in trained_models.items():
            if results and hasattr(results['model'], 'predict_proba'):
                stacking_models.append((name, results['model']))

        if len(stacking_models) < 2:
            print("  [WARNING] Not enough models with probabilities for stacking")
            return None

        # Create voting classifier
        if STACKING_CONFIG['stacking_method'] == 'voting':
            voting_clf = VotingClassifier(
                estimators=stacking_models,
                voting='soft' if STACKING_CONFIG['use_proba'] else 'hard'
            )
            
            voting_clf.fit(X_train, y_train)
            y_pred = voting_clf.predict(X_test)
            y_pred_proba = voting_clf.predict_proba(X_test)[:, 1] if STACKING_CONFIG['use_proba'] else None

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label='yes')
            recall = recall_score(y_test, y_pred, pos_label='yes')
            f1 = f1_score(y_test, y_pred, pos_label='yes')
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

            stacking_results = {
                'model': voting_clf,
                'model_name': 'stacking_voting',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'base_models': [name for name, _ in stacking_models]
            }

            print(f"  Stacking Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
            return stacking_results

        return None

    def train_all_models_enhanced(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Main training workflow:
        1. Train baseline versions of all models
        2. Tune hyperparameters for applicable models
        3. Optimize prediction thresholds
        4. Create ensemble model from best performers
        """
        print("="*80)
        print("ENHANCED MODEL TRAINING PIPELINE")
        print("="*80)

        # Initialize models
        self.initialize_models()

        # Create validation set for tuning
        from sklearn.model_selection import train_test_split
        X_train_tune, X_val, y_train_tune, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Train and tune models
        total_models = len(self.models)
        for idx, (model_name, model) in enumerate(self.models.items(), 1):
            print(f"\n{'='*60}")
            print(f"Processing {model_name.upper()} ({idx}/{total_models})")
            print(f"{'='*60}", flush=True)

            # 1. Train baseline model
            baseline_results = self.train_single_model(
                model_name, model, X_train, y_train, X_test, y_test
            )
            if baseline_results:
                self.training_results[model_name] = baseline_results
                self.trained_models[model_name] = model

            # 2. Hyperparameter tuning
            tuning_results = self.hyperparameter_tuning(
                model_name, model, X_train_tune, y_train_tune, X_val, y_val
            )
            if tuning_results:
                self.tuning_results[model_name] = tuning_results
                self.tuned_models[model_name] = tuning_results['model']

                # Evaluate tuned model
                tuned_results = self.train_single_model(
                    f"{model_name}_tuned", tuning_results['model'], X_train, y_train, X_test, y_test
                )
                if tuned_results:
                    self.training_results[f"{model_name}_tuned"] = tuned_results

            # 3. Threshold optimization
            if baseline_results and hasattr(baseline_results['model'], 'predict_proba'):
                best_threshold = self.optimize_threshold(baseline_results['model'], X_val, y_val)
                self.best_thresholds[model_name] = best_threshold

        # 4. Model stacking
        stacking_results = self.create_model_stacking(
            self.training_results, X_train, y_train, X_test, y_test
        )
        if stacking_results:
            self.stacking_results = stacking_results
            self.training_results['stacking_voting'] = stacking_results

        return self.training_results

    def compare_all_models(self) -> pd.DataFrame:
        """Compare performance of all trained models."""
        if not self.training_results:
            return pd.DataFrame()

        comparison_data = []
        for model_name, results in self.training_results.items():
            if results:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'ROC-AUC': results.get('roc_auc', 'N/A')
                })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

        print("\n" + "="*80)
        print("ENHANCED MODEL COMPARISON")
        print("="*80)
        print(comparison_df.round(4))

        return comparison_df

    def save_enhanced_models(self) -> Dict[str, str]:
        """Save all trained and tuned models."""
        saved_paths = {}
        print(f"\n[SAVE] Saving enhanced models to {MODELS_DIR}...")

        # Save baseline models
        for model_name, model in self.trained_models.items():
            filename = f"{model_name}_enhanced_{self.timestamp}.pkl"
            filepath = MODELS_DIR / filename
            joblib.dump(model, filepath)
            saved_paths[model_name] = str(filepath)
            print(f"  [OK] {model_name} -> {filename}")

        # Save tuned models
        for model_name, model in self.tuned_models.items():
            filename = f"{model_name}_tuned_{self.timestamp}.pkl"
            filepath = MODELS_DIR / filename
            joblib.dump(model, filepath)
            saved_paths[f"{model_name}_tuned"] = str(filepath)
            print(f"  [OK] {model_name}_tuned -> {filename}")

        # Save stacking model
        if self.stacking_results:
            filename = f"stacking_voting_{self.timestamp}.pkl"
            filepath = MODELS_DIR / filename
            joblib.dump(self.stacking_results['model'], filepath)
            saved_paths['stacking_voting'] = str(filepath)
            print(f"  [OK] stacking_voting -> {filename}")

        # Save thresholds
        if self.best_thresholds:
            filename = f"best_thresholds_{self.timestamp}.pkl"
            filepath = MODELS_DIR / filename
            joblib.dump(self.best_thresholds, filepath)
            saved_paths['thresholds'] = str(filepath)
            print(f"  [OK] thresholds -> {filename}")

        return saved_paths

def main_enhanced_model_training(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train and optimize multiple ML models.
    Returns trained models, tuning results, and performance metrics.
    """
    print("="*80)
    print("BANK TERM DEPOSIT PREDICTION - ENHANCED MODEL TRAINING")
    print("="*80)

    trainer = EnhancedBankModelTrainer()

    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']

    # Run enhanced training pipeline
    training_results = trainer.train_all_models_enhanced(X_train, y_train, X_test, y_test)
    
    # Compare models
    comparison_df = trainer.compare_all_models()
    
    # Save models
    saved_paths = trainer.save_enhanced_models()

    return {
        'trainer': trainer,
        'training_results': training_results,
        'tuning_results': trainer.tuning_results,
        'stacking_results': trainer.stacking_results,
        'best_thresholds': trainer.best_thresholds,
        'model_comparison': comparison_df,
        'saved_model_paths': saved_paths,
        'trained_models': trainer.trained_models,
        'tuned_models': trainer.tuned_models
    }

if __name__ == "__main__":
    # This would be called from the main pipeline
    pass
