"""
Main pipeline orchestrator for Bank Term Deposit Prediction Project.
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DATA_CONFIG, PROJECT_ROOT
from data_processing import main_data_processing
from feature_engineering import main_feature_engineering
from model_training import main_model_training
from evaluation import main_evaluation
from enhanced_model_training import main_enhanced_model_training
from enhanced_evaluation import main_enhanced_evaluation

class BankPredictionPipeline:
    """Main pipeline class for bank term deposit prediction."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the pipeline."""
        self.data_path = data_path or PROJECT_ROOT / "data" / DATA_CONFIG['file_name']
        self.results = {}

        if not self.data_path.exists():
            print(f"WARNING: Data file not found at: {self.data_path}")
            print(f"   Please place 'bank_data.csv' in the 'data/' directory")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete machine learning pipeline."""

        print("="*100)
        print("BANK TERM DEPOSIT PREDICTION - MACHINE LEARNING PIPELINE")
        print("="*100)
        print(f"Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Data Processing
            print("\nSTEP 1: DATA PROCESSING")
            cleaned_df, data_summary = main_data_processing(self.data_path)
            self.results['data_processing'] = {'cleaned_df': cleaned_df, 'summary': data_summary}

            # Step 2: Feature Engineering
            print("\nSTEP 2: FEATURE ENGINEERING")
            processed_data = main_feature_engineering(cleaned_df)
            self.results['feature_engineering'] = processed_data

            # Step 3: Model Training
            print("\nSTEP 3: MODEL TRAINING")
            training_results = main_model_training(processed_data)
            self.results['model_training'] = training_results

            # Step 4: Model Evaluation
            print("\nSTEP 4: MODEL EVALUATION")
            evaluation_results = main_evaluation(
                training_results['training_results'],
                processed_data['X_test'],
                processed_data['y_test']
            )
            self.results['evaluation'] = evaluation_results

            # Generate final summary
            self.print_pipeline_summary()

            return self.results

        except FileNotFoundError:
            print("\nERROR: Data file not found!")
            print("Please ensure 'bank_data.csv' is in the 'data/' directory")
            raise
        except Exception as e:
            print(f"\nERROR: Pipeline failed: {e}")
            raise

    def run_enhanced_pipeline(self) -> Dict[str, Any]:
        """Execute the enhanced machine learning pipeline with advanced features."""

        print("="*100)
        print("BANK TERM DEPOSIT PREDICTION - ENHANCED ML PIPELINE")
        print("="*100)
        print(f"Enhanced pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Data Processing
            print("\nSTEP 1: DATA PROCESSING")
            cleaned_df, data_summary = main_data_processing(self.data_path)
            self.results['data_processing'] = {'cleaned_df': cleaned_df, 'summary': data_summary}

            # Step 2: Feature Engineering
            print("\nSTEP 2: FEATURE ENGINEERING")
            processed_data = main_feature_engineering(cleaned_df)
            self.results['feature_engineering'] = processed_data

            # Step 3: Enhanced Model Training (with hyperparameter tuning, stacking)
            print("\nSTEP 3: ENHANCED MODEL TRAINING")
            print("   - Training multiple models (RF, XGBoost, LightGBM, SVM, etc.)")
            print("   - Hyperparameter tuning with Optuna")
            print("   - Threshold optimization")
            print("   - Model stacking/ensemble")
            
            enhanced_training_results = main_enhanced_model_training(processed_data)
            self.results['enhanced_training'] = enhanced_training_results

            # Step 4: Enhanced Model Evaluation
            print("\nSTEP 4: ENHANCED MODEL EVALUATION")
            print("   - Advanced metrics (ROC-AUC, Precision-Recall, Calibration)")
            print("   - Comprehensive visualizations")
            print("   - Threshold analysis")
            
            enhanced_evaluation_results = main_enhanced_evaluation(
                enhanced_training_results['training_results'],
                processed_data['X_test'],
                processed_data['y_test']
            )
            self.results['enhanced_evaluation'] = enhanced_evaluation_results

            # Generate enhanced summary
            self.print_enhanced_pipeline_summary()

            return self.results

        except FileNotFoundError:
            print("\nERROR: Data file not found!")
            print("Please ensure 'bank_data.csv' is in the 'data/' directory")
            raise
        except Exception as e:
            print(f"\nERROR: Enhanced pipeline failed: {e}")
            raise

    def print_pipeline_summary(self):
        """Print comprehensive pipeline execution summary."""
        print("\n" + "="*100)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*100)

        print(f"Completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Data Processing Summary
        if 'data_processing' in self.results:
            data_info = self.results['data_processing']['summary']['basic_info']
            print(f"\nDATA: {data_info['shape']} shape, {len(data_info['columns'])} features")

        # Feature Engineering Summary
        if 'feature_engineering' in self.results:
            fe_results = self.results['feature_engineering']
            print(f"FEATURES: {len(fe_results['feature_names'])} final features")
            print(f"   Train: {fe_results['X_train'].shape}, Test: {fe_results['X_test'].shape}")

        # Model Training Summary
        if 'model_training' in self.results:
            training_info = self.results['model_training']
            print(f"MODELS: {len(training_info['training_results'])} trained")

            if 'model_comparison' in training_info and not training_info['model_comparison'].empty:
                best_model = training_info['model_comparison'].iloc[0]
                print(f"   Best: {best_model['Model']} (F1: {best_model['F1-Score']:.4f})")

        # Evaluation Summary
        if 'evaluation' in self.results:
            eval_info = self.results['evaluation']
            best_model = eval_info['summary'].get('best_model', 'N/A')
            print(f"EVALUATION: Best model - {best_model}")

        print(f"\nResults saved to: {PROJECT_ROOT}/models/ and {PROJECT_ROOT}/results/")
        print("\nPIPELINE COMPLETED SUCCESSFULLY")
        print("="*100)

    def print_enhanced_pipeline_summary(self):
        """Print comprehensive enhanced pipeline execution summary."""
        print("\n" + "="*100)
        print("ENHANCED PIPELINE EXECUTION SUMMARY")
        print("="*100)

        print(f"Completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Data Processing Summary
        if 'data_processing' in self.results:
            data_info = self.results['data_processing']['summary']['basic_info']
            print(f"\nDATA: {data_info['shape']} shape, {len(data_info['columns'])} features")

        # Feature Engineering Summary
        if 'feature_engineering' in self.results:
            fe_results = self.results['feature_engineering']
            print(f"FEATURES: {len(fe_results['feature_names'])} final features")
            print(f"   Train: {fe_results['X_train'].shape}, Test: {fe_results['X_test'].shape}")

        # Enhanced Training Summary
        if 'enhanced_training' in self.results:
            training_info = self.results['enhanced_training']
            print(f"ENHANCED MODELS: {len(training_info['training_results'])} trained")
            
            if 'model_comparison' in training_info and not training_info['model_comparison'].empty:
                best_model = training_info['model_comparison'].iloc[0]
                print(f"   Best: {best_model['Model']} (F1: {best_model['F1-Score']:.4f})")
            
            if 'tuning_results' in training_info:
                tuned_count = len(training_info['tuning_results'])
                print(f"   Tuned: {tuned_count} models with hyperparameter optimization")
            
            if 'stacking_results' in training_info:
                print(f"   Stacking: Model ensemble created")

        # Enhanced Evaluation Summary
        if 'enhanced_evaluation' in self.results:
            eval_info = self.results['enhanced_evaluation']
            best_model = eval_info['summary'].get('overall_best_model', 'N/A')
            print(f"ENHANCED EVALUATION: Best model - {best_model}")
            
            if 'best_models' in eval_info['summary']:
                print("   Best models by metric:")
                for metric, info in eval_info['summary']['best_models'].items():
                    print(f"     {metric}: {info['model']} ({info['score']:.4f})")

        print(f"\nEnhanced results saved to: {PROJECT_ROOT}/models/ and {PROJECT_ROOT}/results/")
        print("\nENHANCED PIPELINE COMPLETED SUCCESSFULLY")
        print("="*100)

def main(use_enhanced: bool = True):
    """Main function to run the complete pipeline."""
    print("Bank Term Deposit Prediction Pipeline")
    print("Predicting customer subscription using marketing campaign data")

    pipeline = BankPredictionPipeline()

    try:
        if use_enhanced:
            print("\nRunning ENHANCED pipeline with:")
            print("   - Multiple advanced models (RF, XGBoost, LightGBM, SVM)")
            print("   - Hyperparameter tuning with Optuna")
            print("   - Threshold optimization")
            print("   - Model stacking/ensemble")
            print("   - Advanced evaluation metrics")
            
            results = pipeline.run_enhanced_pipeline()
            print("\nEnhanced pipeline completed successfully")
        else:
            print("\nRunning BASIC pipeline with:")
            print("   - Naive Bayes, Decision Trees")
            print("   - Basic evaluation metrics")
            
            results = pipeline.run_complete_pipeline()
            print("\nBasic pipeline completed successfully")
        
        print("Check the results/ directory for visualizations")
        print("Check the models/ directory for saved models")
        return results
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        return None

def main_basic():
    """Run the basic pipeline."""
    return main(use_enhanced=False)

def main_enhanced():
    """Run the enhanced pipeline."""
    return main(use_enhanced=True)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    results = main()