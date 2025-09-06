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

class BankPredictionPipeline:
    """Main pipeline class for bank term deposit prediction."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the pipeline."""
        self.data_path = data_path or PROJECT_ROOT / "data" / DATA_CONFIG['file_name']
        self.results = {}

        if not self.data_path.exists():
            print(f"⚠️  Data file not found at: {self.data_path}")
            print(f"   Please place 'bank_data.csv' in the 'data/' directory")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete machine learning pipeline."""

        print("="*100)
        print("🏦 BANK TERM DEPOSIT PREDICTION - MACHINE LEARNING PIPELINE")
        print("="*100)
        print(f"Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Step 1: Data Processing
            print("\n🔍 STEP 1: DATA PROCESSING")
            cleaned_df, data_summary = main_data_processing(self.data_path)
            self.results['data_processing'] = {'cleaned_df': cleaned_df, 'summary': data_summary}

            # Step 2: Feature Engineering
            print("\n⚙️ STEP 2: FEATURE ENGINEERING")
            processed_data = main_feature_engineering(cleaned_df)
            self.results['feature_engineering'] = processed_data

            # Step 3: Model Training
            print("\n🧠 STEP 3: MODEL TRAINING")
            training_results = main_model_training(processed_data)
            self.results['model_training'] = training_results

            # Step 4: Model Evaluation
            print("\n📊 STEP 4: MODEL EVALUATION")
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
            print("\n❌ Data file not found!")
            print("Please ensure 'bank_data.csv' is in the 'data/' directory")
            raise
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise

    def print_pipeline_summary(self):
        """Print comprehensive pipeline execution summary."""
        print("\n🎯 " + "="*100)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*100)

        print(f"📅 Completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Data Processing Summary
        if 'data_processing' in self.results:
            data_info = self.results['data_processing']['summary']['basic_info']
            print(f"\n📋 DATA: {data_info['shape']} shape, {len(data_info['columns'])} features")

        # Feature Engineering Summary
        if 'feature_engineering' in self.results:
            fe_results = self.results['feature_engineering']
            print(f"🔧 FEATURES: {len(fe_results['feature_names'])} final features")
            print(f"   Train: {fe_results['X_train'].shape}, Test: {fe_results['X_test'].shape}")

        # Model Training Summary
        if 'model_training' in self.results:
            training_info = self.results['model_training']
            print(f"🤖 MODELS: {len(training_info['training_results'])} trained")

            if 'model_comparison' in training_info and not training_info['model_comparison'].empty:
                best_model = training_info['model_comparison'].iloc[0]
                print(f"   Best: {best_model['Model']} (F1: {best_model['F1-Score']:.4f})")

        # Evaluation Summary
        if 'evaluation' in self.results:
            eval_info = self.results['evaluation']
            best_model = eval_info['summary'].get('best_model', 'N/A')
            print(f"📈 EVALUATION: Best model - {best_model}")

        print(f"\n📂 Results saved to: {PROJECT_ROOT}/models/ and {PROJECT_ROOT}/results/")
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*100)

def main():
    """Main function to run the complete pipeline."""
    print("🏦 Bank Term Deposit Prediction Pipeline")
    print("Predicting customer subscription using marketing campaign data")

    pipeline = BankPredictionPipeline()

    try:
        results = pipeline.run_complete_pipeline()
        print("\n✅ Pipeline completed successfully!")
        print("Check the results/ directory for visualizations")
        print("Check the models/ directory for saved models")
        return results
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return None

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    results = main()