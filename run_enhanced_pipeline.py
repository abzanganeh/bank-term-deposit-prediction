#!/usr/bin/env python3
"""
Run the full enhanced pipeline with all advanced features.

This script executes the complete machine learning workflow including:
- Training 8 different models (Naive Bayes, Decision Trees, Random Forest, 
  XGBoost, LightGBM, SVM, Logistic Regression)
- Hyperparameter tuning for tree-based and gradient boosting models
- Threshold optimization to find best classification cutoffs
- Model stacking to create an ensemble predictor

Execution time: Approximately 10-15 minutes
Output: Trained models saved to models/, visualizations to results/
"""

import sys
from pathlib import Path

# Add src directory to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import main_enhanced

if __name__ == "__main__":
    print("Starting Enhanced Bank Term Deposit Prediction Pipeline")
    print("="*80)
    
    try:
        results = main_enhanced()
        
        if results:
            print("\nEnhanced pipeline completed successfully")
            print("\nKey Results:")
            
            # Print best model information
            if 'enhanced_evaluation' in results:
                eval_summary = results['enhanced_evaluation']['summary']
                print(f"   Best Model: {eval_summary.get('overall_best_model', 'N/A')}")
                
                if 'best_models' in eval_summary:
                    print("   Best models by metric:")
                    for metric, info in eval_summary['best_models'].items():
                        print(f"     {metric}: {info['model']} ({info['score']:.4f})")
            
            print(f"\nResults saved to:")
            print(f"   Models: {PROJECT_ROOT}/models/")
            print(f"   Visualizations: {PROJECT_ROOT}/results/")
            
        else:
            print("\nPipeline failed to complete")
            
    except Exception as e:
        print(f"\nError running enhanced pipeline: {e}")
        sys.exit(1)
