#!/usr/bin/env python3
"""
Optimized pipeline configuration for faster execution.

Modifications from full pipeline:
- SVM excluded (training time > 5 minutes)
- Reduced tuning trials: 10 instead of 20
- Reduced CV folds: 3 instead of 5

Still includes: Random Forest, XGBoost, LightGBM, Logistic Regression
with hyperparameter tuning, threshold optimization, and model stacking.

Execution time: Approximately 5-8 minutes
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config

# Remove slow models
if 'svm' in config.MODEL_CONFIG:
    del config.MODEL_CONFIG['svm']

# Optimize tuning settings for speed
config.TUNING_CONFIG['n_trials'] = 15
config.TUNING_CONFIG['cv_folds'] = 3

# Keep threshold optimization and stacking enabled
config.THRESHOLD_CONFIG['enable_threshold_optimization'] = True
config.STACKING_CONFIG['enable_stacking'] = True

print("="*80)
print("OPTIMIZED PIPELINE CONFIGURATION")
print("="*80)
print(f"Models to train: {len(config.MODEL_CONFIG)}")
print(f"  {', '.join(config.MODEL_CONFIG.keys())}")
print(f"\nTuning configuration:")
print(f"  Trials per model: {config.TUNING_CONFIG['n_trials']}")
print(f"  Cross-validation folds: {config.TUNING_CONFIG['cv_folds']}")
print(f"  Scoring metric: {config.TUNING_CONFIG['scoring']}")
print(f"\nThreshold optimization: Enabled")
print(f"Model stacking: Enabled")
print("="*80)
print()

from main import main_enhanced

if __name__ == "__main__":
    try:
        results = main_enhanced()
        
        if results:
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            
            # Print quick summary
            if 'enhanced_evaluation' in results:
                summary = results['enhanced_evaluation']['summary']
                best = summary.get('overall_best_model', 'N/A')
                print(f"\nBest Overall Model: {best}")
                
                if 'best_models' in summary:
                    print("\nBest models by metric:")
                    for metric, info in summary['best_models'].items():
                        print(f"  {metric:15s}: {info['model']:30s} ({info['score']:.4f})")
            
            print(f"\nResults saved to:")
            print(f"  Models: {PROJECT_ROOT}/models/")
            print(f"  Visualizations: {PROJECT_ROOT}/results/")
            
        else:
            print("\nPipeline failed to complete")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

