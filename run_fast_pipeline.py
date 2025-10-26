#!/usr/bin/env python3
"""
Fast test pipeline - temporarily excludes SVM for quick testing.
Use this to verify the pipeline works correctly.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Modify config for fast testing only
import config

# Temporarily remove SVM for testing (uncomment for full run)
if 'svm' in config.MODEL_CONFIG:
    del config.MODEL_CONFIG['svm']
    print("NOTE: SVM temporarily excluded for fast testing")

# Reduce tuning for testing
config.TUNING_CONFIG['n_trials'] = 10
config.TUNING_CONFIG['cv_folds'] = 3

print("="*80)
print("FAST TEST CONFIGURATION (not for final results)")
print("="*80)
print(f"Models: {list(config.MODEL_CONFIG.keys())}")
print(f"Tuning trials: {config.TUNING_CONFIG['n_trials']}")
print(f"CV folds: {config.TUNING_CONFIG['cv_folds']}")
print("="*80)
print()

from main import main_enhanced

if __name__ == "__main__":
    print("Running Fast Test Pipeline (SVM excluded for speed)")
    print("For full results with all models, run: python src/main.py")
    print("="*80)
    
    try:
        results = main_enhanced()
        
        if results:
            print("\nFast test pipeline completed successfully")
            print("\nThis was a quick test. For full results with all models, run:")
            print("  python src/main.py")
            
        else:
            print("\nPipeline failed to complete")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

