#!/usr/bin/env python3
"""
Quick test script for the enhanced pipeline.
Runs with reduced tuning for faster execution.
"""

import sys
from pathlib import Path

# Add src directory to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Temporarily disable tuning for quick test
import config
config.TUNING_CONFIG['enable_tuning'] = False
config.TUNING_CONFIG['n_trials'] = 5
config.STACKING_CONFIG['enable_stacking'] = True
config.THRESHOLD_CONFIG['enable_threshold_optimization'] = True

from main import main_enhanced

if __name__ == "__main__":
    print("🚀 Starting Quick Test of Enhanced Pipeline")
    print("   (Tuning disabled for faster execution)")
    print("="*80)
    
    try:
        results = main_enhanced()
        
        if results:
            print("\n✅ Quick test completed successfully!")
            
        else:
            print("\n❌ Pipeline failed to complete")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


