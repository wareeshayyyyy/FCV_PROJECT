"""
MODULE 2: CLASSICAL FEATURES - FEATURE EXTRACTION AND MACHINE LEARNING
=======================================================================
Runs Module 2: Classical feature extraction with SIFT, SURF, HOG, BoVW, etc.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')
from complete_all_modules import CompleteBoneFractureSystem

def main():
    """Run Module 2: Classical Features"""
    dataset_root = r'data\archive\bone fracture detection.v4-v4.yolov8'
    
    # Normalize path
    dataset_root = str(Path(dataset_root).resolve())
    
    if not Path(dataset_root).exists():
        print(f"❌ Dataset not found at: {dataset_root}")
        print("Please check the dataset path")
        return
    
    # Initialize system
    system = CompleteBoneFractureSystem(dataset_root)
    
    # Run Module 2 only
    print("="*80)
    print("RUNNING MODULE 2: CLASSICAL FEATURES")
    print("="*80)
    
    features_df, classification_results, ablation_results = system.module_2_classical_features()
    
    print("\n" + "="*80)
    print("✅ MODULE 2 COMPLETE!")
    print("="*80)
    print(f"📁 Results saved in: {system.results_dir}/")
    print(f"📄 Report saved in: {system.reports_dir}/Module2_Classical_Features_Midterm_Report.md")
    print("="*80)

if __name__ == '__main__':
    main()

