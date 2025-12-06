"""
MODULE 3: DEEP LEARNING AND INTELLIGENT VISION (ANALYSIS ONLY - NO TRAINING)
============================================================================
Runs Module 3 analysis: Model loading, explainability, comparison, reports.
EXCLUDES: Fine-tuning and YOLO training (assumes already done on Colab)
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')
from complete_all_modules import CompleteBoneFractureSystem

def main():
    """Run Module 3: Deep Learning Analysis (No Training)"""
    dataset_root = r'data\archive\bone fracture detection.v4-v4.yolov8'
    
    # Normalize path
    dataset_root = str(Path(dataset_root).resolve())
    
    if not Path(dataset_root).exists():
        print(f"[ERROR] Dataset not found at: {dataset_root}")
        print("Please check the dataset path")
        return
    
    # Initialize system
    system = CompleteBoneFractureSystem(dataset_root)
    
    # Run Module 3 analysis only (NO TRAINING)
    print("="*80)
    print("RUNNING MODULE 3: DEEP LEARNING ANALYSIS (NO TRAINING)")
    print("="*80)
    print("NOTE: Skipping fine-tuning and YOLO training (assumed already done on Colab)")
    print("="*80)
    
    dl_model, comparison_results, explainability_results = system.module_3_deep_learning_complete()
    
    print("\n" + "="*80)
    print("[SUCCESS] MODULE 3 ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved in: {system.results_dir}/")
    print(f"Report saved in: {system.reports_dir}/Module3_Deep_Learning_Final_Report.md")
    print("="*80)

if __name__ == '__main__':
    main()

