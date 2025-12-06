"""
MODULE 1: FOUNDATIONS OF VISION AND IMAGE ANALYSIS
==================================================
Complete preprocessing pipeline with:
- Geometric & Intensity Transformations
- Filtering & Edge Extraction
- Noise Modeling & Restoration
- Comparative Studies & Metrics
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.append('src')
from bonefracture.bone_yolo_dataset import BoneFractureDatasetYOLO


class Module1Preprocessing:
    """Module 1: Foundations - Image Preprocessing and Enhancement"""
    
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.results_dir = 'module1_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        self.train_dataset = BoneFractureDatasetYOLO(root_dir=dataset_root, split='train')
        print(f"✅ Loaded {len(self.train_dataset)} training images")
    
    def apply_geometric_transformations(self, img):
        """Apply geometric transformations"""
        results = {}
        
        # Resize
        results['resized'] = cv2.resize(img, (224, 224))
        
        # Rotation
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
        results['rotated'] = cv2.warpAffine(img, M, (w, h))
        
        # Affine transformation
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        results['affine'] = cv2.warpAffine(img, M, (w, h))
        
        return results
    
    def apply_intensity_transformations(self, img):
        """Apply intensity transformations"""
        results = {}
        
        # Histogram Equalization
        results['hist_eq'] = cv2.equalizeHist(img)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        results['clahe'] = clahe.apply(img)
        
        # Gamma Correction
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        results['gamma'] = cv2.LUT(img, table)
        
        # Intensity Normalization
        results['normalized'] = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        return results
    
    def apply_filtering(self, img):
        """Apply filtering operations"""
        results = {}
        
        # Gaussian Blur
        results['gaussian'] = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Median Filter
        results['median'] = cv2.medianBlur(img, 5)
        
        # Bilateral Filter (edge-preserving)
        results['bilateral'] = cv2.bilateralFilter(img, 9, 75, 75)
        
        return results
    
    def apply_edge_detection(self, img):
        """Apply edge detection methods"""
        results = {}
        
        # Sobel
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        results['sobel'] = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
        
        # Laplacian
        results['laplacian'] = cv2.Laplacian(img, cv2.CV_64F).astype(np.uint8)
        
        # Canny
        results['canny'] = cv2.Canny(img, 50, 150)
        
        return results
    
    def apply_morphological_operations(self, img):
        """Apply morphological operations"""
        results = {}
        
        # Binary threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Opening (erosion followed by dilation)
        kernel = np.ones((5, 5), np.uint8)
        results['opening'] = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Closing (dilation followed by erosion)
        results['closing'] = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return results
    
    def calculate_quality_metrics(self, original, processed):
        """Calculate PSNR and SSIM"""
        # Ensure same size
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        # Calculate PSNR
        psnr_value = psnr(original, processed, data_range=255)
        
        # Calculate SSIM
        ssim_value = ssim(original, processed, data_range=255)
        
        return psnr_value, ssim_value
    
    def characterize_dataset(self):
        """Characterize dataset with statistics"""
        print("\n" + "="*80)
        print("MODULE 1: DATASET CHARACTERIZATION")
        print("="*80)
        
        stats = {
            'intensity_mean': [],
            'intensity_std': [],
            'contrast': [],
            'noise_level': [],
            'edge_density': []
        }
        
        sample_size = min(100, len(self.train_dataset))
        print(f"Analyzing {sample_size} images...")
        
        for item in tqdm(self.train_dataset.data[:sample_size], desc="Processing"):
            img = cv2.imread(item['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = cv2.resize(img, (224, 224))
            
            # Intensity statistics
            stats['intensity_mean'].append(np.mean(img))
            stats['intensity_std'].append(np.std(img))
            stats['contrast'].append(np.std(img))
            
            # Noise level (Laplacian variance)
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            stats['noise_level'].append(np.var(laplacian))
            
            # Edge density
            edges = cv2.Canny(img, 50, 150)
            stats['edge_density'].append(np.sum(edges > 0) / (224 * 224))
        
        # Print statistics
        print("\n📊 Dataset Statistics:")
        print(f"  Mean Intensity: {np.mean(stats['intensity_mean']):.2f} ± {np.std(stats['intensity_mean']):.2f}")
        print(f"  Mean Contrast: {np.mean(stats['contrast']):.2f} ± {np.std(stats['contrast']):.2f}")
        print(f"  Noise Level: {np.mean(stats['noise_level']):.2f} ± {np.std(stats['noise_level']):.2f}")
        print(f"  Edge Density: {np.mean(stats['edge_density']):.4f} ± {np.std(stats['edge_density']):.4f}")
        
        return stats
    
    def compare_preprocessing_methods(self, num_samples=10):
        """Compare different preprocessing methods"""
        print("\n" + "="*80)
        print("MODULE 1: PREPROCESSING METHODS COMPARISON")
        print("="*80)
        
        # Select sample images
        sample_indices = np.random.choice(len(self.train_dataset), min(num_samples, len(self.train_dataset)), replace=False)
        
        all_results = []
        
        for idx in tqdm(sample_indices, desc="Processing images"):
            item = self.train_dataset.data[idx]
            img = cv2.imread(item['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = cv2.resize(img, (224, 224))
            
            # Apply all preprocessing methods
            methods = {}
            
            # Intensity transformations
            intensity_results = self.apply_intensity_transformations(img)
            methods.update(intensity_results)
            
            # Filtering
            filter_results = self.apply_filtering(img)
            methods.update(filter_results)
            
            # Edge detection
            edge_results = self.apply_edge_detection(img)
            methods.update(edge_results)
            
            # Calculate metrics for each method
            for method_name, processed_img in methods.items():
                psnr_val, ssim_val = self.calculate_quality_metrics(img, processed_img)
                
                all_results.append({
                    'image_id': idx,
                    'method': method_name,
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                    'label': item['label']
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Print summary
        print("\n📊 Preprocessing Methods Comparison:")
        print(results_df.groupby('method')[['psnr', 'ssim']].mean().sort_values('ssim', ascending=False))
        
        # Save results
        results_df.to_csv(os.path.join(self.results_dir, 'preprocessing_comparison.csv'), index=False)
        print(f"\n✅ Results saved to: {self.results_dir}/preprocessing_comparison.csv")
        
        return results_df
    
    def visualize_preprocessing(self, num_samples=5):
        """Visualize preprocessing results"""
        print("\n" + "="*80)
        print("MODULE 1: PREPROCESSING VISUALIZATION")
        print("="*80)
        
        sample_indices = np.random.choice(len(self.train_dataset), min(num_samples, len(self.train_dataset)), replace=False)
        
        fig, axes = plt.subplots(num_samples, 8, figsize=(20, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for row, idx in enumerate(sample_indices):
            item = self.train_dataset.data[idx]
            img = cv2.imread(item['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img = cv2.resize(img, (224, 224))
            
            # Original
            axes[row, 0].imshow(img, cmap='gray')
            axes[row, 0].set_title('Original')
            axes[row, 0].axis('off')
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(img)
            axes[row, 1].imshow(clahe_img, cmap='gray')
            axes[row, 1].set_title('CLAHE')
            axes[row, 1].axis('off')
            
            # Bilateral Filter
            bilateral = cv2.bilateralFilter(img, 9, 75, 75)
            axes[row, 2].imshow(bilateral, cmap='gray')
            axes[row, 2].set_title('Bilateral')
            axes[row, 2].axis('off')
            
            # Gaussian Blur
            gaussian = cv2.GaussianBlur(img, (5, 5), 0)
            axes[row, 3].imshow(gaussian, cmap='gray')
            axes[row, 3].set_title('Gaussian')
            axes[row, 3].axis('off')
            
            # Canny Edges
            canny = cv2.Canny(img, 50, 150)
            axes[row, 4].imshow(canny, cmap='gray')
            axes[row, 4].set_title('Canny')
            axes[row, 4].axis('off')
            
            # Sobel
            sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
            axes[row, 5].imshow(sobel, cmap='gray')
            axes[row, 5].set_title('Sobel')
            axes[row, 5].axis('off')
            
            # Histogram Equalization
            hist_eq = cv2.equalizeHist(img)
            axes[row, 6].imshow(hist_eq, cmap='gray')
            axes[row, 6].set_title('Hist Eq')
            axes[row, 6].axis('off')
            
            # Combined (Bilateral + CLAHE)
            combined = clahe.apply(bilateral)
            axes[row, 7].imshow(combined, cmap='gray')
            axes[row, 7].set_title('Combined')
            axes[row, 7].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.results_dir, 'preprocessing_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        plt.close()
    
    def run_module1(self):
        """Run complete Module 1 pipeline"""
        print("="*80)
        print("MODULE 1: FOUNDATIONS OF VISION AND IMAGE ANALYSIS")
        print("="*80)
        
        # 1. Dataset Characterization
        stats = self.characterize_dataset()
        
        # 2. Preprocessing Comparison
        results_df = self.compare_preprocessing_methods(num_samples=20)
        
        # 3. Visualization
        self.visualize_preprocessing(num_samples=5)
        
        print("\n" + "="*80)
        print("✅ MODULE 1 COMPLETE!")
        print("="*80)
        print(f"📁 Results saved in: {self.results_dir}/")
        print("   - preprocessing_comparison.csv")
        print("   - preprocessing_comparison.png")
        print("="*80)


def main():
    """Main execution"""
    dataset_root = r'data\archive\bone fracture detection.v4-v4.yolov8'
    
    # Normalize path
    dataset_root = str(Path(dataset_root).resolve())
    
    if not Path(dataset_root).exists():
        print(f"❌ Dataset not found at: {dataset_root}")
        print("Please check the dataset path")
        return
    
    # Run Module 1
    module1 = Module1Preprocessing(dataset_root)
    module1.run_module1()


if __name__ == '__main__':
    main()

