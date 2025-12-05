"""
Advanced Feature Extraction Module
Implements SIFT, SURF, HOG, BoVW, and Geometric/Temporal Extensions
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import os


class AdvancedFeatureExtractor:
    """Advanced feature extraction including SIFT, SURF, HOG, BoVW"""
    
    def __init__(self, vocab_size=100):
        """
        Initialize advanced feature extractor
        
        Args:
            vocab_size: Size of visual vocabulary for BoVW
        """
        self.vocab_size = vocab_size
        self.sift = None
        self.surf = None
        self.hog_features_shape = None
        self.vocabulary = None  # For BoVW
        self.scaler = StandardScaler()
        
        # Initialize SIFT (always available in OpenCV)
        try:
            self.sift = cv2.SIFT_create(nfeatures=100)
        except:
            try:
                self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=100)
            except:
                print("Warning: SIFT not available")
        
        # Initialize SURF (may not be available in all OpenCV versions)
        try:
            self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        except:
            try:
                self.surf = cv2.SURF_create(hessianThreshold=400)
            except:
                print("Warning: SURF not available (requires opencv-contrib-python)")
    
    def extract_sift_features(self, image):
        """
        Extract SIFT keypoints and descriptors
        
        Args:
            image: Grayscale image (numpy array)
            
        Returns:
            Dictionary with SIFT features
        """
        features = {}
        
        if self.sift is None:
            # Return zeros if SIFT not available
            features['sift_keypoints_count'] = 0
            features['sift_descriptor_mean'] = 0
            features['sift_descriptor_std'] = 0
            features['sift_descriptor_max'] = 0
            features['sift_descriptor_min'] = 0
            return features
        
        try:
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            
            features['sift_keypoints_count'] = len(keypoints) if keypoints else 0
            
            if descriptors is not None and len(descriptors) > 0:
                # Statistical features from descriptors
                features['sift_descriptor_mean'] = np.mean(descriptors)
                features['sift_descriptor_std'] = np.std(descriptors)
                features['sift_descriptor_max'] = np.max(descriptors)
                features['sift_descriptor_min'] = np.min(descriptors)
                
                # Keypoint response statistics
                if keypoints:
                    responses = [kp.response for kp in keypoints]
                    features['sift_response_mean'] = np.mean(responses)
                    features['sift_response_std'] = np.std(responses)
                    features['sift_response_max'] = np.max(responses)
                    
                    # Spatial distribution
                    x_coords = [kp.pt[0] for kp in keypoints]
                    y_coords = [kp.pt[1] for kp in keypoints]
                    features['sift_x_mean'] = np.mean(x_coords)
                    features['sift_y_mean'] = np.mean(y_coords)
                    features['sift_x_std'] = np.std(x_coords)
                    features['sift_y_std'] = np.std(y_coords)
                else:
                    features['sift_response_mean'] = 0
                    features['sift_response_std'] = 0
                    features['sift_response_max'] = 0
                    features['sift_x_mean'] = 0
                    features['sift_y_mean'] = 0
                    features['sift_x_std'] = 0
                    features['sift_y_std'] = 0
            else:
                features['sift_descriptor_mean'] = 0
                features['sift_descriptor_std'] = 0
                features['sift_descriptor_max'] = 0
                features['sift_descriptor_min'] = 0
                features['sift_response_mean'] = 0
                features['sift_response_std'] = 0
                features['sift_response_max'] = 0
                features['sift_x_mean'] = 0
                features['sift_y_mean'] = 0
                features['sift_x_std'] = 0
                features['sift_y_std'] = 0
                
        except Exception as e:
            print(f"Error extracting SIFT features: {e}")
            features['sift_keypoints_count'] = 0
            features['sift_descriptor_mean'] = 0
            features['sift_descriptor_std'] = 0
            features['sift_descriptor_max'] = 0
            features['sift_descriptor_min'] = 0
        
        return features
    
    def extract_surf_features(self, image):
        """
        Extract SURF keypoints and descriptors
        
        Args:
            image: Grayscale image (numpy array)
            
        Returns:
            Dictionary with SURF features
        """
        features = {}
        
        if self.surf is None:
            # Return zeros if SURF not available
            features['surf_keypoints_count'] = 0
            features['surf_descriptor_mean'] = 0
            features['surf_descriptor_std'] = 0
            return features
        
        try:
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.surf.detectAndCompute(image, None)
            
            features['surf_keypoints_count'] = len(keypoints) if keypoints else 0
            
            if descriptors is not None and len(descriptors) > 0:
                features['surf_descriptor_mean'] = np.mean(descriptors)
                features['surf_descriptor_std'] = np.std(descriptors)
                features['surf_descriptor_max'] = np.max(descriptors)
                features['surf_descriptor_min'] = np.min(descriptors)
                
                if keypoints:
                    responses = [kp.response for kp in keypoints]
                    features['surf_response_mean'] = np.mean(responses)
                    features['surf_response_std'] = np.std(responses)
                    features['surf_response_max'] = np.max(responses)
                else:
                    features['surf_response_mean'] = 0
                    features['surf_response_std'] = 0
                    features['surf_response_max'] = 0
            else:
                features['surf_descriptor_mean'] = 0
                features['surf_descriptor_std'] = 0
                features['surf_descriptor_max'] = 0
                features['surf_descriptor_min'] = 0
                features['surf_response_mean'] = 0
                features['surf_response_std'] = 0
                features['surf_response_max'] = 0
                
        except Exception as e:
            print(f"Error extracting SURF features: {e}")
            features['surf_keypoints_count'] = 0
            features['surf_descriptor_mean'] = 0
            features['surf_descriptor_std'] = 0
        
        return features
    
    def extract_hog_features(self, image, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Extract HOG (Histogram of Oriented Gradients) features
        
        Args:
            image: Grayscale image (numpy array)
            pixels_per_cell: Size of cells for HOG
            cells_per_block: Number of cells per block
            
        Returns:
            Dictionary with HOG features
        """
        features = {}
        
        try:
            # Compute HOG features
            hog_features, hog_image = hog(
                image,
                orientations=9,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=True,
                feature_vector=True
            )
            
            # Store shape for consistency
            if self.hog_features_shape is None:
                self.hog_features_shape = hog_features.shape[0]
            
            # Statistical features from HOG
            features['hog_mean'] = np.mean(hog_features)
            features['hog_std'] = np.std(hog_features)
            features['hog_max'] = np.max(hog_features)
            features['hog_min'] = np.min(hog_features)
            features['hog_median'] = np.median(hog_features)
            
            # Histogram of HOG values
            hog_hist, _ = np.histogram(hog_features, bins=10)
            hog_hist = hog_hist.astype(float) / (hog_hist.sum() + 1e-7)
            features['hog_entropy'] = -np.sum(hog_hist * np.log2(hog_hist + 1e-7))
            
            # Store full HOG vector length
            features['hog_vector_length'] = len(hog_features)
            
        except Exception as e:
            print(f"Error extracting HOG features: {e}")
            features['hog_mean'] = 0
            features['hog_std'] = 0
            features['hog_max'] = 0
            features['hog_min'] = 0
            features['hog_median'] = 0
            features['hog_entropy'] = 0
            features['hog_vector_length'] = 0
        
        return features
    
    def build_vocabulary(self, descriptor_list, vocab_size=None):
        """
        Build visual vocabulary for Bag of Visual Words
        
        Args:
            descriptor_list: List of descriptors from multiple images
            vocab_size: Size of vocabulary (uses self.vocab_size if None)
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        try:
            # Stack all descriptors
            all_descriptors = np.vstack(descriptor_list)
            
            # Remove any NaN or Inf values
            all_descriptors = all_descriptors[~np.isnan(all_descriptors).any(axis=1)]
            all_descriptors = all_descriptors[~np.isinf(all_descriptors).any(axis=1)]
            
            if len(all_descriptors) < vocab_size:
                print(f"Warning: Only {len(all_descriptors)} descriptors available, reducing vocab size")
                vocab_size = max(10, len(all_descriptors) // 2)
            
            # K-means clustering to create vocabulary
            kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init=10)
            kmeans.fit(all_descriptors)
            
            self.vocabulary = kmeans.cluster_centers_
            print(f"Built vocabulary with {vocab_size} visual words")
            
        except Exception as e:
            print(f"Error building vocabulary: {e}")
            self.vocabulary = None
    
    def extract_bovw_features(self, image, use_sift=True):
        """
        Extract Bag of Visual Words (BoVW) features
        
        Args:
            image: Grayscale image (numpy array)
            use_sift: If True, use SIFT descriptors; else use SURF
            
        Returns:
            Dictionary with BoVW histogram features
        """
        features = {}
        
        if self.vocabulary is None:
            # Return zeros if vocabulary not built
            for i in range(self.vocab_size):
                features[f'bovw_bin_{i}'] = 0
            features['bovw_entropy'] = 0
            return features
        
        try:
            # Extract descriptors
            if use_sift and self.sift is not None:
                _, descriptors = self.sift.detectAndCompute(image, None)
            elif not use_sift and self.surf is not None:
                _, descriptors = self.surf.detectAndCompute(image, None)
            else:
                # No descriptors available
                for i in range(self.vocab_size):
                    features[f'bovw_bin_{i}'] = 0
                features['bovw_entropy'] = 0
                return features
            
            if descriptors is None or len(descriptors) == 0:
                for i in range(self.vocab_size):
                    features[f'bovw_bin_{i}'] = 0
                features['bovw_entropy'] = 0
                return features
            
            # Assign descriptors to nearest vocabulary words
            from scipy.spatial.distance import cdist
            distances = cdist(descriptors, self.vocabulary, metric='euclidean')
            assignments = np.argmin(distances, axis=1)
            
            # Create histogram
            histogram = np.bincount(assignments, minlength=self.vocab_size)
            histogram = histogram.astype(float) / (histogram.sum() + 1e-7)  # Normalize
            
            # Store histogram bins
            for i in range(self.vocab_size):
                features[f'bovw_bin_{i}'] = histogram[i]
            
            # Entropy of histogram
            features['bovw_entropy'] = -np.sum(histogram * np.log2(histogram + 1e-7))
            
        except Exception as e:
            print(f"Error extracting BoVW features: {e}")
            for i in range(self.vocab_size):
                features[f'bovw_bin_{i}'] = 0
            features['bovw_entropy'] = 0
        
        return features
    
    def extract_geometric_features(self, image1, image2=None):
        """
        Extract geometric features including image registration
        
        Args:
            image1: First image (reference)
            image2: Second image (for registration) - optional
            
        Returns:
            Dictionary with geometric features
        """
        features = {}
        
        try:
            # Geometric moments
            moments = cv2.moments(image1)
            if moments['m00'] != 0:
                # Centroid
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                features['centroid_x'] = cx
                features['centroid_y'] = cy
                
                # Central moments (translation invariant)
                features['mu20'] = moments['mu20'] / (moments['m00'] ** 2)
                features['mu02'] = moments['mu02'] / (moments['m00'] ** 2)
                features['mu11'] = moments['mu11'] / (moments['m00'] ** 2)
                
                # Orientation
                features['orientation'] = 0.5 * np.arctan2(2 * moments['mu11'], 
                                                          moments['mu20'] - moments['mu02'])
            else:
                features['centroid_x'] = 0
                features['centroid_y'] = 0
                features['mu20'] = 0
                features['mu02'] = 0
                features['mu11'] = 0
                features['orientation'] = 0
            
            # Image registration (if second image provided)
            if image2 is not None:
                try:
                    # Use ORB for feature matching
                    orb = cv2.ORB_create()
                    kp1, des1 = orb.detectAndCompute(image1, None)
                    kp2, des2 = orb.detectAndCompute(image2, None)
                    
                    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                        # Match features
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)
                        
                        features['registration_matches'] = len(matches)
                        if len(matches) > 0:
                            features['registration_match_ratio'] = len(matches) / min(len(kp1), len(kp2))
                            features['registration_avg_distance'] = np.mean([m.distance for m in matches])
                        else:
                            features['registration_match_ratio'] = 0
                            features['registration_avg_distance'] = 0
                    else:
                        features['registration_matches'] = 0
                        features['registration_match_ratio'] = 0
                        features['registration_avg_distance'] = 0
                        
                except Exception as e:
                    features['registration_matches'] = 0
                    features['registration_match_ratio'] = 0
                    features['registration_avg_distance'] = 0
            else:
                features['registration_matches'] = 0
                features['registration_match_ratio'] = 0
                features['registration_avg_distance'] = 0
            
        except Exception as e:
            print(f"Error extracting geometric features: {e}")
            features['centroid_x'] = 0
            features['centroid_y'] = 0
            features['mu20'] = 0
            features['mu02'] = 0
            features['mu11'] = 0
            features['orientation'] = 0
        
        return features
    
    def extract_temporal_features(self, image_sequence):
        """
        Extract temporal features from image sequence
        
        Args:
            image_sequence: List of images (temporal sequence)
            
        Returns:
            Dictionary with temporal features
        """
        features = {}
        
        if len(image_sequence) < 2:
            features['temporal_variance'] = 0
            features['temporal_mean_diff'] = 0
            features['temporal_flow_magnitude'] = 0
            return features
        
        try:
            # Convert to grayscale if needed
            seq_gray = []
            for img in image_sequence:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                seq_gray.append(img.astype(np.float32))
            
            # Temporal variance (change over time)
            seq_array = np.array(seq_gray)
            temporal_variance = np.var(seq_array, axis=0)
            features['temporal_variance'] = np.mean(temporal_variance)
            features['temporal_variance_std'] = np.std(temporal_variance)
            
            # Mean difference between consecutive frames
            diffs = []
            for i in range(len(seq_gray) - 1):
                diff = np.abs(seq_gray[i+1] - seq_gray[i])
                diffs.append(np.mean(diff))
            features['temporal_mean_diff'] = np.mean(diffs) if diffs else 0
            features['temporal_mean_diff_std'] = np.std(diffs) if diffs else 0
            
            # Optical flow (if sequence available)
            if len(seq_gray) >= 2:
                try:
                    # Calculate optical flow between first two frames
                    flow = cv2.calcOpticalFlowFarneback(
                        seq_gray[0], seq_gray[1], None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    features['temporal_flow_magnitude'] = np.mean(magnitude)
                    features['temporal_flow_std'] = np.std(magnitude)
                except:
                    features['temporal_flow_magnitude'] = 0
                    features['temporal_flow_std'] = 0
            else:
                features['temporal_flow_magnitude'] = 0
                features['temporal_flow_std'] = 0
                
        except Exception as e:
            print(f"Error extracting temporal features: {e}")
            features['temporal_variance'] = 0
            features['temporal_mean_diff'] = 0
            features['temporal_flow_magnitude'] = 0
        
        return features
    
    def extract_all_advanced_features(self, image, image2=None, image_sequence=None):
        """
        Extract all advanced features
        
        Args:
            image: Primary image (grayscale)
            image2: Second image for registration (optional)
            image_sequence: List of images for temporal analysis (optional)
            
        Returns:
            Dictionary with all advanced features
        """
        all_features = {}
        
        # SIFT features
        sift_features = self.extract_sift_features(image)
        all_features.update(sift_features)
        
        # SURF features
        surf_features = self.extract_surf_features(image)
        all_features.update(surf_features)
        
        # HOG features
        hog_features = self.extract_hog_features(image)
        all_features.update(hog_features)
        
        # BoVW features (if vocabulary is built)
        if self.vocabulary is not None:
            bovw_features = self.extract_bovw_features(image)
            all_features.update(bovw_features)
        
        # Geometric features
        geometric_features = self.extract_geometric_features(image, image2)
        all_features.update(geometric_features)
        
        # Temporal features (if sequence provided)
        if image_sequence is not None:
            temporal_features = self.extract_temporal_features(image_sequence)
            all_features.update(temporal_features)
        
        return all_features

