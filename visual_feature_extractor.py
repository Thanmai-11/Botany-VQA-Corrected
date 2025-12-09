"""
Visual feature extraction utilities for validating VQA answers.
Extracts color, shape, and other visual features from flower images.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import cv2


class VisualFeatureExtractor:
    """Extract visual features from flower images for validation."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.color_names = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'orange': ([11, 100, 100], [25, 255, 255]),
            'yellow': ([26, 100, 100], [35, 255, 255]),
            'green': ([36, 100, 100], [85, 255, 255]),
            'blue': ([86, 100, 100], [125, 255, 255]),
            'purple': ([126, 100, 100], [155, 255, 255]),
            'pink': ([156, 100, 100], [170, 255, 255]),
            'white': ([0, 0, 200], [180, 30, 255]),
        }
    
    def extract_dominant_colors(self, image_path: str, n_colors: int = 3) -> List[str]:
        """
        Extract dominant colors from an image using K-means clustering.
        
        Args:
            image_path: Path to the image
            n_colors: Number of dominant colors to extract
            
        Returns:
            List of color names
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Reshape to 2D array of pixels
        pixels = image_array.reshape(-1, 3)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors_rgb = kmeans.cluster_centers_.astype(int)
        
        # Convert to color names
        color_names = []
        for rgb in colors_rgb:
            color_name = self._rgb_to_color_name(rgb)
            color_names.append(color_name)
        
        return color_names
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB to nearest color name."""
        # Convert RGB to HSV for better color matching
        rgb_normalized = rgb.reshape(1, 1, 3).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)[0][0]
        
        # Find closest color name
        for color_name, (lower, upper) in self.color_names.items():
            if (lower[0] <= hsv[0] <= upper[0] and 
                lower[1] <= hsv[1] <= upper[1] and 
                lower[2] <= hsv[2] <= upper[2]):
                return color_name
        
        # Default to describing brightness
        if hsv[2] < 50:
            return "dark"
        elif hsv[2] > 200 and hsv[1] < 30:
            return "white"
        else:
            return "mixed"
    
    def estimate_petal_count(self, image_path: str) -> int:
        """
        Estimate petal count using contour detection.
        Note: This is a rough estimate and may not be accurate for all flowers.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Estimated petal count (or -1 if cannot determine)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area (assume petals are medium-sized)
            petal_contours = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
            
            return len(petal_contours) if len(petal_contours) > 0 else -1
        except Exception as e:
            print(f"Error estimating petal count: {e}")
            return -1
    
    def extract_all_features(self, image_path: str) -> Dict[str, any]:
        """
        Extract all visual features from an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'dominant_colors': self.extract_dominant_colors(image_path),
            'petal_count_estimate': self.estimate_petal_count(image_path),
        }
        
        return features
    
    def validate_color_answer(self, image_path: str, answer: str) -> bool:
        """
        Validate if a color answer matches the image.
        
        Args:
            image_path: Path to the image
            answer: Predicted color answer
            
        Returns:
            True if answer is plausible, False otherwise
        """
        dominant_colors = self.extract_dominant_colors(image_path, n_colors=5)
        answer_lower = answer.lower()
        
        # Check if any dominant color is mentioned in the answer
        for color in dominant_colors:
            if color in answer_lower:
                return True
        
        return False
    
    def validate_count_answer(self, image_path: str, answer: str) -> bool:
        """
        Validate if a counting answer is plausible.
        
        Args:
            image_path: Path to the image
            answer: Predicted count answer
            
        Returns:
            True if answer is plausible, False otherwise
        """
        estimated_count = self.estimate_petal_count(image_path)
        
        if estimated_count == -1:
            # Cannot validate, assume correct
            return True
        
        try:
            # Extract number from answer
            answer_num = int(''.join(filter(str.isdigit, answer)))
            
            # Allow ±2 margin of error
            return abs(answer_num - estimated_count) <= 2
        except ValueError:
            # Cannot parse number, assume correct
            return True


def get_image_statistics(image_path: str) -> Dict[str, any]:
    """
    Get basic image statistics.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Dictionary of image statistics
    """
    image = Image.open(image_path)
    
    return {
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format,
    }
