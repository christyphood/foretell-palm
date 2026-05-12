"""
Feature extraction from palm line segmentation masks.
Extracts geometric features like line length, curvature, intersections, etc.
"""

import cv2
import numpy as np
from scipy import ndimage
from sklearn.metrics import pairwise_distances


def extract_skeleton(mask):
    """
    Extract the skeleton of a binary mask using morphological thinning.
    
    Args:
        mask: Binary mask (0 or 255)
    
    Returns:
        skeleton: Thinned binary mask
    """
    mask_binary = (mask > 0).astype(np.uint8)
    skeleton = cv2.ximgproc.thinning(mask_binary * 255) if hasattr(cv2, 'ximgproc') else mask_binary
    return skeleton


def get_line_length(mask):
    """
    Calculate the length of a line in the mask by counting skeleton pixels.
    
    Args:
        mask: Binary mask of the line
    
    Returns:
        length: Approximate length in pixels
    """
    if mask.max() == 0:
        return 0
    
    # Get skeleton
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # Simple approach: count non-zero pixels as approximation
    # For better accuracy, we could trace the contour
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return 0
    
    # Get the longest contour
    longest_contour = max(contours, key=cv2.contourArea)
    length = cv2.arcLength(longest_contour, closed=False)
    
    return length


def get_curvature(mask):
    """
    Calculate the curvature/tortuosity of a line.
    Tortuosity = actual_length / straight_line_distance
    
    Args:
        mask: Binary mask of the line
    
    Returns:
        curvature: Ratio of arc length to endpoint distance
    """
    if mask.max() == 0:
        return 0
    
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return 0
    
    longest_contour = max(contours, key=cv2.contourArea)
    arc_length = cv2.arcLength(longest_contour, closed=False)
    
    # Get endpoints
    contour_points = longest_contour.reshape(-1, 2)
    if len(contour_points) < 2:
        return 0
    
    start, end = contour_points[0], contour_points[-1]
    straight_distance = np.linalg.norm(start - end)
    
    if straight_distance < 1:
        return 1.0
    
    tortuosity = arc_length / straight_distance
    return tortuosity


def get_line_angle(mask):
    """
    Get the angle of the line with respect to horizontal axis.
    
    Args:
        mask: Binary mask of the line
    
    Returns:
        angle: Angle in degrees
    """
    if mask.max() == 0:
        return 0
    
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return 0
    
    longest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a line
    [vx, vy, x, y] = cv2.fitLine(longest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx) * 180 / np.pi
    
    return float(angle[0])


def count_intersections(mask1, mask2):
    """
    Count intersection points between two line masks.
    
    Args:
        mask1, mask2: Binary masks of lines
    
    Returns:
        count: Number of intersection regions
    """
    intersection = cv2.bitwise_and(mask1, mask2)
    
    if intersection.max() == 0:
        return 0
    
    # Count connected components in intersection
    num_labels, _ = cv2.connectedComponents(intersection)
    
    # Subtract 1 for background
    return max(0, num_labels - 1)


def extract_palm_features(segmentation_mask):
    """
    Extract all features from a palm segmentation mask.
    
    Args:
        segmentation_mask: Mask with values 0 (bg), 1 (life), 2 (head), 3 (heart)
    
    Returns:
        features: Dictionary of extracted features
    """
    # Separate masks for each line
    life_mask = (segmentation_mask == 1).astype(np.uint8) * 255
    head_mask = (segmentation_mask == 2).astype(np.uint8) * 255
    heart_mask = (segmentation_mask == 3).astype(np.uint8) * 255
    
    features = {}
    
    # Line lengths
    features['life_length'] = get_line_length(life_mask)
    features['head_length'] = get_line_length(head_mask)
    features['heart_length'] = get_line_length(heart_mask)
    
    # Curvatures
    features['life_curvature'] = get_curvature(life_mask)
    features['head_curvature'] = get_curvature(head_mask)
    features['heart_curvature'] = get_curvature(heart_mask)
    
    # Angles
    features['life_angle'] = get_line_angle(life_mask)
    features['head_angle'] = get_line_angle(head_mask)
    features['heart_angle'] = get_line_angle(heart_mask)
    
    # Intersections
    features['life_head_intersection'] = count_intersections(life_mask, head_mask)
    features['life_heart_intersection'] = count_intersections(life_mask, heart_mask)
    features['head_heart_intersection'] = count_intersections(head_mask, heart_mask)
    
    # Total line coverage
    total_pixels = segmentation_mask.shape[0] * segmentation_mask.shape[1]
    line_pixels = np.sum(segmentation_mask > 0)
    features['line_coverage'] = line_pixels / total_pixels if total_pixels > 0 else 0
    
    return features


def classify_palm(features):
    """
    Classify palm based on extracted features.
    
    Args:
        features: Dictionary of features
    
    Returns:
        classification: Dictionary with labels and confidence
    """
    classification = {}
    
    # Determine dominant line based on length
    lengths = {
        'Life': features.get('life_length', 0),
        'Head': features.get('head_length', 0),
        'Heart': features.get('heart_length', 0)
    }
    
    total_length = sum(lengths.values())
    
    if total_length == 0:
        classification['dominant_line'] = 'Unknown'
        classification['confidence'] = 0.0
    else:
        dominant_line = max(lengths, key=lengths.get)
        confidence = lengths[dominant_line] / total_length
        
        classification['dominant_line'] = dominant_line
        classification['confidence'] = round(confidence, 3)
    
    # Palm type based on curvature
    avg_curvature = (
        features.get('life_curvature', 0) + 
        features.get('head_curvature', 0) + 
        features.get('heart_curvature', 0)
    ) / 3
    
    if avg_curvature > 1.3:
        classification['palm_type'] = 'Curved/Expressive'
    elif avg_curvature > 1.1:
        classification['palm_type'] = 'Balanced'
    else:
        classification['palm_type'] = 'Straight/Practical'
    
    # Simple synthetic prediction (career shift)
    # Based on head line angle and intersection count
    head_angle = abs(features.get('head_angle', 0))
    intersections = features.get('life_head_intersection', 0)
    
    if head_angle > 10 and intersections > 0:
        classification['career_shift_indicator'] = 'Yes'
        classification['career_shift_confidence'] = 0.7
    else:
        classification['career_shift_indicator'] = 'No'
        classification['career_shift_confidence'] = 0.6
    
    return classification
