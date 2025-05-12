import cv2
import numpy as np
import os

def draw_predictions(frame, classifications, thickness=2):
    """
    Draw bounding boxes and labels on the frame.
    
    Args:
        frame: Input frame
        classifications: List of (detection, vehicle_type) tuples
        thickness: Line thickness
        
    Returns:
        Frame with bounding boxes and labels
    """
    result_frame = frame.copy()
    
    # Define colors for different vehicle types (BGR format)
    colors = {
        'car': (0, 255, 0),       # Green
        'motorcycle': (0, 165, 255),  # Orange
        'bus': (255, 0, 0),       # Blue
        'truck': (0, 0, 255),     # Red
        'unknown': (128, 128, 128)  # Gray
    }
    
    for detection, vehicle_type in classifications:
        x1, y1, x2, y2, conf, cls = detection
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get color for vehicle type
        color = colors.get(vehicle_type, colors['unknown'])
        
        # Draw bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text with confidence
        label = f"{vehicle_type} {conf:.2f}"
        
        # Calculate text size and position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
        
        # Draw filled rectangle for text background
        cv2.rectangle(result_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness - 1)
    
    return result_frame

def create_output_directory(output_dir='output'):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def resize_frame(frame, width=None, height=None):
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        width: Target width (if None, will be calculated from height)
        height: Target height (if None, will be calculated from width)
        
    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if width is None:
        aspect_ratio = width / float(w)
        dim = (int(w * aspect_ratio), height)
    else:
        aspect_ratio = width / float(w)
        dim = (width, int(h * aspect_ratio))
    
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def count_vehicles(classifications):
    """
    Count vehicles by type.
    
    Args:
        classifications: List of (detection, vehicle_type) tuples
        
    Returns:
        Dictionary with counts for each vehicle type
    """
    counts = {
        'car': 0,
        'motorcycle': 0,
        'bus': 0,
        'truck': 0,
        'unknown': 0
    }
    
    for _, vehicle_type in classifications:
        if vehicle_type in counts:
            counts[vehicle_type] += 1
        else:
            counts['unknown'] += 1
    
    return counts
