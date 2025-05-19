import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil
from vehicle_detector import VehicleDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Dataset from Video')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output-dir', type=str, default='dataset', help='Output directory for dataset')
    parser.add_argument('--frame-interval', type=int, default=30, help='Extract one frame every N frames')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to extract')
    return parser.parse_args()

def extract_frames(video_path, output_dir, frame_interval, max_frames=None):
    """
    Extract frames from a video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        frame_interval: Extract one frame every N frames
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of paths to extracted frames
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    # Calculate number of frames to extract
    if max_frames:
        num_frames = min(total_frames // frame_interval, max_frames)
    else:
        num_frames = total_frames // frame_interval
    
    print(f"Extracting {num_frames} frames...")
    
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    # Extract frames
    with tqdm(total=num_frames) as pbar:
        while cap.isOpened() and (max_frames is None or extracted_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                extracted_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    # Release resources
    cap.release()
    
    print(f"Extracted {len(frame_paths)} frames")
    return frame_paths

def create_dataset_structure(output_dir):
    """
    Create dataset directory structure.
    
    Args:
        output_dir: Output directory for dataset
    """
    # Create main directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train/val/test directories
    for split in ['train', 'val']:
        # For detector (YOLO format)
        os.makedirs(os.path.join(output_dir, 'detector', split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'detector', split, 'labels'), exist_ok=True)
        
        # For classifier
        os.makedirs(os.path.join(output_dir, 'classifier', split), exist_ok=True)
        for cls in ['car', 'motorcycle', 'bus', 'truck']:
            os.makedirs(os.path.join(output_dir, 'classifier', split, cls), exist_ok=True)

def prepare_detector_dataset(frame_paths, output_dir, conf_threshold, split_ratio):
    """
    Prepare dataset for detector training.
    
    Args:
        frame_paths: List of paths to frames
        output_dir: Output directory for dataset
        conf_threshold: Confidence threshold for detection
        split_ratio: Train/val split ratio
    """
    # Initialize detector
    detector = VehicleDetector()
    
    # Split frames into train/val
    random.shuffle(frame_paths)
    split_idx = int(len(frame_paths) * split_ratio)
    train_frames = frame_paths[:split_idx]
    val_frames = frame_paths[split_idx:]
    
    print(f"Preparing detector dataset...")
    print(f"Train frames: {len(train_frames)}")
    print(f"Val frames: {len(val_frames)}")
    
    # Process frames
    for split, frames in [('train', train_frames), ('val', val_frames)]:
        print(f"Processing {split} frames...")
        
        for frame_path in tqdm(frames):
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Get frame name
            frame_name = os.path.basename(frame_path)
            
            # Copy frame to dataset
            dst_img_path = os.path.join(output_dir, 'detector', split, 'images', frame_name)
            shutil.copy(frame_path, dst_img_path)
            
            # Detect vehicles
            detections = detector.detect(frame, conf_threshold)
            
            # Create YOLO format labels
            h, w = frame.shape[:2]
            label_path = os.path.join(output_dir, 'detector', split, 'labels', 
                                     os.path.splitext(frame_name)[0] + '.txt')
            
            with open(label_path, 'w') as f:
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    
                    # Convert to YOLO format (class, x_center, y_center, width, height)
                    # All values normalized to [0, 1]
                    x_center = (x1 + x2) / (2 * w)
                    y_center = (y1 + y2) / (2 * h)
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Map COCO class to our class
                    # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
                    # Our: 0=car, 1=motorcycle, 2=bus, 3=truck
                    if cls == 2:  # car
                        yolo_cls = 0
                    elif cls == 3:  # motorcycle
                        yolo_cls = 1
                    elif cls == 5:  # bus
                        yolo_cls = 2
                    elif cls == 7:  # truck
                        yolo_cls = 3
                    else:
                        continue
                    
                    f.write(f"{yolo_cls} {x_center} {y_center} {width} {height}\n")

def prepare_classifier_dataset(frame_paths, output_dir, conf_threshold, split_ratio):
    """
    Prepare dataset for classifier training.
    
    Args:
        frame_paths: List of paths to frames
        output_dir: Output directory for dataset
        conf_threshold: Confidence threshold for detection
        split_ratio: Train/val split ratio
    """
    # Initialize detector
    detector = VehicleDetector()
    
    # Split frames into train/val
    random.shuffle(frame_paths)
    split_idx = int(len(frame_paths) * split_ratio)
    train_frames = frame_paths[:split_idx]
    val_frames = frame_paths[split_idx:]
    
    print(f"Preparing classifier dataset...")
    print(f"Train frames: {len(train_frames)}")
    print(f"Val frames: {len(val_frames)}")
    
    # Process frames
    for split, frames in [('train', train_frames), ('val', val_frames)]:
        print(f"Processing {split} frames...")
        
        for frame_idx, frame_path in enumerate(tqdm(frames)):
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Detect vehicles
            detections = detector.detect(frame, conf_threshold)
            
            # Extract vehicle crops
            for det_idx, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls = det
                
                # Crop vehicle
                vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
                if vehicle_img.size == 0:
                    continue
                
                # Map COCO class to our class
                if cls == 2:  # car
                    class_name = 'car'
                elif cls == 3:  # motorcycle
                    class_name = 'motorcycle'
                elif cls == 5:  # bus
                    class_name = 'bus'
                elif cls == 7:  # truck
                    class_name = 'truck'
                else:
                    continue
                
                # Save crop
                crop_name = f"frame_{frame_idx:06d}_det_{det_idx:03d}.jpg"
                crop_path = os.path.join(output_dir, 'classifier', split, class_name, crop_name)
                cv2.imwrite(crop_path, vehicle_img)

def main():
    args = parse_args()
    
    # Create dataset structure
    create_dataset_structure(args.output_dir)
    
    # Extract frames from video
    frames_dir = os.path.join(args.output_dir, 'frames')
    frame_paths = extract_frames(
        args.input, 
        frames_dir, 
        args.frame_interval,
        args.max_frames
    )
    
    if not frame_paths:
        print("No frames extracted. Exiting.")
        return
    
    # Prepare detector dataset
    prepare_detector_dataset(
        frame_paths, 
        args.output_dir, 
        args.conf_threshold, 
        args.split_ratio
    )
    
    # Prepare classifier dataset
    prepare_classifier_dataset(
        frame_paths, 
        args.output_dir, 
        args.conf_threshold, 
        args.split_ratio
    )
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
