import argparse
import cv2
import time
import os
from vehicle_detector import VehicleDetector
from vehicle_classifier import VehicleClassifier
from utils import draw_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Vehicle Classification from Video')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--show', action='store_true', help='Display the output in real-time')
    return parser.parse_args()

def process_video(input_path, output_path, conf_threshold, show_video):
    # Initialize detector and classifier
    detector = VehicleDetector()
    classifier = VehicleClassifier()
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles in the frame
        detections = detector.detect(frame, conf_threshold)
        
        # Classify detected vehicles
        classifications = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
            if vehicle_img.size > 0:  # Make sure the crop is valid
                vehicle_type = classifier.classify(vehicle_img)
                classifications.append((det, vehicle_type))
        
        # Draw predictions on the frame
        result_frame = draw_predictions(frame, classifications)
        
        # Write frame to output video
        out.write(result_frame)
        
        # Display the frame if requested
        if show_video:
            cv2.imshow('Vehicle Classification', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time
            print(f"Processed {frame_count} frames. FPS: {fps_processing:.2f}")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Output saved to {output_path}")

def main():
    args = parse_args()
    process_video(args.input, args.output, args.conf_threshold, args.show)

if __name__ == "__main__":
    main()
