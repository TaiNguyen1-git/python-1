import cv2
import argparse
from vehicle_detector import VehicleDetector
from utils import draw_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Test Vehicle Detector')
    parser.add_argument('--image', type=str, required=True, help='Path to input image file')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to output image file')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for detection')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image file {args.image}")
        return
    
    # Initialize detector
    detector = VehicleDetector()
    
    # Detect vehicles
    detections = detector.detect(image, args.conf_threshold)
    
    # Create dummy classifications (without actual classification)
    classifications = [(det, "vehicle") for det in detections]
    
    # Draw predictions
    result_image = draw_predictions(image, classifications)
    
    # Save output image
    cv2.imwrite(args.output, result_image)
    print(f"Output saved to {args.output}")
    
    # Display image
    cv2.imshow('Vehicle Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
