import os
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image

class VehicleDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # Use the best model from training
            model_path = 'runs/train/yolov8n_vehicles3/weights/best.pt'

        # Load the model
        self.model = YOLO(model_path)

        # Class names
        self.class_names = ['car', 'motorcycle', 'bus', 'truck']

    def detect(self, frame, conf_threshold=0.5):
        # Run inference
        results = self.model(frame, conf=conf_threshold)

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                detections.append([x1, y1, x2, y2, conf, cls])

        return detections

class VehicleClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            # Use the best model from training
            model_path = 'models/vehicle_classifier_best.pth'

        # Initialize model
        self.model = models.resnet50(pretrained=False)
        num_classes = 4  # car, motorcycle, bus, truck
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Load model weights if file exists
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
        else:
            print(f"Warning: Model file {model_path} not found. Using heuristic classification.")

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Class names
        self.class_names = ['car', 'motorcycle', 'bus', 'truck']

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify(self, img):
        # If model file doesn't exist, use heuristic classification
        if not hasattr(self, 'model') or self.model is None:
            return self._heuristic_classify(img)

        # Convert to PIL Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Apply transforms
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        return self.class_names[predicted.item()]

    def _heuristic_classify(self, img):
        """Improved heuristic classification based on aspect ratio, size and position"""
        h, w = img.shape[:2]
        aspect_ratio = w / h
        area = w * h

        # Xe máy thường nhỏ hơn và có tỷ lệ gần 1:1
        if area < 20000 or (0.7 < aspect_ratio < 1.3 and area < 30000):
            return 'motorcycle'
        elif aspect_ratio > 1.8:  # Xe bus thường rất dài
            return 'bus'
        elif aspect_ratio < 0.8 and area > 30000:  # Xe tải thường cao
            return 'truck'
        else:
            return 'car'

def process_video(input_path, output_path, detector_model=None, classifier_model=None, conf_threshold=0.5, show=False):
    print(f"Starting video processing: {input_path} -> {output_path}")
    print(f"Detector model: {detector_model}")
    print(f"Classifier model: {classifier_model}")

    # Initialize detector and classifier
    detector = VehicleDetector(detector_model)
    classifier = VehicleClassifier(classifier_model)

    # Open video file
    print(f"Opening video file: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

    # Create video writer
    print(f"Creating video writer: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frames
    frame_count = 0
    detection_count = 0

    print("Starting frame processing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        # Detect vehicles
        detections = detector.detect(frame, conf_threshold)
        detection_count += len(detections)

        # Draw bounding boxes and labels
        for det in detections:
            x1, y1, x2, y2, conf, cls = det

            # Get vehicle type from detector
            det_class = detector.class_names[int(cls)]

            # Crop vehicle image
            vehicle_img = frame[int(y1):int(y2), int(x1):int(x2)]
            if vehicle_img.size == 0:
                continue

            # Classify vehicle
            vehicle_type = classifier.classify(vehicle_img)

            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label
            label = f"{vehicle_type} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame to output video
        out.write(frame)

        # Show frame
        if show:
            cv2.imshow('Vehicle Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing stopped by user")
                break

        frame_count += 1
        if frame_count % 10 == 0:  # Print more frequently
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%), {detection_count} detections so far")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {detection_count}")

def main():
    parser = argparse.ArgumentParser(description='Run Vehicle Detection and Classification on Video')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    parser.add_argument('--detector', type=str, default=None, help='Path to detector model')
    parser.add_argument('--classifier', type=str, default=None, help='Path to classifier model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--show', action='store_true', help='Show video during processing')
    args = parser.parse_args()

    process_video(
        args.input,
        args.output,
        args.detector,
        args.classifier,
        args.conf,
        args.show
    )

if __name__ == "__main__":
    main()
