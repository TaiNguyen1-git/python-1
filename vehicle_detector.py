import torch
import numpy as np
import cv2
import os
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path=None):
        """
        Initialize the vehicle detector with a pre-trained YOLOv8 model.

        Args:
            model_path: Path to a custom YOLO model. If None, will use the pre-trained model.
        """
        # Load YOLO model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom model from {model_path}")
        else:
            self.model = YOLO("yolov8n.pt")  # Use YOLOv8 nano model
            print("Loaded pre-trained YOLOv8n model")

        # Vehicle classes in COCO dataset (car, motorcycle, bus, truck)
        self.vehicle_classes = [2, 3, 5, 7]

        # Device (CPU or GPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Vehicle detector initialized using {self.device}")

    def detect(self, frame, conf_threshold=0.5):
        """
        Detect vehicles in a frame.

        Args:
            frame: Input frame (numpy array)
            conf_threshold: Confidence threshold for detections

        Returns:
            List of detections [x1, y1, x2, y2, confidence, class]
        """
        # Perform inference
        results = self.model(frame, conf=conf_threshold)

        # Extract detections
        vehicle_detections = []

        # Process results (YOLOv8 format)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()

                # Check if the class is a vehicle
                if cls in self.vehicle_classes and conf >= conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    vehicle_detections.append([x1, y1, x2, y2, conf, cls])

        return vehicle_detections

    def download_model(self, save_dir='models'):
        """
        Download and save the YOLOv8 model for offline use.

        Args:
            save_dir: Directory to save the model
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'yolov8n.pt')

        if not os.path.exists(save_path):
            # YOLOv8 models are already saved during initialization
            # Just copy the model file to the specified directory
            import shutil
            model_path = self.model.ckpt_path
            shutil.copy(model_path, save_path)
            print(f"Model saved to {save_path}")
        else:
            print(f"Model already exists at {save_path}")
