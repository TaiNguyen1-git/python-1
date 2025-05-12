import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image

class VehicleClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the vehicle classifier with a pre-trained model.
        
        Args:
            model_path: Path to a custom classification model. If None, will use a pre-trained model.
        """
        # Load pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        
        # Modify the final layer for vehicle classification
        num_classes = 4  # car, motorcycle, bus, truck
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        
        # If a custom model path is provided, load it
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded custom model from {model_path}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define class labels
        self.class_names = ['car', 'motorcycle', 'bus', 'truck']
        
        print(f"Vehicle classifier initialized using {self.device}")
    
    def classify(self, vehicle_img):
        """
        Classify a vehicle image.
        
        Args:
            vehicle_img: Cropped vehicle image (numpy array)
            
        Returns:
            Predicted class name
        """
        # For a real implementation, you would train a custom classifier
        # Here we'll use a simple heuristic based on aspect ratio and size
        
        # Convert numpy array to PIL Image
        try:
            # Ensure the image is valid
            if vehicle_img.size == 0 or vehicle_img.shape[0] == 0 or vehicle_img.shape[1] == 0:
                return "unknown"
            
            # Convert to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB))
            
            # Apply transformations
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # In a real implementation, you would use your trained model:
            # with torch.no_grad():
            #     outputs = self.model(img_tensor)
            #     _, predicted = torch.max(outputs, 1)
            #     return self.class_names[predicted.item()]
            
            # For now, use a simple heuristic based on aspect ratio and size
            height, width = vehicle_img.shape[0], vehicle_img.shape[1]
            aspect_ratio = width / height
            area = width * height
            
            if aspect_ratio > 1.5:
                if area > 20000:
                    return "truck" if aspect_ratio > 2.0 else "car"
                else:
                    return "car"
            else:
                if area < 15000:
                    return "motorcycle"
                else:
                    return "bus" if aspect_ratio < 1.2 else "truck"
                    
        except Exception as e:
            print(f"Error in classification: {e}")
            return "unknown"
