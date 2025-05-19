# Training Guide for Vehicle Classification System

This guide explains how to train the models used in the Vehicle Classification System.

## Overview

The system consists of two main components that can be trained:

1. **Vehicle Detector**: YOLOv8 model for detecting vehicles in frames
2. **Vehicle Classifier**: ResNet50 model for classifying detected vehicles

## Prerequisites

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- Other dependencies listed in `requirements.txt`

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Preparing a Dataset

You can prepare a dataset from video footage using the provided script:

```bash
python prepare_dataset.py --input path/to/video.mp4 --output-dir dataset --frame-interval 30
```

This script will:
1. Extract frames from the video
2. Detect vehicles in each frame
3. Create a dataset for both detector and classifier training

### Command-line Arguments

- `--input`: Path to input video file (required)
- `--output-dir`: Output directory for dataset (default: dataset)
- `--frame-interval`: Extract one frame every N frames (default: 30)
- `--conf-threshold`: Confidence threshold for detection (default: 0.5)
- `--split-ratio`: Train/val split ratio (default: 0.8)
- `--max-frames`: Maximum number of frames to extract (optional)

### Dataset Structure

The script creates the following directory structure:

```
dataset/
├── frames/                # Extracted video frames
├── detector/              # Dataset for detector training (YOLO format)
│   ├── train/
│   │   ├── images/        # Training images
│   │   └── labels/        # Training labels (YOLO format)
│   └── val/
│       ├── images/        # Validation images
│       └── labels/        # Validation labels (YOLO format)
└── classifier/            # Dataset for classifier training
    ├── train/
    │   ├── car/           # Car images for training
    │   ├── motorcycle/    # Motorcycle images for training
    │   ├── bus/           # Bus images for training
    │   └── truck/         # Truck images for training
    └── val/
        ├── car/           # Car images for validation
        ├── motorcycle/    # Motorcycle images for validation
        ├── bus/           # Bus images for validation
        └── truck/         # Truck images for validation
```

## Training the Vehicle Detector

To train the YOLOv8 detector:

```bash
python train_detector.py --data dataset.yaml --model-size n --epochs 100 --batch-size 16
```

### Command-line Arguments

- `--data`: Path to dataset YAML file (required)
- `--model-size`: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge) (default: n)
- `--epochs`: Number of epochs to train (default: 100)
- `--batch-size`: Batch size for training (default: 16)
- `--img-size`: Image size for training (default: 640)
- `--output-dir`: Directory to save results (default: runs/train)
- `--resume`: Resume training from last checkpoint (flag)
- `--device`: Device to use (empty for auto)

### Dataset YAML Configuration

The `dataset.yaml` file should be configured as follows:

```yaml
path: ./dataset/detector  # dataset root directory
train: train/images       # train images (relative to 'path')
val: val/images           # val images (relative to 'path')

# Classes
names:
  0: car
  1: motorcycle
  2: bus
  3: truck
```

## Training the Vehicle Classifier

To train the ResNet50 classifier:

```bash
python train_classifier.py --data-dir dataset/classifier --output-dir models --batch-size 32 --epochs 20 --pretrained
```

### Command-line Arguments

- `--data-dir`: Path to dataset directory (required)
- `--output-dir`: Directory to save model (default: models)
- `--batch-size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs to train (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--pretrained`: Use pretrained model (flag)
- `--resume`: Path to checkpoint to resume from (optional)

## Using the Trained Models

After training, you can use the trained models in your application:

### Vehicle Detector

```python
from vehicle_detector import VehicleDetector

# Initialize with your trained model
detector = VehicleDetector(model_path='runs/train/yolov8n_vehicles/weights/best.pt')

# Use the detector
detections = detector.detect(frame, conf_threshold=0.5)
```

### Vehicle Classifier

```python
from vehicle_classifier import VehicleClassifier

# Initialize with your trained model
classifier = VehicleClassifier(model_path='models/vehicle_classifier_best.pth')

# Use the classifier
vehicle_type = classifier.classify(vehicle_img)
```

## Tips for Better Training Results

1. **Data Quality**: Ensure your dataset contains diverse images with different lighting conditions, angles, and backgrounds.

2. **Data Augmentation**: The training scripts include basic data augmentation. For more complex scenarios, consider adding more augmentations.

3. **Model Size**: For the detector, larger models (YOLOv8m, YOLOv8l) will generally perform better but require more computational resources.

4. **Transfer Learning**: Using pre-trained models (enabled by default) usually leads to better results, especially with limited data.

5. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and other hyperparameters to find the optimal configuration.

6. **Validation**: Regularly monitor validation metrics to avoid overfitting.

7. **Class Balance**: Ensure your dataset has a balanced number of examples for each class.

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size or image size.
- **Slow Training**: Consider using a smaller model or reducing image size.
- **Poor Performance**: Check data quality, increase dataset size, or try a larger model.
- **Overfitting**: Add more data augmentation, reduce model complexity, or use regularization techniques.

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
