import os
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Vehicle Detector')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], 
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--output-dir', type=str, default='runs/train', help='Directory to save results')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load pre-trained YOLOv8 model
    model_path = f'yolov8{args.model_size}.pt'
    model = YOLO(model_path)
    
    print(f"Training YOLOv8{args.model_size} model on {args.data}")
    
    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=args.output_dir,
        name=f'yolov8{args.model_size}_vehicles',
        resume=args.resume,
        device=args.device
    )
    
    # Validate the model
    print("Validating the trained model...")
    model.val()
    
    print(f"Training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
