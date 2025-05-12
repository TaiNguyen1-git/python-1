# Vehicle Classification System

This project detects and classifies different types of vehicles (cars, motorcycles, buses, trucks) in video footage using computer vision and deep learning techniques.

## Features

- Vehicle detection using YOLOv5
- Vehicle classification into different categories
- Real-time processing and visualization
- Output video generation with bounding boxes and labels

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/vehicle-classification.git
   cd vehicle-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script with a video file:

```
python main.py --input path/to/your/video.mp4 --output results.mp4 --show
```

### Command-line Arguments

- `--input`: Path to the input video file (required)
- `--output`: Path to save the output video (default: output.mp4)
- `--conf-threshold`: Confidence threshold for detection (default: 0.5)
- `--show`: Display the output in real-time (optional)

## Project Structure

- `main.py`: Entry point for the application
- `vehicle_detector.py`: Contains the vehicle detection logic
- `vehicle_classifier.py`: Contains the classification logic
- `utils.py`: Utility functions for video processing and visualization
- `requirements.txt`: List of dependencies

## How It Works

1. The system reads frames from the input video
2. Each frame is processed by the vehicle detector to identify potential vehicles
3. Detected vehicles are cropped and passed to the classifier
4. The classifier determines the type of each vehicle
5. Results are visualized on the frame with bounding boxes and labels
6. Processed frames are written to the output video

## Limitations

- The current classifier uses a simple heuristic based on aspect ratio and size
- For better accuracy, a custom classifier should be trained on a dataset of vehicle images
- Performance may vary depending on video quality and lighting conditions

## Future Improvements

- Train a custom classifier on a dataset of vehicle images
- Add tracking to maintain vehicle IDs across frames
- Implement more vehicle categories (e.g., bicycles, vans)
- Add counting and statistics features
