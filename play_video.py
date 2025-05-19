import cv2
import argparse

def play_video(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create window
    window_name = "Video Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    
    # Play video
    frame_count = 0
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            frame_count += 1
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(int(1000/fps) if not paused else 0) & 0xFF
        
        if key == ord('q'):  # Quit
            print("Playback stopped by user")
            break
        elif key == ord(' '):  # Pause/Resume
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):  # Restart
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            print("Restarted playback")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Playback complete. Total frames played: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Play Video')
    parser.add_argument('--input', type=str, default='output.mp4', help='Path to input video file')
    args = parser.parse_args()
    
    play_video(args.input)

if __name__ == "__main__":
    main()
