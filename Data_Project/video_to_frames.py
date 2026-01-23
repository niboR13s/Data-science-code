import cv2
import os
import glob
# Code for making images from video for training






# --- CONFIGURATION ---

# 1. Folder containing video files (.mp4, .avi, .mov, etc.)
INPUT_VIDEO_FOLDER = "Data_Project/videos_input"

# 2. Destination folders for the split images
OUTPUT_TOP_FOLDER = "Data_Project/dataset_raw/camera_top_view"
OUTPUT_BOTTOM_FOLDER = "Data_Project/dataset_raw/camera_bottom_view"

# 3. Frame step: How often to save a frame?
# 1  = Save every frame (Result: HUGE dataset, often too much duplicate data)
# 10 = Save every 10th frame (Recommended for ~30fps video)
# 30 = Save every 30th frame (Approx. 1 image per second)
FRAME_STEP = 60

def split_and_save_frames():
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_TOP_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_BOTTOM_FOLDER, exist_ok=True)

    # Find all video files (add extensions if needed)
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_VIDEO_FOLDER, ext)))
    
    print(f"Found {len(video_files)} videos to process.")
    print("-" * 30)

    total_saved_pairs = 0

    for video_path in video_files:
        # Extract filename without extension for naming the images
        filename = os.path.basename(video_path).split('.')[0]
        print(f"Processing: {filename}...")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {video_path}")
            continue
            
        frame_id = 0
        saved_count_video = 0
        
        while True:
            ret, frame = cap.read()
            
            # Stop if video is finished
            if not ret:
                break
            
            # Only save if we hit the 'step' to avoid duplicate/too many data
            if frame_id % FRAME_STEP == 0:
                h, w, _ = frame.shape
                half_h = h // 2
                
                # Split frames horizontally
                # Top: 0 to half height
                # Bottom: half height to end
                top_img = frame[0:half_h, :]
                bottom_img = frame[half_h:, :]
                
                # Create unique filenames
                # Example: video1_frame00150.jpg
                save_name = f"{filename}_frame{frame_id:06d}.jpg"
                
                top_path = os.path.join(OUTPUT_TOP_FOLDER, save_name)
                bottom_path = os.path.join(OUTPUT_BOTTOM_FOLDER, save_name)
                
                # Save images
                cv2.imwrite(top_path, top_img)
                cv2.imwrite(bottom_path, bottom_img)
                
                saved_count_video += 1
                total_saved_pairs += 1
            
            frame_id += 1
            
        cap.release()
        print(f" -> Done. {saved_count_video} frames (x2) extracted.")

    print("-" * 30)
    print(f"Total process completed!")
    print(f"Total images generated: {total_saved_pairs * 2}") # x2 because top + bottom
    print(f"Location Top: {os.path.abspath(OUTPUT_TOP_FOLDER)}")
    print(f"Location Bottom: {os.path.abspath(OUTPUT_BOTTOM_FOLDER)}")

if __name__ == "__main__":
    split_and_save_frames()