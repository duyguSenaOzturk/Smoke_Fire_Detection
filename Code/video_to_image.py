import os
import cv2

def extract_frames(video_path, output_folder):
    # Extract video clip name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_folder, video_name)
    os.makedirs(output_path, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    frame_index = 0

    while True:
        # Read the next frame
        ret, frame = video.read()
        if not ret:
            break

        # Save the frame
        frame_path = os.path.join(output_path, f"{video_name}_frame{frame_index}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_index += 1

        frame_count += 1

    # Release the video file
    video.release()

# Example usage
output_folder = "folder/test/fire"
folder_path = "dataset/test/fire"

# List all files in the folder
files = os.listdir(folder_path)

for file_name in files:
    extract_frames(folder_path + "/" + file_name, output_folder)