import cv2
import os

clip_count = 0
def extract_clips(video_path, clip_duration, output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames for the desired clip duration
    clip_frames = int(clip_duration * frame_rate)

    # Initialize variables
    frame_count = 0

    clip_count_internal = 0
    global clip_count

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Write the frame to the output clip file
        if clip_count_internal == 15:
            break
        output_path = f"{output_dir}/clip{clip_count}.mp4"
        if frame_count % clip_frames == 0:
            output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame.shape[1], frame.shape[0]))

        output_video.write(frame)

        frame_count += 1

        # Close the output clip file after writing the desired number of frames
        if frame_count % clip_frames == 0:
            output_video.release()
            clip_count += 1
            clip_count_internal += 1

    video.release()
    cv2.destroyAllWindows()

# Example usage
clip_duration = 2  # Duration of each clip in seconds
output_dir = 'output'

folder_path = "dataset/normal"

# List all files in the folder
files = os.listdir(folder_path)

for file_name in files:
    extract_clips(folder_path + "/" + file_name, clip_duration, output_dir)