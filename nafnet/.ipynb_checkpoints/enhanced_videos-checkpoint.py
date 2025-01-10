import argparse
import cv2
import os
from tqdm import tqdm

arg_parser = argparse.ArgumentParser(description="Save enhanced version of videos")
arg_parser.add_argument("-video_path", "--video_path", type=str, default="../../../COOOL_Benchmark/COOOL_Benchmark/", 
                        help="Path to save enhanced frames")
arg_parser.add_argument("-frame_folder", "--frames_folder", type=str, default="../../../COOOL_Benchmark/frames/", 
                        help="Path to save enhanced frames")
arg_parser.add_argument("-processed_videos_folder", "--processed_videos_folder", type=str, default="../../COOOL_Benchmark/processed_videos/", 
                        help="Path to save enhanced videos")

args = arg_parser.parse_args()
video_path_videos = args.video_path
frames_folder = args.frames_folder
processed_videos_folder = args.processed_videos_folder

def get_video_files(directory, extension=".mp4"):
    return sorted([file for file in os.listdir(directory) if file.endswith(extension)])

def get_video_properties(video_path):
    """Width, height, and FPS of each video"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def create_video_from_frames(processed_frames, processed_frame_folder, output_video_path, width, height, fps):
    """
    Create a video from processed frames.

    Parameters:
        processed_frames (list): List of processed frame filenames.
        processed_frame_folder (str): Path to the folder containing processed frames.
        output_video_path (str): Path to save the output video.
        width (int): Width of the video.
        height (int): Height of the video.
        fps (float): Frames per second of the video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_name in tqdm(processed_frames, desc=f"Creating video for {os.path.basename(output_video_path)}"):
        frame_path = os.path.join(processed_frame_folder, frame_name)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video {output_video_path} created successfully.")

def main(frames_folder, processed_videos_folder, video_path_videos):
    """Create enhanced versions from processed frames by NAFNet"""
    video_files = get_video_files(video_path_videos)

    for video_num, video in enumerate(video_files, start=1):
        video_name = os.path.splitext(video)[0]
        processed_frame_folder = os.path.join(frames_folder, f"{video_name}_processed")
        output_video_path = os.path.join(processed_videos_folder, f"{video_name}.mp4")

        # Get processed frames
        processed_frames = sorted([f for f in os.listdir(processed_frame_folder) if f.endswith('.png')])
        if not processed_frames:
            print(f"No frames found for {video_name}. Skipping...")
            continue

        input_video_path = os.path.join(video_path_videos, video)
        original_width, original_height, fps = get_video_properties(input_video_path)

        # Check if frame dimensions match the original video
        sample_frame_path = os.path.join(processed_frame_folder, processed_frames[0])
        sample_frame = cv2.imread(sample_frame_path)
        processed_height, processed_width, _ = sample_frame.shape

        if original_width != processed_width or original_height != processed_height:
            print(f"Dimension mismatch for {video_name}:")
            print(f"Original: {original_width}x{original_height}, Processed: {processed_width}x{processed_height}")
            continue

        create_video_from_frames(processed_frames, processed_frame_folder, output_video_path, original_width, original_height, fps)

if __name__ == "__main__":
    os.makedirs(processed_videos_folder, exist_ok=True)
    main(frames_folder, processed_videos_folder, video_path_videos)
