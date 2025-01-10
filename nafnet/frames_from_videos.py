import argparse
import cv2
import os
from tqdm import tqdm

arg_parser = argparse.ArgumentParser(description="Save enhanced version of videos")
arg_parser.add_argument("-video_path", "--video_path", type=str, default="../../../COOOL_Benchmark/COOOL_Benchmark/", 
                        help="Path to save enhanced frames")
arg_parser.add_argument("-frame_folder", "--frames_folder", type=str, default="../../../COOOL_Benchmark/frames/", 
                        help="Path to save enhanced frames")

args = arg_parser.parse_args()
video_path_videos = args.video_path
frames_folder = args.frames_folder

def get_video_files(directory, extension=".mp4"):
    return sorted([file for file in os.listdir(directory) if file.endswith(extension)])

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def extract_frames_from_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Save frame as PNG
            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'")

def main(video_directory, output_directory):
    video_files = get_video_files(video_directory)
    for video in video_files:
        # Create a folder for each video
        video_name = os.path.splitext(video)[0]
        output_folder = os.path.join(output_directory, video_name)
        create_directory(output_folder)
        
        video_path = os.path.join(video_directory, video)
        extract_frames_from_video(video_path, output_folder)

if __name__ == "__main__":
    create_directory(frames_folder)
    main(video_path_videos, frames_folder)
