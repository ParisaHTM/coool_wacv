import argparse
import cv2
import os
import subprocess
from tqdm import tqdm

arg_parser = argparse.ArgumentParser(description="Save enhanced version of videos")
arg_parser.add_argument("-video_path", "--video_path", type=str, default="../../COOOL_Benchmark/COOOL_Benchmark/", 
                        help="Path to save enhanced frames")
arg_parser.add_argument("-frame_folder", "--frames_folder", type=str, default="../../COOOL_Benchmark/frames_test/", 
                        help="Path to save enhanced frames")
arg_parser.add_argument("-processed_videos_folder", "--processed_videos_folder", type=str, default="../../COOOL_Benchmark/processed_videos_test/", 
                        help="Path to save enhanced videos")
arg_parser.add_argument("-script_path", "--script_path", type=str, default="../../NAFNet/basicsr/demo.py", 
                        help="Path to save enhanced videos")
arg_parser.add_argument("-config_path", "--config_path", type=str, default="../../NAFNet/options/test/REDS/NAFNet-width64.yml", 
                        help="Path to NAFNet model")
arg_parser.add_argument("-python_executable", "--python_executable", type=str, default="/home/booster/miniconda3/envs/nafnet/bin/python", 
                        help="Path to Python in your env")


args = arg_parser.parse_args()
video_path_videos = args.video_path
frames_folder = args.frames_folder
processed_videos_folder = args.processed_videos_folder
script_path = args.script_path
config_path = args.config_path
python_executable = args.python_executable

os.makedirs(processed_videos_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)

def get_video_files(video_path):
    return sorted([file for file in os.listdir(video_path) if file.endswith('.mp4')])

def process_frame(input_frame, output_frame, script_path, config_path, python_executable):
    subprocess.run(
        [
            python_executable, 
            script_path, 
            "-opt", config_path, 
            "--input_path", input_frame, 
            "--output_path", output_frame
        ],
        check=True
    )
    print("Process was completed!")

def process_video_frames(video_name, frames_folder, script_path, config_path, python_executable):
    frame_folder = os.path.join(frames_folder, video_name)
    processed_frame_folder = os.path.join(frames_folder, f"{video_name}_processed")
    os.makedirs(processed_frame_folder, exist_ok=True)

    # Process each frame
    frames = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])
    for frame in tqdm(frames, desc=f"Processing frames for {video_name}"):
        input_frame = os.path.join(frame_folder, frame)
        print("input_frame:", input_frame)
        output_frame = os.path.join(processed_frame_folder, frame)
        process_frame(input_frame, output_frame, script_path, config_path, python_executable)

def main():
    video_files = get_video_files(video_path_videos)

    for video in video_files:
        video_name = os.path.splitext(video)[0]
        print(f"Processing video: {video_name}")
        process_video_frames(video_name, frames_folder, script_path, config_path, python_executable)

if __name__ == "__main__":
    main()