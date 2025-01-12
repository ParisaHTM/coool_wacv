import cv2
import pickle
import os

def load_annotations(annotation_path):
    with open(annotation_path, 'rb') as file:
        annotations = pickle.load(file)
    return annotations

def create_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def get_video_features(video_stream):
    fps = int(video_stream.get(cv2.CAP_PROP_FPS))
    width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return fourcc, fps, width, height

def create_output_video(video_stream, output_dir, name, video_path):
    fourcc, fps, width, height = get_video_features(video_stream)
    # Initialize output video writer 
    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_{name}.mp4")   
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    return out