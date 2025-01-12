import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import constants as CONST
from constants.utils import create_directories, load_annotations
import cv2
from hazard_caption import set_blip, process_video_cap
import torch


if __name__ == "__main__":
    with open("results.csv", 'w') as results_file:
        results_file.write("ID,Driver_State_Changed")
        for i in range(23):
            results_file.write(f",Hazard_Track_{i},Hazard_Name_{i}")
        results_file.write("\n")
        
        # Paths and configuration
        annotation_path = CONST.ANNOTATION
        video_root = CONST.VIDEO_ROOT
        output_dir = CONST.OUTPUT_DIR
        create_directories(output_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor_hazard, model_hazard = set_blip(device)

        annotations = load_annotations(annotation_path)

        video_num = 0
        for video in sorted(list(annotations.keys())):
            print(f'Processing:{video}')
            video_num += 1
            if video_num > 0:
                video_name = video+'.mp4'
                video_path = os.path.join(video_root, video_name)
                process_video_cap(video, video_name, output_dir, video_path, annotations)
                
