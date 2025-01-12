import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import constants as CONST
from constants.utils import create_directories, load_annotations, create_output_video
from driver_state import Driver_State
from midas_hazard import *


if __name__ == "__main__":
    with open("results.csv", 'w') as results_file:
        results_file.write("ID,Driver_State_Changed")
        for i in range(23):
            results_file.write(f",Hazard_Track_{i},Hazard_Name_{i}")
        results_file.write("\n")
        
        # Paths and configuration
        annotation_path = CONST.ANNOTATION
        video_root = CONST.VIDEO_ROOT
        output_dir = "../COOOL_Benchmark/processed_videos_midas_test/"
        create_directories(output_dir)
    
        # Initialize MiDas
        midas, transform = setup_midas()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        annotations = load_annotations(annotation_path)

        # Find MiDaS value of each object
        video_num = 0
        for video in sorted(list(annotations.keys())):
            print(f'Processing:{video}')
            video_num += 1
            video_name = video+'.mp4'
            driver_state = Driver_State(video_root, video_name)
            video_path = os.path.join(video_root, video_name)
            if video_num > 0:
                driver_state.get_good_feature_from_first_frame()
                def_far_all, track_id_lifecycle = process_video(
                video_path, annotations.get(video_name, {}), video, midas, transform, device, output_dir, driver_state, results_file,
                )
            find_close_object(annotations, def_far_all, track_id_lifecycle, output_dir, video, video_path)
results_file.close()
                