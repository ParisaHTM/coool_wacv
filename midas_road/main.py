import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from constants import constants as CONST
from constants.utils import create_directories, load_annotations
from road_midas import *

if __name__ == "__main__":
   annotation_path = CONST.ANNOTATION
   video_root = CONST.VIDEO_ROOT
   output_dir = CONST.OUTPUT_DIR
   create_directories(output_dir)
   device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
   
   annotations = load_annotations(annotation_path)
   
   result = object_on_road()
   results_df = pd.read_csv("results.csv")

   ci = set_vit()

   video_num = 0
   for video in sorted(list(annotations.keys())):
        print("Processing:", video)
        video_num += 1
        captioned_tracks = {}
        forbidden_word_count = {}
        if video_num > 0:
            video_name = video+'.mp4'
            video_path = os.path.join(video_root, video_name)
            results_df = process_video_road_midas_cap(video, video_path, output_dir, name, results_df, device, ci, result)
    # Save updated DataFrame to CSV
    results_df.to_csv(results_file_path_out, index=False)
    print("Results file updated successfully!")
       
