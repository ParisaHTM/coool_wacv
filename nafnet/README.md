# Apply NAFNet
NAFNet is pre-trained models for image restoration.
## Steps
1. Clone NAFNet:
   
   ```git clone https://github.com/megvii-research/NAFNet.git ```
3. Download a pre-trained model from [NAFNet GitHub repository](https://github.com/megvii-research/NAFNet/tree/main) (We used the model that deblurs image ([link](https://github.com/megvii-research/NAFNet/tree/main/options/test/REDS)))
4. Extract all frames from videos.
   
   ```python3 frames_from_videos.py -video_path <PATH_TO_ORG_VIDEOS> -frame_folder <PATH_FOR_SAVING_ENHANCED_FRAMES>```
6. Apply NAFNet model to each frame.
   
   ```python3 nafnet.py -video_path <PATH_TO_ORG_VIDEOS> -frame_folder <PATH_TO_ENHANCED_FRAMES> -processed_videos_folder <PATH_TO_SAVE_ENHANCED_VIDEOS> -script_path <PATH_TO_demo.py_NAFNet> -config_path <PATH_To_DOWNLOADED_MODEL> -python_executable <PATH_TO_PYTHON_ENV>```
8. Integrate enhanced frames to created enhanced videos.
   
 ```python3 enhanced_videos.py -video_path <PATH_TO_ORG_VIDEOS> -frame_folder <PATH_TO_ENHANCED_FRAMES> -processed_videos_folder <PATH_TO_SAVED_ENHANCED_VIDEOS>```
