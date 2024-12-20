# Challenge Of Out-Of-Label in Autonomous Driving
![COOOL](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/Logo%20maker%20project-3%20(11).png)
 WACV2025 Challenge for detecting Out-Of-Label hazards from dashcam videos

 ## Overview
 We used different tools and models like optical flow, NAFNet, MiDaS, Segformer, and BLIP to do its tasks.

## Structure and Steps

1. **[nafnet_on_frames.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/nafnet_on_frames.ipynb)**  
   Utilizes NAFNet to improve the quality of videos.

2. **[baseline_OF_roadmask_blip_midas1.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/baseline_OF_roadmask_blip_midas1.ipynb)**  
   Applying optical flow to determine the driver state and uses MiDaS to exclude far objects. Saving frames where close objects are nearest to the dashcam.

3. **[baseline_OF_roadmask_blip_midas2.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/baseline_OF_roadmask_blip_midas2.ipynb)**  
   Generating captions for detected close objects using grounding captions in BLIP (We expanded the boxes for BLIP to see context as well).Excluding close objects recognized as "car" from this step.

4. **[road_masking.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/road_masking.ipynb)**  
   Using Segformer for road segmentation and saves the IDs of objects located on the road.

5. **[baseline_OF_roadmask_blip_midas3.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/baseline_OF_roadmask_blip_midas3.ipynb)**  
   Matching IDs of objects identified as being on the road (from Step 4) and close to the dashcam (from Step 2). Creating captions for objects without captions from Step 3. The process involves handling six scenarios:
      
      | **Scenario** | **Condition**                                                                                   | **Action**                                                                                                    |
      |--------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
      | **1**        | Objects from road segmentation and MiDaS overlap, and captions are already generated.           | Use the generated captions.                                                                                  |
      | **2**        | Objects overlap, but captions are missing.                                                     | Apply Ground Captioning. If still no captions, generate captions directly using BLIP.                        |
      | **3**        | No overlap between objects from road segmentation and MiDaS (for flying objects)               | Use IDs and captions from **Step 2**. If no captions exist, generate them directly using BLIP.               |
      | **4**        | Overlap exceeds 10 objects.                                                                    | Consider all objects from both sets. Use captions from **Step 2** or generate directly using BLIP.           |
      | **5**        | Objects on the road exceed 12.                                                                 | Focus only on these objects to avoid exceeding limits. Use captions from **Step 2** or generate directly using BLIP. |
      | **6**        | No overlap between road segmentation and MiDaS, and no captions from **Step 2**.               | Apply Ground Captioning. If still no captions, generate them directly using BLIP.                            |


6. **[correct_wrong_caption_dr_result.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/correct_wrong_caption_dr_result.ipynb)**  
   Handling cases where videos recognize wrong hazards or have incorrect captions. Reappling depth calculation to identify the frame closest to the dashcam and regenerating captions for this frame and surrounding frames. Select captions containing specific keywords, considering their frequency.

7. **[Hazard_via_depth_and_motion_slopes.py](https://github.com/ParisaHTM/coool_wacv/blob/main/Hazard_via_depth_and_motion_slopes.py)**

## Results
Original frame

![Original](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/frame_0095_original.png)

1. Applying NAFNet to improve the quality of videos
   
   ![NAFNET](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/frame_0095.png)
   
2. All objects
   
   ![All_objects](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/video_0001_hazard%20(2)_frame95.jpg)
   
3. Applying optical flow to determine driver_state and MiDaS to exclude far objects:
   
   ![MIDAS](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/video_0001_midas_hazard_v1_frame95.jpg)
   
4. Generate captions for those close objects and exclude those objects whose captions contain the word "car":
 
   ![exclude](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/video_0001_midas_hazard_v2%20(1)_frame95.jpg)
   
5. Applying road segmentation to exclude objects not on the road:
   
   ![seg](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/video_0001_midas_hazard_v4_road%20(1)_frame95.jpg)


