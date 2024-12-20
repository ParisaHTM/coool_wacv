# Challenge Of Out-Of-Label in Autonomous Driving
 WACV2025 Challenge for detecting Out-Of-Label hazards from dashcam videos

 ## Overview
 We used different tools and models like optical flow, NAFNet, MiDaS, Segformer, and BLIP to do its tasks.

## Structure and Steps

1. **[nafnet_on_frames.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/nafnet_on_frames.ipynb)**  
   Utilizes NAFNet to improve the quality of videos.

2. **[baseline_OF_roadmask_blip_midas1.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/baseline_OF_roadmask_blip_midas1.ipynb)**  
   Applies optical flow to determine the driver state and uses MiDaS to exclude far objects. Saves frames where close objects are nearest to the dashcam.

3. **[baseline_OF_roadmask_blip_midas2.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/baseline_OF_roadmask_blip_midas2.ipynb)**  
   Generates captions for detected close objects using grounding captions in BLIP. Excludes close objects recognized as "car" from this step.

4. **[road_masking.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/road_masking.ipynb)**  
   Uses Segformer for road segmentation and saves the IDs of objects located on the road.

5. **[baseline_OF_roadmask_blip_midas3.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/baseline_OF_roadmask_blip_midas3.ipynb)**  
   Matches IDs of objects identified as being on the road (from Step 4) and close to the dashcam (from Step 2). Creates captions for objects without captions from Step 3. The process involves handling six scenarios:

   This step focuses on matching objects identified as being on the road (from **Step 4**) and close to the dashcam (from **Step 2**) to generate captions for objects without captions from **Step 3**. Below are the six scenarios handled in this process:
      
      | **Scenario** | **Condition**                                                                                   | **Action**                                                                                                    |
      |--------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
      | **1**        | Objects from road segmentation and MiDaS overlap, and captions are already generated.           | Use the generated captions.                                                                                  |
      | **2**        | Objects overlap, but captions are missing.                                                     | Apply Ground Captioning. If still no captions, generate captions directly using BLIP.                        |
      | **3**        | No overlap between objects from road segmentation and MiDaS.                                   | Use IDs and captions from **Step 2**. If no captions exist, generate them directly using BLIP.               |
      | **4**        | Overlap exceeds 10 objects.                                                                    | Consider all objects from both sets. Use captions from **Step 2** or generate directly using BLIP.           |
      | **5**        | Objects on the road exceed 12.                                                                 | Focus only on these objects to avoid exceeding limits. Use captions from **Step 2** or generate directly using BLIP. |
      | **6**        | No overlap between road segmentation and MiDaS, and no captions from **Step 2**.               | Apply Ground Captioning. If still no captions, generate them directly using BLIP.                            |
      
      This table provides a clear guide to handle all possible scenarios in Step 5 efficiently.


6. **[correct_wrong_caption_dr_result.ipynb](https://github.com/ParisaHTM/coool_wacv/blob/main/correct_wrong_caption_dr_result.ipynb)**  
   Handles cases where videos recognize wrong hazards or have incorrect captions. Reapplies depth calculation to identify the frame closest to the dashcam and regenerates captions for this frame and surrounding frames. Selects captions containing specific keywords, considering their frequency.


