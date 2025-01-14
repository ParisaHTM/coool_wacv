# Challenge Of Out-Of-Label in Autonomous Driving
![COOOL](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/Logo%20maker%20project-3%20(11).png)
 WACV2025 Challenge for detecting Out-Of-Label hazards from dashcam videos

 ## Overview
 We used different tools and models like optical flow, NAFNet, MiDaS, Segformer, and BLIP to do its tasks.
 ![nafnet](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/nafnet_new.jpg)
 ![overview](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/all_show_objects_new.jpg)

## Structure and Steps

1. **[nafnet](https://github.com/ParisaHTM/coool_wacv/tree/main/nafnet)**  
   Utilizes NAFNet to improve the quality of videos. Do all the steps in the folder to have access to NAFNet model.

2. **[midas_driverstate](https://github.com/ParisaHTM/coool_wacv/tree/main/midas_driverstate)**  
   Applying optical flow to determine the driver state and uses MiDaS to exclude far objects. Saving frames where close objects are nearest to the dashcam.

   Run:

   ```python3 <root_repository>/coool_wacv/midas_driverstate/main.py```

4. **[hazard_caption](https://github.com/ParisaHTM/coool_wacv/tree/main/hazard_caption)**  
   Generating captions for detected close objects using grounding captions in BLIP (We expanded the boxes for BLIP to see context as well).Excluding close objects recognized as "car" from this step.

   Run:

   ```python3 <root_repository>/coool_wacv/hazard_caption/main.py```

6. **[midas_road](https://github.com/ParisaHTM/coool_wacv/tree/main/midas_road)**  
   Using Segformer for road segmentation and saves the IDs of objects located on the road. Matching IDs of objects identified as being on the road (from Step 4) and close to the dashcam (from Step 2). Creating captions for objects without captions from Step 3. The process involves handling six scenarios:

   Run:

   ```python3 <root_repository>/coool_wacv/midas_road/main.py```
  
      
      | **Scenario** | **Condition**                                                                                   | **Action**                                                                                                    |
      |--------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
      | **1**        | Objects from road segmentation and MiDaS overlap, and captions are already generated.           | Use the generated captions.                                                                                  |
      | **2**        | Objects overlap, but captions are missing.                                                     | Apply Ground Captioning. If still no captions, generate captions directly using BLIP.                        |
      | **3**        | No overlap between objects from road segmentation and MiDaS (for flying objects)               | Use IDs and captions from **Step 2**. If no captions exist, generate them directly using BLIP.               |
      | **4**        | Overlap exceeds 10 objects.                                                                    | Consider all objects from both sets. Use captions from **Step 2** or generate directly using BLIP.           |
      | **5**        | Objects on the road exceed 12.                                                                 | Focus only on these objects to avoid exceeding limits. Use captions from **Step 2** or generate directly using BLIP. |
      | **6**        | No overlap between road segmentation and MiDaS, and no captions from **Step 2**.               | Apply Ground Captioning. If still no captions, generate them directly using BLIP.                            |

## Results

![Original](https://github.com/ParisaHTM/coool_wacv/blob/main/sample_images/detected_hazard.gif)





