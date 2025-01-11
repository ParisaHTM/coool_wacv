import os
import cv2
import numpy as np
import pickle
import torch
from tqdm import tqdm
from constants import create_directories

def write_csv(video_frame, driver_state_flag, results_file):
    row_data = {"ID": video_frame, "Driver_State_Changed": driver_state_flag}
    for i in range(23):
        row_data[f"Hazard_Track_{i}"] = ""
        row_data[f"Hazard_Name_{i}"] = ""
        
    results_file.write(
        f"{row_data['ID']},{row_data['Driver_State_Changed']}" +
        "".join([f",{row_data[f'Hazard_Track_{i}']},{row_data[f'Hazard_Name_{i}']}" for i in range(23)]) +
        "\n"
    )
def setup_midas(model_type="DPT_Large"):
    """Initialize the MiDas model and transformations."""
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else transforms.small_transform
    return midas, transform

def is_object_far(x1, y1, x2, y2, frame_image, midas, transform, device):
    """Calculate depth of an object using MiDas."""
    input_batch = transform(frame_image).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    output = prediction.cpu().numpy()
    object_color = output[int(y1 + (abs(y2 - y1) / 2)), int(x1 + (abs(x2 - x1) / 2))]
    return object_color

def filter_objects_by_midas(det_far):    
    obj_far = []
    for track_id, color in det_far.items():
        obj_far.append((track_id, color))
    obj_far.sort(key=lambda x: x[-1], reverse=True) #in descending order
    filtered_dist_color = det_far.copy()
    
    if len(obj_far) == 2: # if there are two objects, exclude with caution
        _, cent_0 = obj_far[0]
        _, cent_1 = obj_far[1]
        dist_color = abs(cent_0 - cent_1)
        if dist_color > 6:
            filtered_dist_color = {}
            track, color = obj_far[0]
            filtered_dist_color[track] = color
            
    elif len(obj_far) > 2:# if there are more than two objects, exclude objects more freely :)
        filter_average = {}
        color_list = [color_x for _, color_x, in obj_far]
        if color_list:
            avg_color_list = sum(color_list) / len(color_list)
        else:
            avg_color_list = 0  # No objects to compare
        for track_id_obj, color in obj_far:
            if color > avg_color_list:
                filter_average[track_id_obj] = color
        if len(filter_average) == 1 and len(obj_far)>1 and abs(obj_far[0][1] - obj_far[1][1]) < 6: #to address horse video
            filter_average[obj_far[1][0]] = obj_far[1][1] # to avoid filtering out all objects
                    
        filtered_dist_color = filter_average
    return filtered_dist_color 

def retain_first_and_get_unique_ids(def_far_all, num_id):
    # first_id_per_frame = {}
    unique_ids = set()
    unique_hazards = set()
    object_values = {}  # To track values of objects across all frames
    brightest_frame_per_object = {}  # Track the brightest frame for each object

    for frame, objects in def_far_all.items():
        objects = dict(sorted(objects.items(), key=lambda item: item[1], reverse=True))
        bright_objects = []
        # Check for objects with brightness value > 15
        if num_id <= 4:
            bright_objects.extend([obj_id for obj_id, value in objects.items() if value > 15])
            unique_hazards.update(bright_objects)

            # Update the brightest frame for bright objects
            for obj_id in bright_objects:
                if obj_id not in brightest_frame_per_object or objects[obj_id] > object_values.get(obj_id, 0):
                    brightest_frame_per_object[obj_id] = (frame, objects[obj_id])
                    object_values[obj_id] = objects[obj_id]


        if objects and len(unique_hazards) == 0:  #4
            first_object_id = list(objects.keys())[0]
            unique_ids.add(first_object_id)

           # Track the brightness value and brightest frame for fallback objects
            if first_object_id not in object_values or objects[first_object_id] > object_values.get(first_object_id, 0):
                brightest_frame_per_object[first_object_id] = (frame, objects[first_object_id])
                object_values[first_object_id] = objects[first_object_id]

        
    # If num_id > 3, remove the object with the least value in `unique_ids`
    if num_id>4 and len(unique_ids)>3:
        sorted_objects = sorted(unique_ids, key=lambda obj_id: object_values.get(obj_id, float('inf')))
    
        if len(sorted_objects) > 1:  # Ensure there are at least two objects to compare
            min_value_object = sorted_objects[0]
            second_min_value = object_values.get(sorted_objects[1], float('inf'))
            
            # Check the difference between the smallest and the second smallest value
            if second_min_value - object_values[min_value_object] > 1:
                unique_ids.remove(min_value_object)

    return unique_ids, unique_hazards, brightest_frame_per_object
    
def process_video(video_path, annotations, video, midas, transform, device, output_dir, driver_state, results_file):
    video_stream = cv2.VideoCapture(video_path)
    assert video_stream.isOpened()
    out = create_output_video(video_stream, output_dir, name="midas")

    frame_num = 0
    track_id_lifecycle = {}
    def_far_all = {}
    
    while video_stream.isOpened():
        video_frame = f'{video}_{frame_num}'
        ret, frame_image = video_stream.read()
        if not ret:
            break

        # Retrieve bounding boxes and track IDs from annotations
        bboxes, det_far = {}, {}
        for ann_type in ['challenge_object']:
            for ann in annotations.get(frame_num, {}).get(ann_type, []):
                x1, y1, x2, y2 = ann['bbox']
                track_id = ann['track_id']

                # Update lifecycle
                track_id_lifecycle.setdefault(track_id, {'first_frame': frame_num, 'last_frame': frame_num})
                track_id_lifecycle[track_id]['last_frame'] = frame_num
                
                bboxes[track_id] = {'frame': frame_num, 'bbox': [x1, y1, x2, y2]}
                det_far[track_id] = is_object_far(x1, y1, x2, y2, frame_image, midas, transform, device).item()

        # Filter objects by depth
        filtered_objects = filter_objects_by_midas(det_far)
        def_far_all[frame_num] = filtered_objects
        if frame_num == 0:
            frame_num += 1
            continue
        driver_state_flag = diver_state.calc_slop(frame, frame_image)
        write_csv(video_frame, driver_state_flag, results_file)
        cv2.putText(frame_image, str(driver_state_flag), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        out.write(frame_image)

        # Draw objects on the frame
        for track_id, color in filtered_objects.items():
            bbox = bboxes[track_id]['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_image, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame_image)
        frame_num += 1

    video_stream.release()
    out.release()
    return def_far_all, track_id_lifecycle

def find_close_object(def_far_all, track_id_lifecycle, output_dir, video):
    num_id = len(track_id_lifecycle)
    frame_midas = 0        
    unique_ids, unique_hazards, brightest_frame_per_object = retain_first_and_get_unique_ids(def_far_all, num_id)
    if len(unique_hazards) > 0:
        unique_ids = unique_hazards
    video_stream_midas = cv2.VideoCapture(os.path.join(output_dir, f"{video}_midas.mp4"))
     out = create_output_video(video_stream_midas, output_dir, "midas_hazard_v1.mp4")
     hazard_midas = {}
     while video_stream_midas.isOpened():
           ret, frame_image_midas = video_stream_midas.read()
           if ret == False: #False means end of video or error
                assert frame_midas == len(annotations[video].keys())-1 #End of the video must be final frame
                break
                
            for ann_type in ['challenge_object']:
                for i in range(len(annotations[video][frame_midas][ann_type])):
                    x1, y1, x2, y2 = annotations[video][frame_midas][ann_type][i]['bbox']
                    track_id = annotations[video][frame_midas][ann_type][i]['track_id']
                    if str(track_id) in unique_ids:  # Only process unique IDs
                        cv2.rectangle(frame_image_midas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame_image_midas, f"ID: {track_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
            out_midas.write(frame_image_midas)
            frame_midas += 1
            
     unique_ids_with_brightest_frame = {obj_id: brightest_frame_per_object[obj_id][0] for obj_id in unique_ids}
        # Load the hazard_midas dictionary
        with open(f"./unique_ids/unique_ids_{video}.pkl", "wb") as f:
            pickle.dump(unique_ids_with_brightest_frame, f)
    
        video_stream_midas.release()
        out_midas.release()
    