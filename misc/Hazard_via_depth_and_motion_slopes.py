import os
import cv2
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def process_videos(annotations_path, video_root, results_csv, output_video_dir, submission_file, T_d=0.9, alpha=1.0, beta=0.5, gamma=1.5):
    with open(annotations_path, 'rb') as f:
        annotations = pickle.load(f)
    results_df = pd.read_csv(results_csv)
    results_df[['video','frame']] = results_df['ID'].str.rsplit('_', n=1, expand=True)
    results_df['frame'] = results_df['frame'].astype(int)
    driver_state_dict = {}
    for idx, row in results_df.iterrows():
        video_name = row['video']
        frame_num = int(row['frame'])
        driver_state_changed = (str(row['Driver_State_Changed']).lower() == "true")
        driver_state_dict[(video_name, frame_num)] = driver_state_changed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth_model_name = "Intel/dpt-large"
    depth_model = DPTForDepthEstimation.from_pretrained(depth_model_name).to(device)
    depth_processor = DPTImageProcessor.from_pretrained(depth_model_name)
    depth_model.eval()
    def estimate_depth(frame_image):
        pil_image = Image.fromarray(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
        inputs = depth_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        return prediction
    caption_model_name = "Salesforce/blip2-flan-t5-xl"
    blip_processor = Blip2Processor.from_pretrained(caption_model_name)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(caption_model_name).to(device)
    blip_model.eval()
    def generate_caption_blip2(chip_pil):
        chip_pil = chip_pil.resize((1024, 1024), Image.BICUBIC)
        prompt = "Describe the object in detail: "
        inputs = blip_processor(images=chip_pil, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        caption = caption.replace(",", " ").strip()
        return caption
    def iou(bb1, bb2):
        x1 = max(bb1[0], bb2[0])
        y1 = max(bb1[1], bb2[1])
        x2 = min(bb1[2], bb2[2])
        y2 = min(bb1[3], bb2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bb1_area = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
        bb2_area = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
        union_area = bb1_area + bb2_area - inter_area
        if union_area == 0:
            return 0
        return inter_area / union_area
    def match_existing_object(global_objects, bbox, iou_threshold=0.5):
        best_iou = 0
        best_gid = None
        for gid, gbbox in global_objects.items():
            current_iou = iou(bbox, gbbox)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gid = gid
        if best_iou >= iou_threshold:
            return best_gid
        return None
    def determine_hazards(track_positions, track_depths, T_d, alpha, beta, gamma):
        if not track_depths:
            return [], {}
        avg_depths = {gid: np.mean(dlist[len(dlist)//2:]) for gid, dlist in track_depths.items() if len(dlist) > 0}
        if not avg_depths:
            return [], {}
        sorted_tracks = sorted(avg_depths.items(), key=lambda x: x[1], reverse=True)
        top_avg_depth = sorted_tracks[0][1]
        initial_hazards = [gid for (gid, val) in sorted_tracks if (top_avg_depth - val) <= top_avg_depth]
        hazard_slopes = {}
        final_hazards = []
        for hazard_gid in initial_hazards:
            positions = track_positions.get(hazard_gid, [])
            if len(positions) > 1:
                frames_list = [p[0] for p in positions]
                x_list = [p[1] for p in positions]
                y_list = [p[2] for p in positions]
                px = np.polyfit(frames_list, x_list, 1)
                py = np.polyfit(frames_list, y_list, 1)
                slope_x = px[0]
                slope_y = py[0]
                slope_mag = np.sqrt(slope_x**2 + slope_y**2)
                avg_depth = np.mean([p[3] for p in positions])
                d_prime = avg_depth / top_avg_depth
                hazard_score = alpha * d_prime + beta * slope_mag
                if avg_depth >= T_d * top_avg_depth and hazard_score >= gamma:
                    final_hazards.append(hazard_gid)
                    hazard_slopes[hazard_gid] = (slope_x, slope_y)
            else:
                avg_depth = np.mean([p[3] for p in positions]) if positions else 0
                d_prime = avg_depth / top_avg_depth
                hazard_score = alpha * d_prime
                if avg_depth >= T_d * top_avg_depth and hazard_score >= gamma:
                    final_hazards.append(hazard_gid)
                    hazard_slopes[hazard_gid] = None
        return final_hazards, hazard_slopes
    os.makedirs(output_video_dir, exist_ok=True)
    num_hazards_per_frame = 22
    submission_rows = []
    header = ["ID","Driver_State_Changed"]
    for i in range(num_hazards_per_frame):
        header.append(f"Hazard_Track_{i}")
        header.append(f"Hazard_Name_{i}")
    submission_rows.append(",".join(header))
    for video in sorted(annotations.keys()):
        video_path = os.path.join(video_root, video + ".mp4")
        if not os.path.exists(video_path):
            continue
        frame_data = {}
        track_depths = {}
        track_positions = {}
        global_id_counter = 0
        track_to_global = {}
        global_last_bbox = {}
        global_occurrences = {}
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_num = 0
        while True:
            ret, frame_image = cap.read()
            if not ret:
                break
            depth_map = estimate_depth(frame_image)
            frame_data[frame_num] = []
            if frame_num in annotations[video]:
                objs = annotations[video][frame_num]['challenge_object']
                for obj in objs:
                    track_id = obj.get('track_id', None)
                    if track_id is None:
                        continue
                    track_id = int(track_id)
                    x1, y1, x2, y2 = obj['bbox']
                    x1c, y1c = max(int(x1),0), max(int(y1),0)
                    x2c, y2c = min(int(x2), width), min(int(y2), height)
                    if x2c <= x1c or y2c <= y1c:
                        continue
                    if x2c > 0 and y2c > 0 and x2c<=width and y2c<=height:
                        cx = (x1c+x2c)//2
                        cy = (y1c+y2c)//2
                        obj_depth = depth_map[cy, cx]
                        if track_id in track_to_global:
                            gid = track_to_global[track_id]
                        else:
                            new_gid = match_existing_object(global_last_bbox, (x1c,y1c,x2c,y2c))
                            if new_gid is not None:
                                gid = new_gid
                            else:
                                gid = global_id_counter
                                global_id_counter += 1
                            track_to_global[track_id] = gid
                        global_last_bbox[gid] = (x1c,y1c,x2c,y2c)
                        if gid not in track_depths:
                            track_depths[gid] = []
                        track_depths[gid].append(obj_depth)
                        frame_data[frame_num].append((track_id, gid, (x1c, y1c, x2c, y2c), obj_depth))
                        if gid not in track_positions:
                            track_positions[gid] = []
                        track_positions[gid].append((frame_num, cx, cy, obj_depth))
                        if gid not in global_occurrences:
                            global_occurrences[gid] = []
                        global_occurrences[gid].append((frame_num, track_id, (x1c,y1c,x2c,y2c), obj_depth))
            frames.append(frame_image.copy())
            frame_num += 1
        cap.release()
        hazards, hazard_slopes = determine_hazards(track_positions, track_depths, T_d, alpha, beta, gamma)
        global_id_to_caption = {}
        for gid in hazards:
            occ = global_occurrences[gid]
            max_occ = max(occ, key=lambda x: x[3])
            (f_num, t_id, bb, depth) = max_occ
            frame_image = frames[f_num]
            (x1c, y1c, x2c, y2c) = bb
            chip = frame_image[y1c:y2c, x1c:x2c]
            if chip.size > 0:
                chip_pil = Image.fromarray(cv2.cvtColor(chip, cv2.COLOR_BGR2RGB))
                caption = generate_caption_blip2(chip_pil)
            else:
                caption = "Hazard object"
            global_id_to_caption[gid] = caption
        out_video_path = os.path.join(output_video_dir, f"{video}_improved_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        avg_depths = {gid: np.mean(dlist[len(dlist)//2:]) for gid, dlist in track_depths.items() if len(dlist)>0} if track_depths else {}
        for f_idx, frame_image in enumerate(frames):
            driver_state_changed = driver_state_dict.get((video, f_idx), False)
            cv2.putText(frame_image, f"Driver State Changed: {driver_state_changed}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            hazard_tracks = []
            hazard_names = []
            if f_idx in frame_data:
                for (track_id, gid, bbox, obj_depth) in frame_data[f_idx]:
                    (x1c, y1c, x2c, y2c) = bbox
                    if gid in hazards:
                        hazard_tracks.append(str(track_id))
                        cv2.rectangle(frame_image, (x1c, y1c), (x2c, y2c), (0,0,255), 2)
                        track_avg_depth = avg_depths.get(gid, obj_depth)
                        caption = global_id_to_caption.get(gid, "Hazard object")
                        cv2.putText(frame_image, f"Avg Depth={track_avg_depth:.2f}", (x1c, y1c - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                        cv2.putText(frame_image, caption, (x1c, y1c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        slopes = hazard_slopes.get(gid, None)
                        if slopes is not None:
                            slope_x, slope_y = slopes
                            slope_text = f"Slope X={slope_x:.2f}, Y={slope_y:.2f}"
                            cv2.putText(frame_image, slope_text, (x1c, y2c + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                            arrow_scale = 0.3
                            arrow_end_x = int((x1c+x2c)//2 + slope_x*arrow_scale)
                            arrow_end_y = int((y1c+y2c)//2 + slope_y*arrow_scale)
                            cv2.arrowedLine(frame_image, ((x1c+x2c)//2, (y1c+y2c)//2), (arrow_end_x, arrow_end_y), (0,255,255), 2, tipLength=0.3)
                        hazard_names.append(caption)
            while len(hazard_tracks) < num_hazards_per_frame:
                hazard_tracks.append("-1")
                hazard_names.append(" ")
            row = [f"{video}_{f_idx}", str(driver_state_changed)]
            for h_t, h_n in zip(hazard_tracks, hazard_names):
                h_n = h_n.replace(",", " ")
                row.append(h_t)
                row.append(h_n)
            submission_rows.append(",".join(row))
            out_video.write(frame_image)
        out_video.release()
    with open(submission_file, 'w') as f:
        for line in submission_rows:
            f.write(line+"\n")
