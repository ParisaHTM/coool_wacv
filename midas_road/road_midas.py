from constants.utils import create_output_video
from hazard_caption.hazard_caption import set_blip
import pandas as pd



def set_vit():
    ci = Interrogator(Config(clip_model_name="ViT-B-32/openai"))
    return ci

def choose_longest_cap(captions, mutual_values):
    for value in mutual_values:
        if value in caption_midas_ids:
           captions[value] = caption_midas_ids[value]['longest_caption']
    return captions
        
def object_on_road():
    df = pd.read_csv("./road_objects.csv")
    result = df.groupby('video')['track_id'].apply(list).to_dict()
    return result

def get_objects_mutual_indiv(video, result):
    if video in result:
        road_mask_result = result[video]  # int
    else:
        road_mask_result = {}            
    road_mask_str = [str(val) for val in road_mask_result] #str
           
    id_caption_v3 = f"hazard_results/hazard_results_{video}_v3.pkl"
    if os.path.exists(id_caption_v3):
        with open(id_caption_v3, "rb") as f:
            caption_midas_ids = pickle.load(f)  #str
    else:
        caption_midas_ids = {}
           
    midas = f"unique_ids/unique_ids_{video}.pkl"
    if os.path.exists(midas):
        with open(midas, "rb") as f:
            midas_ids = pickle.load(f)  #str
    else:
        midas_ids = {}

    mutual_values = [key for key in road_mask_str if key in midas_ids]  # Objects in both road and close to the vehicle
    captions = {}
    captions = choose_longest_cap(captions, mutual_values)
    return road_mask_str, caption_midas_ids, midas_ids, mutual_values, captions

def detect_hazard(object_cor, frame):
    is_it_hazard = False
    caption = ""
    x1, y1, x2, y2 = object_cor
    if x1 < 0:
       x1 = 0
    if x2 < 0:
       x2 = 0
    if y1 < 0:
       y1 = 0
    if y2 < 0:
       y2 = 0
    frame_height, frame_width, _ = frame.shape

    # Check if the coordinates are within bounds
    if (y1 - 20 >= 0 and y2 + 20 <= frame_height and 
        x1 - 20 >= 0 and x2 + 20 <= frame_width):
        cropped_object = frame[y1-20 :y2 + 20, x1 - 20:x2 + 20]
    else:
        cropped_object = frame[y1:y2, x1:x2]

    prompt0 = "Question: Is this an animal or a car or a human or a flying-object or an floating-object on the road or an alien? Answer:"
    cropped_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
    cropped_image = cropped_image.resize((512,512))
    inputs = processor_hazard(cropped_image, text=prompt0, return_tensors="pt").to(device, torch.float16)
    generated_ids = model_hazard.generate(**inputs, max_new_tokens=10)
    generated_text_general = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    generated_text_general = generated_text_general.split()[-1]

    contains_car = "car" in generated_text_general.lower()
    contains_human = any(word in generated_text_general.lower() for word in ["human", "person", "man", "woman", "men", "women", "kid"])
    contains_animal = any(word in generated_text_general.lower() for word in ["animal", "dog", "cat", "snake", "bird", "Kangaroo", "moose", "deer", "rabbit", "lizard", "cow", "horse", "goose", "duck", "mouse"])
    contains_flyingobject = "flying-object" in generated_text_general.lower()
    contains_object = any(word in generated_text_general.lower() for word in ["road", "alien"])
    
    if contains_car:
        prompt1 = "Question: Is this car in the opposing lane or a preceding vehicle or in the wrong way? Answer:"
        inputs = processor_hazard(cropped_image, text=prompt1, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=100)
        generated_text = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        contains_lane = any(word in generated_text.lower() for word in ["wrong", "opposing"])
        if contains_lane:
            is_it_hazard = False

    if contains_human:
        # Specific prompt to describe appearance
        prompt_appearance = " This person is wearing a"
        inputs = processor_hazard(cropped_image, text=prompt_appearance, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=20)  # Limit the response to approximately 10 words
        appearance_caption = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        prompt1 = "Question: Is this person crossing the street? Answer:"
        inputs = processor_hazard(cropped_image, text=prompt1, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=100)
        generated_text = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        contains_lane = any(word in generated_text.lower() for word in ["yes"])
        
        if contains_lane:
            caption = str(generated_text_general) + " The person is going to cross the road " + appearance_caption
            is_it_hazard = True 
            
    if contains_animal:
        # Specific prompt to describe appearance
        prompt_color = f" The color of the {generated_text_general} "
        inputs = processor_hazard(cropped_image, text=prompt_color, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=20)  # Limit the response to approximately 10 words
        appearance_color = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        prompt_appearance = f" The characteristic of the {generated_text_general} "
        inputs = processor_hazard(cropped_image, text=prompt_appearance, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=20)  # Limit the response to approximately 10 words
        appearance_caption = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        prompt1 = "Question: Is this animal crossing the street? Answer:"
        inputs = processor_hazard(cropped_image, text=prompt1, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=100)
        generated_text = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        contains_lane = any(word in generated_text.lower() for word in ["yes"])
        
        if contains_lane:
            caption = "It is a "+ str(generated_text_general) + f". The {generated_text_general} is going to cross the road {appearance_color}. {appearance_caption}."
            is_it_hazard = True  
            
    if contains_flyingobject:
        # Specific prompt to describe appearance
        prompt_color = f" The color of the {generated_text_general} "
        inputs = processor_hazard(cropped_image, text=prompt_color, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=20)  # Limit the response to approximately 10 words
        appearance_color = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        prompt_appearance = f" The characteristic of the {generated_text_general} "
        inputs = processor_hazard(cropped_image, text=prompt_appearance, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=20)  # Limit the response to approximately 10 words
        appearance_caption = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        prompt1 = "Question: Is this object thrown into the air? Answer:"
        inputs = processor_hazard(cropped_image, text=prompt1, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=100)
        generated_text = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        contains_lane = any(word in generated_text.lower() for word in ["yes"])
        if contains_lane:
            caption = "It is a "+ str(generated_text_general) + f". The {generated_text_general} is thrown to air {appearance_color}. {appearance_caption}."
            is_it_hazard = True

    if contains_object:
        # Specific prompt to describe appearance
        prompt_color = f" The color of the {generated_text_general} "
        inputs = processor_hazard(cropped_image, text=prompt_color, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=20)  # Limit the response to approximately 10 words
        appearance_color = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        prompt_appearance = f" The characteristic of the {generated_text_general} "
        inputs = processor_hazard(cropped_image, text=prompt_appearance, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=20)  # Limit the response to approximately 10 words
        appearance_caption = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        prompt1 = "Question: Is this object on the road? Answer:"
        inputs = processor_hazard(cropped_image, text=prompt1, return_tensors="pt").to(device, torch.float16)
        generated_ids = model_hazard.generate(**inputs, max_new_tokens=100)
        generated_text = processor_hazard.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        contains_lane = any(word in generated_text.lower() for word in ["yes"])
        if contains_lane:
            caption = "It is an object on the "+ str(generated_text_general) + f". The object is on the road {appearance_color}. {appearance_caption}."
            is_it_hazard = True
        
    return is_it_hazard, caption

def process_track_ids(caption_midas_ids, forbidden_word_count, hazard_results, captioned_tracks, frame_image, ci, x1, y1, x2, y2):
    # Case 1: Caption already exists in caption_midas_ids (from hazard_results_{video}_v3.pkl:contains caption from most frequent caption that were produced in last step and save the longest caption)
    if track_id in caption_midas_ids:
        caption_to_display = caption_midas_ids[track_id]['longest_caption']
        hazard_tracks.append(track_id)
        hazard_captions.append(caption_to_display)
        
        # Draw bounding box and caption
        cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        display_text = f"ID: {track_id} | {caption_to_display}"
        cv2.putText(frame_image, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)
            
    else:
        if track_id not in captioned_tracks:
            print("track_id3:", track_id)   
            # Case 2: Generate caption if not in caption_midas_ids
            object_cor = (int(x1), int(y1), int(x2), int(y2))
            is_hazard, caption = detect_hazard(object_cor, frame_image)
    
            if len(caption) > 1:
                # Successful caption generation
                hazard_results.setdefault(track_id, []).append(caption)
                hazard_tracks.append(track_id)
                hazard_captions.append(caption)
                captioned_tracks[track_id] = caption
    
                # Draw bounding box and caption
                cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                display_text = f"ID: {track_id} | {caption}"
                cv2.putText(frame_image, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 2)
            else:
                # Use VIT-B-L again for caption
                hazard_tracks.append(track_id)
                forbidden_word_count[track_id] = forbidden_word_count.get(track_id, 0)

                chip = frame_image[int(y1):int(y2), int(x1):int(x2)]
                hazard_chip = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
                hazard_chip = Image.fromarray(hazard_chip)
                hazard_chip = hazard_chip.resize((512, 512))

                # Generate caption
                caption = ci.interrogate(hazard_chip)
                caption1 = caption.replace(",", " ")
                caption = " ".join(caption1.split()[:10])

                # Check for forbidden words
                if "car" in caption1.lower() or "vehicle" in caption1.lower():
                    forbidden_word_count[track_id] += 1
                    hazard_tracks.remove(track_id)  # Remove invalid track
                else:
                    captioned_tracks[track_id] = caption
                    hazard_captions.append(caption)

                    # Draw bounding box and caption
                    cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    display_text = f"ID: {track_id} | {caption}"
                    cv2.putText(frame_image, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 2)
        else:
            hazard_tracks.append(track_id)
            caption = captioned_tracks[track_id]
            hazard_captions.append(caption)

        # Draw bounding box and caption
        cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        display_text = f"ID: {track_id} | {caption}"
        cv2.putText(frame_image, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)

    return hazard_tracks, hazard_captions

def more_than_ten_obj_road(road_mask_str, midas_ids, track_id, forbidden_word_count, caption_midas_ids, hazard_results, captioned_tracks, frame_image, ci, x1, y1, x2, y2):
    combined_ids_road = list(set(road_mask_str) | set(midas_ids.keys()))
    if track_id in forbidden_word_count:
        if forbidden_word_count[track_id] >= 10:
            continue
    if track_id in combined_ids_road:   
        if caption_midas_ids:
          caption_create = False
          if caption_midas_ids and caption_create == False:
              caption_create = True
              first_key = list(caption_midas_ids.keys())[0]
              caption_0 = caption_midas_ids[first_key]['longest_caption']
          hazard_tracks.append(track_id)
          hazard_captions.append(caption_0)
        else:                           
          hazard_tracks, hazard_captions = process_track_ids(
                                            caption_midas_ids, 
                                            forbidden_word_count, 
                                            hazard_results, 
                                            captioned_tracks, 
                                            frame_image, 
                                            ci, 
                                            x1, y1, x2, y2,
                                        )            
    return  hazard_tracks, hazard_captions

def all_in_road_midas(road_mask_str, caption_midas_ids):
    combined_ids = list(set(road_mask_str) | set(caption_midas_ids.keys()))
    caption_create = False
    if caption_midas_ids and caption_create == False:
       caption_create = True
       first_key = list(caption_midas_ids.keys())[0]
       caption_0 = caption_midas_ids[first_key]['longest_caption']
    hazard_tracks.append(track_id)
    hazard_captions.append(caption_0)
    return hazard_tracks, hazard_captions

def no_cap(track_id, frame_image, hazard_results, hazard_tracks, hazard_captions, ci, x1, y1, x2, y2):
    object_cor = int(x1), int(y1), int(x2), int(y2)                           
    is_hazard, caption = detect_hazard(object_cor, frame_image)
    
    ###### Case 2) If blip produces any caption
    if len(caption) > 1:
        if track_id not in hazard_results:
            hazard_results[track_id] = []
        hazard_results[track_id].append(caption)
        hazard_tracks.append(track_id)
        hazard_captions.append(caption)
        
        cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        display_text = f"ID: {track_id} | {caption}"                            
        cv2.putText(frame_image, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)
    #If blip cannot produce any caption: we get help from VIT-B-L again with bbox 
    if len(caption) == 0:
      hazard_tracks.append(track_id)
      if track_id not in captioned_tracks:
        chip = frame_image[int(y1):int(y2), int(x1):int(x2)]
        hazard_chip = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
        hazard_chip = Image.fromarray(hazard_chip)
        hazard_chip = hazard_chip.resize((512, 512))
        # Generate caption
        caption = ci.interrogate(hazard_chip)

        caption1 = caption.replace(","," ")
        caption = " ".join(caption1.split()[:10])
        # print("caption;", caption)
        
        captioned_tracks[track_id] = caption
      else:
        caption = captioned_tracks[track_id]
      hazard_captions.append(caption)  

      cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
      display_text = f"ID: {track_id} | {caption}"                            
      cv2.putText(frame_image, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)
    return hazard_tracks, hazard_captions
    
def process_video_road_midas_cap(video, video_path, output_dir, name, results_df, device, ci, result):
    forbidden_word_count = {}
    video_stream = cv2.VideoCapture(video_path)
    out = create_output_video(video_stream, output_dir, name, video_path)
    processor_hazard, model_hazard = set_blip(device)
    road_mask_str, caption_midas_ids, midas_ids, mutual_values, captions = get_objects_mutual_indiv(video, result)
    
    frame = 0
    captioned_tracks = {}
    hazard_results = {}  
    while video_stream.isOpened():    
        ret, frame_image = video_stream.read()
        if ret == False: #False means end of video or error
            assert frame == len(annotations[video].keys()) #End of the video must be final frame
            break
            
        if frame == 0:
            frame += 1
            continue
      
        hazard_tracks = []
        hazard_captions = []
        
        video_frame_id = f"{video}_{frame}"
        # print(video_frame_id)
        driver_state_flag = results_df.loc[results_df['ID'] == video_frame_id, 'Driver_State_Changed'].values[0] 
        row_data = {"ID": video_frame_id, "Driver_State_Changed": driver_state_flag}
        for i in range(23):
            row_data[f"Hazard_Track_{i}"] = ""
            row_data[f"Hazard_Name_{i}"] = ""
            
        for i in range(len(annotations[video][frame][ann_type])):
            x1, y1, x2, y2 = annotations[video][frame][ann_type][i]['bbox']
            track_id = annotations[video][frame][ann_type][i]['track_id']
            
            # Case 1) caption from midas when there are mutual objects from road mask and MiDas
            if 0 < len(captions) <= 6:
               if track_id in captions:
                   hazard_tracks.append(track_id)
                   hazard_captions.append(captions[track_id])
                   cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                   display_text = f"ID: {track_id} | {captions[track_id]}"                            
                   cv2.putText(frame_image, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 2)
            # Case 6) when more than 10 hazards and midas cannot find them --> objects from road mask and createing caption for them
            if len(road_mask_str) > 12 and len(captions) < 7:
                hazard_tracks, hazard_captions = more_than_ten_obj_road(road_mask_str, midas_ids, track_id, forbidden_word_count, caption_midas_ids, hazard_results, captioned_tracks, frame_image, ci, x1, y1, x2, y2)

            # Case 5) vidoes with more than 10 hazard ---> combinaiton of all objects from MiDas and road masking
            if len(captions) >= 7:
                hazard_tracks, hazard_captions = all_in_road_midas(road_mask_str, caption_midas_ids)

            if len(captions) == 0:
                if len(caption_midas_ids) > 0:  #Case 5) when no road but caption midas: caption was produces in previous step
                    if track_id in caption_midas_ids:
                       hazard_tracks.append(track_id)
                       caption_to_display = caption_midas_ids[track_id]['longest_caption']
                       hazard_captions.append(caption_to_display)

                       cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                       display_text = f"ID: {track_id} | {caption_to_display}"                            
                       cv2.putText(frame_image, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 2)
                        
                if len(caption_midas_ids) == 0:
                    if mutual_values:
                       if track_id in mutual_values:  #4 both road and midas same but no caption midas
                          if track_id in forbidden_word_count:
                              if forbidden_word_count[track_id] >= 10:
                                continue
                       hazard_tracks, hazard_captions = no_cap(track_id, frame_image, hazard_results, hazard_tracks, hazard_captions, ci, x1, y1, x2, y2)
                        
                    else:
                       if track_id in midas_ids:
                          if track_id in forbidden_word_count:
                              if forbidden_word_count[track_id] >= 10:
                                continue
                          hazard_tracks, hazard_captions = no_cap(track_id, frame_image, hazard_results, hazard_tracks, hazard_captions, ci, x1, y1, x2, y2)
    
        #write csv file
        for i in range(min(len(hazard_tracks), 23)):
            row_data[f"Hazard_Track_{i}"] = hazard_tracks[i]
            row_data[f"Hazard_Name_{i}"] = hazard_captions[i]

         # Update DataFrame
        if video_frame_id in results_df['ID'].values:
            for key, value in row_data.items():
                results_df.loc[results_df['ID'] == video_frame_id, key] = value
        else:
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
                  

        frame += 1
        out.write(frame_image)
        
    # Save updated DataFrame to CSV
    results_df.to_csv(results_file_path_out, index=False)
    print("Results file updated successfully!")
    video_stream.release()
    out.release()
    return results_df
                
                
    
    