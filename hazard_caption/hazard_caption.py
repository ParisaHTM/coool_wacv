from collections import Counter
from constants.utils import create_output_video, create_directories
import cv2
import os
import pickle
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def set_blip(device):
    processor_hazard = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model_hazard = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    return processor_hazard, model_hazard
    
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

    prompt0 = "Question: Is this an animal or a car or a human or a flying-object or a floating-object or an alien? Answer:"
    cropped_image = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
    
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
    

def analyze_hazard_results(hazard_results):
    object_summary = {}

    for obj_id, captions in hazard_results.items():
        # Extract the first four words of each caption
        first_four_words = [" ".join(caption.split()[:4]) for caption in captions]
        
        # Count the occurrences of each phrase
        word_counts = Counter(first_four_words)
        
        # Find the most repetitive phrase
        most_repetitive_phrase = word_counts.most_common(1)[0][0]
        
        # Find the caption with the highest length for the most repetitive phrase
        filtered_captions = [caption for caption in captions if most_repetitive_phrase in caption]
        longest_caption = max(filtered_captions, key=len)
        
        # Save results for this object
        object_summary[obj_id] = {
            "most_repetitive_phrase": most_repetitive_phrase,
            "count": word_counts[most_repetitive_phrase],
            "longest_caption": longest_caption
        }
    
    return object_summary

def process_video_cap(video, video_name, output_dir, video_path, annotations):
    video_stream = cv2.VideoCapture(video_path)
    out = create_output_video(video_stream, output_dir, "midas_hazard_cap", video_path)
    frame = 0
    hazard_results = {}
    while video_stream.isOpened():        
        # Get objects in closer positions to dashcam using results out of MiDas
        file_path = f"./unique_ids/unique_ids_{video}_test.pkl"        
        with open(file_path, "rb") as f:
            unique_ids = pickle.load(f)
            
        ret, frame_image = video_stream.read()
        if ret == False: 
            break
            
        # Gather BBoxes from annotations
        for ann_type in ['challenge_object']:
            for i in range(len(annotations[video][frame][ann_type])):
                x1, y1, x2, y2 = annotations[video][frame][ann_type][i]['bbox']
                track_id = annotations[video][frame][ann_type][i]['track_id']
                if track_id in unique_ids:               
                    hazard_track = track_id
                    object_cor = int(x1), int(y1), int(x2), int(y2)
                    
                    # Produce captions for each object and filter them if they are cars
                    is_hazard, caption = detect_hazard(object_cor, frame_image)
                    if len(caption) > 1:
                        # print("caption:", caption)
                        if track_id not in hazard_results:
                            hazard_results[track_id] = []
                        hazard_results[track_id].append(caption)
                        
                    # Draw bounding box and object ID on the frame
                    if is_hazard:
                        cv2.rectangle(frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame_image, f"ID: {hazard_track}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(frame_image, str(is_hazard)[0], (int(x2), int(y2) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(frame_image, caption, (int(x2), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        frame +=1
        out.write(frame_image)                

    object_summary = analyze_hazard_results(hazard_results)
    create_directories(f"./hazard_results")
    output_file_path = f"./hazard_results/hazard_results_{video}.pkl"
    with open(output_file_path, "wb") as pkl_file:
        pickle.dump(object_summary, pkl_file)

    video_stream.release()
    out.release()