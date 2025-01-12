import os
import cv2
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from clip_interrogator import Config, Interrogator
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

annotations_path = 'annotations_public.pkl'
video_root = './COOOL_Benchmark/processed_videos/'
output_video_dir = 'COOOL_Benchmark/road_mask_Dr_sh/'
os.makedirs(output_video_dir, exist_ok=True)

assert os.path.exists(annotations_path), f"Annotations file not found at {annotations_path}"
with open(annotations_path, 'rb') as f:
    annotations = pickle.load(f)

for vid in annotations.keys():
    video_path = os.path.join(video_root, vid + ".mp4")
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
print("All videos verified.")

ci = Interrogator(Config(clip_model_name="ViT-B-32/openai"))

def clean_caption(caption):
    unwanted_terms = [
        'ape', 'xqcow', 'emote', 'bad vhs quality', 'damaged webcam image',
        'aliased', 'orb', 'round-cropped', 'zoom shot', 'full figure',
        '200mm', 'dehazed image', 'zoomed', 'circle beard', 'clenched fist'
    ]
    for term in unwanted_terms:
        caption = caption.replace(term, '')
    caption = caption.replace(',', ' ')
    caption = ' '.join(caption.split())
    caption = caption[:35]
    return caption.strip()

seg_model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
image_processor_road = SegformerImageProcessor.from_pretrained(seg_model_name)
seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_name).to(device)
seg_model.eval()

def get_road_mask(frame_image):
    image_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
    inputs = image_processor_road(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    logits = outputs.logits
    h, w = frame_image.shape[:2]
    upsampled_logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    # class 0 is road
    road_mask = (pred == 0).astype(np.uint8) * 255
    return road_mask

text_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
reference_words = ["animal", "human"]
ref_embeddings = text_model.encode(reference_words)
similarity_threshold = 0

video_road_objects = {}  

for video in sorted(annotations.keys()):
    video_path = os.path.join(video_root, video + ".mp4")
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        print(f"Failed to open video: {video}")
        continue

    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    output_video_path = os.path.join(output_video_dir, f"{video}_processed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    video_road_objects[video] = set()
    captioned_tracks = {}
    frame_num = 0

    while True:
        ret, frame_image = video_stream.read()
        if not ret:
            break

        print(f'Processing {video}, frame {frame_num}')

        road_mask = get_road_mask(frame_image)
        img_h, img_w = frame_image.shape[:2]

        if frame_num in annotations[video]:
            for obj in annotations[video][frame_num]['challenge_object']:
                track_id = obj.get('track_id', None)
                if track_id is None:
                    continue

                x1, y1, x2, y2 = obj['bbox']
                x1_clamp = max(0, int(x1))
                y1_clamp = max(0, int(y1))
                x2_clamp = min(img_w, int(x2))
                y2_clamp = min(img_h, int(y2))

                if x2_clamp <= x1_clamp or y2_clamp <= y1_clamp:
                    continue


                if track_id not in captioned_tracks:
                    chip = frame_image[y1_clamp:y2_clamp, x1_clamp:x2_clamp]
                    chip_rgb = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
                    chip_pil = Image.fromarray(chip_rgb)
                    obj_caption = ci.interrogate(chip_pil)
                    obj_caption = clean_caption(obj_caption)
                    captioned_tracks[track_id] = obj_caption
                else:
                    obj_caption = captioned_tracks[track_id]

                cls_embedding = text_model.encode([obj_caption])[0]
                sim_animal = cosine_similarity([cls_embedding], [ref_embeddings[0]])[0][0]
                sim_person = cosine_similarity([cls_embedding], [ref_embeddings[1]])[0][0]

                if sim_animal > similarity_threshold or sim_person > similarity_threshold:
                    cx_int = int(x1 + (x2 - x1) / 2)
                    cy_int = int(y1 + (y2 - y1) / 2)
                    on_road = False
                    if 0 <= cx_int < img_w and 0 <= cy_int < img_h:
                        on_road = (road_mask[cy_int, cx_int] == 255)

                    if on_road:
                        video_road_objects[video].add(track_id)
                        print(f"Video: {video}, Track ID: {track_id}")

                        cv2.rectangle(frame_image, (x1_clamp, y1_clamp), (x2_clamp, y2_clamp), (0, 255, 0), 2)
                        cv2.putText(frame_image, f"ID:{sim_animal.round(2)}", (x1_clamp, y2_clamp + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(frame_image, f"ID:{sim_person.round(2)}", (x1_clamp, y2_clamp + 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(frame_image, f"ID: {track_id}", (x1_clamp, y1_clamp - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame_image, obj_caption, (x1_clamp, y2_clamp + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out_video.write(frame_image)
        frame_num += 1

    video_stream.release()
    out_video.release()

with open("road_objects.csv", 'w') as f:
    f.write("video,track_id\n")
    for vid, tid_set in video_road_objects.items():
        for tid in tid_set:
            f.write(f"{vid},{tid}\n")

print("Processing complete. Results saved to road_objects.csv and processed videos in 'Processed_Videos/' directory.")