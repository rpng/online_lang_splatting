import cv2
import numpy as np
import yaml
import json
from pathlib import Path
from collections import Counter
import os
import glob

background_cls_list = [126, 0, 95]  # Define your background class IDs here

def get_segmentation_mask(seg_label):
    masks = []
    for sem_id in np.unique(seg_label):
        if sem_id == 0 or sem_id in background_cls_list:
            continue
        mask = seg_label == sem_id
        masks.append((sem_id, mask))
    return masks

def extract_contours(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_bounding_box(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x + w, y + h]

def calculate_area(contour):
    return cv2.contourArea(contour)

def create_labelme_annotation(seg_label, id_to_name, user_label_ids=None):
    masks = get_segmentation_mask(seg_label)
    annotations = []

    for sem_id, mask in masks:
        if user_label_ids is not None:
            if sem_id not in user_label_ids:
                continue
        contours = extract_contours(mask)
        for contour in contours:
            segmentation = [point[0].tolist() for point in contour]  # Create list of [x, y] points
            bbox = calculate_bounding_box(contour)
            area = calculate_area(contour)

            annotation = {
                "category": id_to_name[sem_id],
                "group": 1,  # Assuming a single group for simplicity
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "note": ""
            }
            annotations.append(annotation)
    
    return annotations

def save_annotations_to_json(info, annotations, json_file):
    # Ensure the directory exists
    json_file_path = Path(json_file)
    json_file_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "info": info,
        "objects": annotations
    }
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def load_labels(seg_file):
    with open(seg_file, 'r') as f:
        seg_config = yaml.safe_load(f)

    # Extract the name for each id
    id_to_name = {item['id']: item['name'] for item in seg_config['classes']}

    # for id, name in id_to_name.items():
    #     print(f"{id}: {name}")

    return id_to_name

def get_top_labels(seg_file, label_folder, top_num=10):
    id_to_name = load_labels(seg_file)
    
    label_counter = Counter()
    label_path_list = glob.glob(os.path.join(label_folder, "semantic*.png"))
    label_path_list = label_path_list[::10]  # Sample every 10th image

    # Process each image in the label folder
    for img_path in label_path_list:
        # Load the semantic label image (assuming it is a grayscale image)
        #img_path = os.path.join(label_folder, img_file)
        seg_label = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Update the counter with the semantic IDs present in the image
        label_counter.update(np.unique(seg_label))

    # Get the 10 most common labels
    top_labels = label_counter.most_common(top_num)

    # Create a list of (label_id, label_name, count) for the top 10 labels
    top_label_info = []
    for label_id, count in top_labels:
        if label_id in background_cls_list:
            continue
        label_name = id_to_name.get(label_id, "Unknown")  # Use 'Unknown' if the label ID is not found
        top_label_info.append((label_id, label_name))

    return top_label_info

def save_json_labels(seg_file, seg_label, output_json, img_name, img_idx, user_label_names=None):
    id_to_name = load_labels(seg_file)

    user_label_ids = []
    # get ids for the above label names
    if user_label_names is not None:
        for name in user_label_names:
            for id, label in id_to_name.items():
                if name == label:
                    user_label_ids.append(id)

    info = {
        "name": img_name+"_"+str(img_idx)+".jpg",
        "width": seg_label.shape[1],
        "height": seg_label.shape[0],
        "depth": 3,  # Assuming RGB images
        "note": ""
    }

    annotations = create_labelme_annotation(seg_label, id_to_name, user_label_ids)
    if len(annotations) == 0:
        return False
    save_annotations_to_json(info, annotations, output_json)
    return True
    
# if __name__ == "__main__":
#     print(get_top_labels("/media/sai/External HDD/sai/datasets/langslam/vmap/room_2/imap/00/semantic_config.yaml",
#                    "/media/sai/External HDD/sai/datasets/langslam/vmap/room_2/imap/00/semantic_class"))