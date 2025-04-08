import os
import cv2
import numpy as np
import yaml
import json
from pathlib import Path
from collections import Counter
import os
import glob
from replica_save_labels import create_labelme_annotation, save_annotations_to_json, get_top_labels, save_json_labels
from argparse import ArgumentParser
import re
import random

def get_image_list(directory):
    """Reads and returns the list of image filenames in a directory, sorted by numeric parts of filenames."""
    image_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Sort the image list based on numeric part extracted from the filename
    return sorted(image_list, key=lambda x: int(re.search(r'\d+', x).group()))

def extract_numeric_part(filename):
    """Extracts the numeric part from the filename."""
    match = re.search(r'\d+', filename)
    return match.group() if match else None

def process_images_labels(args, selected_imgs, selected_idx,  output_folder, img_list_path, top_labels, output_name):
    # Assuming seg_feat_dir holds the semantic label folder path
    seg_feat_dir = args.seg_file_config.replace('semantic_config.yaml', 'semantic_class')
    
    for i, img_file_name in enumerate(selected_imgs):
        rgb_img = cv2.imread(os.path.join(img_list_path, img_file_name))
        img_name = img_file_name.split('/')[-1].split('.')[0]
        img_numeric_part = str(int(re.search(r'\d+', img_name).group()))
        seg_label_path = os.path.join(seg_feat_dir, f"semantic_class_{img_numeric_part}.png")
        seg_label = cv2.imread(seg_label_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
        seg_label_resized = cv2.resize(seg_label, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        output_json = os.path.join(output_folder, output_name, f"{img_name}.json")
        
        success = save_json_labels(args.seg_file_config, seg_label_resized, output_json, img_name, selected_idx[i], user_label_names=top_labels)
        if success:
            cv2.imwrite(os.path.join(output_folder, output_name, f"{img_name}.jpg"), rgb_img)
        
if __name__ == "__main__":

    parser = ArgumentParser(description="prompt any label")

    parser.add_argument('--langslam_dir', type=str)
    parser.add_argument('--langsplat_dir', type=str)
    
    parser.add_argument('--seg_file_config', type=str)
    parser.add_argument('--output_name', type=str, default="label")
    
    args = parser.parse_args()

    #Step 1: read images list from langsplat and langslam
    langslam_seed_keys = [5, 20, 120, 270, 340, 410, 490, 560, 630, 700, 780, 850, 920, 1050, 1410, 1850]
    langsplat_seed_keys= [0, 20, 120, 270, 340, 410, 490, 560, 630, 700, 780, 850, 920, 1050, 1410, 1850]
    
    langslam_img_list = get_image_list(os.path.join(args.langslam_dir, 'gt'))
    langsplat_img_list = get_image_list(os.path.join(args.langsplat_dir, 'images'))
    
    #Step 2: for common image names randomy select 7 images and its index
    langslam_numeric_map = {int(extract_numeric_part(f)): f for f in langslam_img_list}
    langsplat_numeric_map = {int(extract_numeric_part(f)): f for f in langsplat_img_list}

    selected_numeric_keys = [key for key in langslam_seed_keys 
                         if key in langslam_numeric_map]
    
    selected_numeric_keys_langpslat = [key for key in langsplat_seed_keys 
                         if key in langsplat_numeric_map]
    
    langsplat_selected_images = [langsplat_numeric_map[key] for key in selected_numeric_keys_langpslat]
    langsplat_indices = [langsplat_img_list.index(image) for image in langsplat_selected_images]
    
    langslam_selected_images = [langslam_numeric_map[key] for key in selected_numeric_keys]
    langslam_indices = [langslam_img_list.index(image) for image in langslam_selected_images]
    
    #Step 3: create label files for langslam and langsplat
    seg_feat_dir = args.seg_file_config.replace('semantic_config.yaml', 'semantic_class')
    
    top_labels = get_top_labels(args.seg_file_config, seg_feat_dir)
    #create a list
    user_label_names = [label[1] for label in top_labels]

    process_images_labels(args, langsplat_selected_images, langsplat_indices, 
                         args.langsplat_dir, os.path.join(args.langsplat_dir, 'images'), top_labels=user_label_names, output_name=args.output_name)
    
    process_images_labels(args, langslam_selected_images, langslam_indices, 
                          args.langslam_dir, os.path.join(args.langslam_dir, 'gt'), top_labels=user_label_names, output_name=args.output_name)