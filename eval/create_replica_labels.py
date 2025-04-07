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

# def create_and_save_replica_labels(args):
#     #get top 10 labels from dataset
#     seg_feat_dir = args.seg_file_config.replace('semantic_config.yaml', 'semantic_class')

#     #get the images paths
#     img_paths = sorted(glob.glob(os.path.join(args.img_folder, 'frame*.jpg')), 
#                        key=lambda file_name: int(os.path.basename(file_name).split('.')[0].split('frame')[1]))
#     semantic_class_paths = sorted(glob.glob(os.path.join(seg_feat_dir, 'semantic_class*.png')),
#                                 key=lambda file_name: int(os.path.basename(file_name).split('_')[-1].split('.')[0]))
    
#     #Create json files
#     # sample_image_idx = [0, 24, 41]
#     # user_label_names = [label[1] for label in top_labels]
#     user_label_names = ["wall", "window", "floor", "sofa", "cushion", "table", "rug", "lamp", "vase", "blinds"]
#     #print("User_label_names: ", user_label_names)
    
#     # #user_label_names = ["door", "blinds", "stool", "cabinet", "vase", "rug"]
#     # image_paths = sorted(glob.glob(os.path.join(args.feat_dir[:-4]+"gt", '*.png')), 
#     #                      key=lambda file_name: int(os.path.basename(file_name).split('/')[-1].split(".png")[0]))
#     # selected_images = [image_paths[i] for i in sample_image_idx if i < len(image_paths)]

#     #image_paths_renamed = sorted(glob.glob(os.path.join(args.feat_dir+"_rename", '*.npy')), key=lambda file_name: int(os.path.basename(file_name).split('_')[-1].split(".npy")[0]))
#     #create semantic labels for the selected images
#     for i in range(0,len(img_paths)):
#         rgb_image = cv2.imread(img_paths[i])
#         img_name = img_paths[i].split('/')[-1].split('.')[0].split('frame')[-1]
#         seg_label = cv2.imread(semantic_class_paths[i], cv2.IMREAD_UNCHANGED).astype(np.int32)
#         seg_label = cv2.resize(seg_label, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
#         output_json = args.json_folder+"/"+img_name+".json"
#         save_json_labels(args.seg_file_config, seg_label, output_json, img_name, user_label_names)
#         #cv2.imwrite(json_folder+"/"+img_name+".png", rgb_image)

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

    parser.add_argument('--langslam_dir', type=str, 
                        default="/media/saimouli/Data6T/Replica/omni_data_result/room_0_small/2025-03-23-12-56-41/psnr/before_opt")
    
    # parser.add_argument('--langsplat_dir', type=str, 
    #                      default="/media/sai/External HDD/sai/gaussian_gripper/langsplat/office4")
    
    # parser.add_argument('--langslam_nohr_dir', type=str, 
    #                     default="/media/sai/External HDD/sai/langslam_results/langslam_Replica/2024-11-01-21-10-31/psnr/before_opt")
    
    parser.add_argument('--seg_file_config', type=str, 
                        default="/media/saimouli/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/semantic_config.yaml")
    parser.add_argument('--output_name', type=str, default="label")
    
    args = parser.parse_args()

    #Step 1: read images list from langsplat and langslam
    # langsplat_img_list_viz = get_image_list(os.path.join(args.langsplat_dir, 'label_viz'))
    #langsplat_img_list = get_image_list(os.path.join(args.langsplat_dir, 'images'))
    
    #langslam_img_list_viz = get_image_list(os.path.join(args.langslam_dir, 'labels_viz', 'plant'))
    langslam_img_list = get_image_list(os.path.join(args.langslam_dir, 'gt'))
    
    #langslam_no_hr_img_list = get_image_list(os.path.join(args.langslam_nohr_dir, 'gt'))#'images'))
    
    # #Step 2: for common image names randomy select 7 images and its index
    #langsplat_numeric_map = {int(extract_numeric_part(f)): f for f in langsplat_img_list}
    # langsplat_viz_numeric_map = {int(extract_numeric_part(f)): f for f in langsplat_img_list_viz}
    
    #langslam_nohr_numeric_map = {int(extract_numeric_part(f)): f for f in langslam_no_hr_img_list}
    langslam_numeric_map = {int(extract_numeric_part(f)): f for f in langslam_img_list}
    #langslam_viz_numeric_map = {int(extract_numeric_part(f)): f for f in langslam_img_list_viz}
    
    #common_numeric_keys = list(
    #set(langsplat_numeric_map.keys())
    #.intersection(set(langslam_numeric_map.keys()))
    #.intersection(set(langslam_nohr_numeric_map.keys())))
    
    # common_numeric_keys = list(
    #     set(langsplat_numeric_map.keys())
    #     .intersection(set(langsplat_viz_numeric_map.keys()))
    # )
    
    # common_numeric_keys = list(
    #     set(langslam_numeric_map.keys())
    #     .intersection(set(langslam_viz_numeric_map.keys()))
    # )
    
    #selected_numeric_keys = random.sample(common_numeric_keys, 7)
    # [5, 20, 120, 270,340,410,490,560,630,700,780,850,920,1050]
    # selected_numeric_keys = random.sample(langslam_numeric_map.keys(), 14)
    selected_numeric_keys = [key for key in [5, 15, 25, 35, 45, 50, 70, 95, 120, 270] #, 340, 410, 490, 560, 630, 700, 780, 850, 920, 1050, 1410, 1850] 
                         if key in langslam_numeric_map]
    
    # selected_numeric_keys_langpslat = [key for key in [0, 20, 120, 270, 340, 410, 490, 560, 630, 700, 780, 850, 920, 1050, 1410, 1850] 
    #                      if key in langsplat_numeric_map]
    
    #langsplat_selected_images = [langsplat_numeric_map[key] for key in selected_numeric_keys_langpslat]
    #langsplat_indices = [langsplat_img_list.index(image) for image in langsplat_selected_images]
    
    langslam_selected_images = [langslam_numeric_map[key] for key in selected_numeric_keys]
    langslam_indices = [langslam_img_list.index(image) for image in langslam_selected_images]
    
    #langslam_nohr_selected_images = [langslam_nohr_numeric_map[key] for key in selected_numeric_keys]
    #langslam_nohr_indices = [langslam_no_hr_img_list.index(image) for image in langslam_nohr_selected_images]
    
    #Step 3: create label files for langslam and langsplat
    seg_feat_dir = args.seg_file_config.replace('semantic_config.yaml', 'semantic_class')
    
    top_labels = get_top_labels(args.seg_file_config, seg_feat_dir)
    #create a list
    user_label_names = [label[1] for label in top_labels]
    #office3: ['tablet', 'trash bin', 'tv-stand', 'camera', 'air vent']
    #room0: ['pillar', 'pot', 'candle', 'air vent', 'basket']
    #room1: ['comforter', 'nightstand', 'basket', 'air vent']
    #room2: ['sculpture', 'air vent']
    #office0: ['trash bin', 'tablet', 'camera', 'pillar', 'air vent']
    #office1: ['monitor', 'pillar', 'trash bin']
    #office3: ['tablet', 'trash bin', 'tv-stand', 'camera', 'air vent']
    #office4: ['trash bin', 'camera', 'air vent']

    # user_label_names = ["chair"]
    # process_images_labels(args, langsplat_selected_images, langsplat_indices, 
    #                      args.langsplat_dir, os.path.join(args.langsplat_dir, 'images'), top_labels=user_label_names, output_name=args.output_name)
    
    process_images_labels(args, langslam_selected_images, langslam_indices, 
                          args.langslam_dir, os.path.join(args.langslam_dir, 'gt'), top_labels=user_label_names, output_name=args.output_name)