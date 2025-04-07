#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from openclip_encoder import OpenCLIPNetwork
import sys
sys.path.append( os.path.dirname(os.path.dirname(os.path.realpath(__file__))) )
#sys.path.append("/home/vision/Documents/GaussianGripMapping")
from language.autoencoder.model import Autoencoder, AutoencoderLangsplat, AutoencoderLight
import torch.nn.functional as F
import re
import shutil

import sys
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import colormaps
import yaml
from replica_save_labels import load_labels, create_labelme_annotation, save_annotations_to_json, get_top_labels
import joblib
import matplotlib.pyplot as plt

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), '*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), '*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) #- 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths

# def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
#     """
#     organise lerf's gt annotations
#     gt format:
#         file name: frame_xxxxx.json
#         file content: labelme format
#     return:
#         gt_ann: dict()
#             keys: str(int(idx))
#             values: dict()
#                 keys: str(label)
#                 values: dict() which contain 'bboxes' and 'mask'
#     """
#     gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame*.json')))
#     img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame*.jpg')))
#     target_h, target_w = 480, 640
    
#     print("gt_json_paths: ", gt_json_paths)
#     print("img_paths: ", img_paths)
#     gt_ann = {}
#     for js_path in gt_json_paths:
#         img_ann = defaultdict(dict)
#         with open(js_path, 'r') as f:
#             gt_data = json.load(f)
        
#         orig_h, orig_w = gt_data['info']['height'], gt_data['info']['width']
#         #h, w = 480, 640
#         scale_x = target_w / orig_w
#         scale_y = target_h / orig_h
        
#         idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        
#         for prompt_data in gt_data["objects"]:
#             label = prompt_data['category']
#             box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            
#             # Resize the bounding box according to the new image size
#             box[0] *= scale_x  # x1
#             box[1] *= scale_y  # y1
#             box[2] *= scale_x  # x2
#             box[3] *= scale_y  # y2
            
#             mask = polygon_to_mask((orig_h, orig_w), prompt_data['segmentation'])
#             resized_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
            
#             if img_ann[label].get('mask', None) is not None:
#                 resized_mask = stack_mask(img_ann[label]['mask'], resized_mask)
#                 img_ann[label]['bboxes'] = np.concatenate(
#                     [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
#             else:
#                 img_ann[label]['bboxes'] = box
#             img_ann[label]['mask'] = resized_mask
            
#             # # save for visulsization
#             save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
#             save_path.parent.mkdir(exist_ok=True, parents=True)
#             vis_mask_save(resized_mask, save_path)
#         gt_ann[f'{idx}'] = img_ann

#     return gt_ann, (target_h, target_w), img_paths


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE Find the maximum value point in the activation map after adding the filtering results
            scale = 30
            kernel = np.ones((scale,scale)) / (scale**2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])
            
            output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            output_path_relev)
            
            # NOTE Consistent with LERF, activation values below 0.5 are considered background.
            p_i = torch.clip(valid_map[i][k] - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (valid_map[i][k] < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, output_path_compo)
            
            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
            #resize mask to match img shape
            mask_gt = cv2.resize(mask_gt, (w, h))
            
            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)
        
        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list


def lerf_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape
    
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k]
        
        # NOTE Find the maximum value point in the smoothed activation value map
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel)
        #avg_filtered = np.expand_dims(avg_filtered, axis=-1) #remove if we have more than 1 channel
        
        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1,0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and 
                    cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
        
        # NOTE Add the averaged result to the original result to suppress noise and maintain clear activation boundaries
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num

def process_single_eval(feat_dir, eval_index, image_path, image_shape, 
                        clip_model, model, gt_ann, output_path, 
                        mask_thresh, colormap_options, logger, device, code_size=15):
    img_name = os.path.basename(image_path).split('.jpg')[0]
    image_name = Path(output_path) / f'{img_name}_{eval_index:0>5}' #f'{eval_index+1:0>5}'
    image_name.mkdir(exist_ok=True, parents=True)
    
    #laod the feature map for the given index
    compressed_sem_feats = np.zeros((len(feat_dir), len([eval_index]), *image_shape, code_size), dtype=np.float32)
    print(compressed_sem_feats.shape)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                               key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
        #print("feat_paths_lvl: ", feat_paths_lvl)
        for j, idx in enumerate([eval_index]):
            array = np.load(feat_paths_lvl[idx])
            resized_array = cv2.resize(array, (image_shape[1], image_shape[0]))
            compressed_sem_feats[i][j] = resized_array
    
    # Load the corresponding RGB image
    rgb_img = cv2.imread(image_path)[..., ::-1]
    rgb_img = (rgb_img / 255.0).astype(np.float32)
    rgb_img = cv2.resize(rgb_img, (image_shape[1], image_shape[0]))
    rgb_img = torch.from_numpy(rgb_img).to(device)
    sem_feat = compressed_sem_feats[:, 0, ...]
    sem_feat = torch.from_numpy(sem_feat).float().to(device)
    
    # Process with model and reshape
    with torch.no_grad():
        lvl, h, w, _ = sem_feat.shape
        new_h, new_w = 640, 480  # Reshaped to save memory during evaluation
        reshaped_sem_feat = F.interpolate(sem_feat.permute(0, 3, 1, 2), size=(new_h, new_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        restored_feat_reshape = model.decode(reshaped_sem_feat.flatten(0, 2))
        restored_feat_reshape = restored_feat_reshape.view(lvl, new_h, new_w, -1)
        restored_feat = F.interpolate(restored_feat_reshape.permute(0, 3, 1, 2), size=(h, w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
    # Get annotations for current eval_index
    img_ann = gt_ann[f'{eval_index}']
    clip_model.set_positives(list(img_ann.keys()))
    
    # Compute IoU and localization accuracy
    chosen_iou_list, chosen_lvl_list = [], []
    c_iou_list, c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann, thresh=mask_thresh, colormap_options=colormap_options)
    chosen_iou_list.extend(c_iou_list)
    chosen_lvl_list.extend(c_lvl)
    
    acc_num_img = lerf_localization(restored_feat, rgb_img, clip_model, image_name, img_ann)
    
    # Clear memory
    torch.cuda.empty_cache()
    del sem_feat, restored_feat, rgb_img
    
    # Compute IoU and Localization statistics
    mean_iou_chosen = sum(chosen_iou_list) / len(chosen_iou_list) if chosen_iou_list else 0
    total_bboxes = len(list(img_ann.keys()))
    acc = acc_num_img / total_bboxes if total_bboxes else 0
    
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"IoU chosen: {mean_iou_chosen:.4f}")
    logger.info(f"Chosen level: \n{chosen_lvl_list}")
    logger.info(f"Localization accuracy: {acc:.4f}")
    
    return mean_iou_chosen, acc
    
    
@torch.no_grad()
def evaluate_per_image(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, 
                       encoder_hidden_dims, decoder_hidden_dims, logger, clip_dim=768, code_size=15):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(colormap="turbo", normalize=True, colormap_min=-1.0, colormap_max=1.0)

    # Load annotations and image information
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    
    # Instantiate autoencoder and CLIP model
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = AutoencoderLangsplat(encoder_hidden_dims, decoder_hidden_dims, clip_dim=clip_dim).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Statistics containers
    chosen_iou_all, acc_num_all = [], []

    # Process each eval index
    counter = 0
    for eval_index in tqdm(eval_index_list):
        if len(gt_ann[str(eval_index)]) == 0:
            continue
        mean_iou_chosen, acc = process_single_eval(feat_dir, eval_index, image_paths[counter], image_shape, 
                                                   clip_model, model, gt_ann, output_path, mask_thresh, 
                                                   colormap_options, logger, device, code_size=15)
        chosen_iou_all.append(mean_iou_chosen)
        acc_num_all.append(acc)
        counter +=1
    
    # Overall IoU and accuracy statistics
    overall_mean_iou = sum(chosen_iou_all) / len(chosen_iou_all) if chosen_iou_all else 0
    overall_acc = sum(acc_num_all) / len(acc_num_all) if acc_num_all else 0
    
    logger.info(f"Overall mean IoU: {overall_mean_iou:.4f}")
    logger.info(f"Overall localization accuracy: {overall_acc:.4f}")
    
# @torch.no_grad()
# def evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, encoder_hidden_dims, 
#              decoder_hidden_dims, logger, clip_dim=768):
#     #device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     #device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     colormap_options = colormaps.ColormapOptions(
#         colormap="turbo",
#         normalize=True,
#         colormap_min=-1.0,
#         colormap_max=1.0,
#     )

#     gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
#     eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
#     print("feat_dir: ", feat_dir)
#     compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
#     print(compressed_sem_feats.shape)
#     for i in range(len(feat_dir)):
#         feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
#                                key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
#         #print("feat_paths_lvl: ", feat_paths_lvl)
#         for j, idx in enumerate(eval_index_list):
#             array = np.load(feat_paths_lvl[idx])
#             resized_array = cv2.resize(array, (image_shape[1], image_shape[0]))
#             compressed_sem_feats[i][j] = resized_array

#     # instantiate autoencoder and openclip
#     clip_model = OpenCLIPNetwork(device)
#     checkpoint = torch.load(ae_ckpt_path, map_location=device)
#     model = AutoencoderLangsplat(encoder_hidden_dims, decoder_hidden_dims,clip_dim=clip_dim).to(device)
#     model.load_state_dict(checkpoint)
#     model.eval()

#     chosen_iou_all, chosen_lvl_list = [], []
#     acc_num = 0
#     for j, idx in enumerate(tqdm(eval_index_list)):
#         image_name = Path(output_path) / f'{idx+1:0>5}'
#         image_name.mkdir(exist_ok=True, parents=True)
        
#         sem_feat = compressed_sem_feats[:, j, ...]
#         sem_feat = torch.from_numpy(sem_feat).float().to(device)
#         rgb_img = cv2.imread(image_paths[j])[..., ::-1]
#         rgb_img = (rgb_img / 255.0).astype(np.float32)
#         #reshape rgb img to match img shape
#         rgb_img = cv2.resize(rgb_img, (image_shape[1], image_shape[0]))
#         rgb_img = torch.from_numpy(rgb_img).to(device)

#         with torch.no_grad():
#             lvl, h, w, _ = sem_feat.shape
#             new_h, new_w = 640, 480 # reshaped to save memory during evaluation
#             reshaped_sem_feat = F.interpolate(sem_feat.permute(0, 3, 1, 2), 
#                                               size=(new_h, new_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            
#             restored_feat_reshape= model.decode(reshaped_sem_feat.flatten(0, 2))
#             restored_feat_reshape = restored_feat_reshape.view(lvl, new_h, new_w, -1)           # 3x832x1264x512
#             restored_feat = F.interpolate(restored_feat_reshape.permute(0, 3, 1, 2),
#                                           size=(h,w), mode='bilinear',align_corners=False).permute(0, 2, 3, 1)
        
#         img_ann = gt_ann[f'{idx}']
#         clip_model.set_positives(list(img_ann.keys()))
#         print("activated stream...")
#         c_iou_list, c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann,
#                                             thresh=mask_thresh, colormap_options=colormap_options)
#         chosen_iou_all.extend(c_iou_list)
#         chosen_lvl_list.extend(c_lvl)
#         print("activated stream...")
#         acc_num_img = lerf_localization(restored_feat, rgb_img, clip_model, image_name, img_ann)
#         acc_num += acc_num_img
#         torch.cuda.empty_cache()
#         del sem_feat, restored_feat, rgb_img

#     # # iou
#     mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
#     logger.info(f'trunc thresh: {mask_thresh}')
#     logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
#     logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

#     # localization acc
#     total_bboxes = 0
#     for img_ann in gt_ann.values():
#         total_bboxes += len(list(img_ann.keys()))
#     acc = acc_num / total_bboxes
#     logger.info("Localization accuracy: " + f'{acc:.4f}')

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_sorted_file_list(directory, extension='.npy', suffix_filter=None):
    """Get a sorted list of files in a directory with the given extension and optional suffix filter."""
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    if suffix_filter:
        files = [f for f in files if f.endswith(suffix_filter)]
    
    # Custom sorting key to extract numerical values
    def numerical_sort_key(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')
    
    return sorted(files, key=numerical_sort_key)

def save_json_labels(seg_file, seg_label, output_json, img_name, user_label_names=None):
    id_to_name = load_labels(seg_file)

    user_label_ids = []
    # get ids for the above label names
    if user_label_names is not None:
        for name in user_label_names:
            for id, label in id_to_name.items():
                if name == label:
                    user_label_ids.append(id)

    info = {
        "name": img_name,  # Example, you should replace it with actual frame name
        "width": seg_label.shape[1],
        "height": seg_label.shape[0],
        "depth": 3,  # Assuming RGB images
        "note": ""
    }
    
    annotations = create_labelme_annotation(seg_label, id_to_name, user_label_ids)
    save_annotations_to_json(info, annotations, output_json)
    
def rename_files_to_format(image_paths, feat_dir, extension='.npy'):
    feat_files = get_sorted_file_list(feat_dir, extension)
    rename_dir = feat_dir + "_rename"

    if not os.path.exists(rename_dir):
        os.makedirs(rename_dir)
    
    for idx, image_path in enumerate(image_paths):
        old_feat_path = os.path.join(feat_dir, feat_files[idx])
        img_name = os.path.basename(image_path).split('.')[0]
        new_feat_filename = f"{img_name}{extension}"
        new_feat_path = os.path.join(rename_dir, new_feat_filename)
        if os.path.exists(new_feat_path):
            continue
        shutil.copy(old_feat_path, new_feat_path)
        print(f"Renamed {old_feat_path} to {new_feat_path}")

if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default="office3")
    parser.add_argument("--clip_model_type", type=str, default="convnext_large_d_320")
    parser.add_argument("--clip_dim", type=int, default=768)
    #feat_dir = "/media/sai/External HDD/sai/gaussian_gripper/langsplat/office4"
    parser.add_argument('--root_dir', type=str, default="/media/sai/External HDD/sai/gaussian_gripper/langsplat/office3")
    parser.add_argument('--label_name', type=str, default="label_ov")
    #parser.add_argument("--ae_ckpt_dir", type=str, default=feat_dir+"/ae_ckpt/best_ckpt.pth")
    #parser.add_argument("--output_dir", type=str, default=feat_dir+"/eval_results")
    # parser.add_argument("--seg_file_config", type=str, 
    #                     default="/media/sai/External HDD/sai/datasets/langslam/vmap/room_0/imap/00/semantic_config.yaml")
    #parser.add_argument("--json_folder", type=str, default=feat_dir+"/label_1")
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[384, 192, 96, 48, 24, 15], #[512, 256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[24, 48, 96, 192, 384, 384, 768], #[16, 32 ,64 ,128, 256, 256, 512, 768],
                        )
    parser.add_argument('--code_size', type=int, default=15)
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    ae_ckpt_path = os.path.join(args.root_dir, "ae_ckpt", "best_ckpt.pth")
    output_dir = os.path.join(args.root_dir, "eval_results_ov")
    json_folder = os.path.join(args.root_dir, args.label_name)
    
    feat_dir = [os.path.join(args.root_dir, dataset_name + f"_{i}", "train/ours_None/renders_npy") for i in range(1,4)]
    print("feat_dir: ", feat_dir)
    output_path = os.path.join(output_dir, dataset_name+"_langsplat")
    #ae_ckpt_path = os.path.join(args.ae_ckpt_dir, "best_ckpt.pth")
    #ae_ckpt_path = args.ae_ckpt_dir
    #json_folder = os.path.join(args.json_folder, dataset_name)
    #json_folder = args.json_folder

    #get top 10 labels from dataset
    #seg_feat_dir = args.seg_file_config.replace('semantic_config.yaml', 'semantic_class')
    #top_labels = get_top_labels(args.seg_file_config, seg_feat_dir)
    
    # for langsplat the images are sampled every 10th image
    #Create json files
    # sample_image_idx = [0, 24, 41]
    # #user_label_names = [label[1] for label in top_labels]
    # user_label_names = ['wall', 'window', 'floor', 'sofa', 'cushion', 'table', 'rug', 'lamp', 'book']
    # print("User_label_names: ", user_label_names)
    
    # #user_label_names = ["door", "blinds", "stool", "cabinet", "vase", "rug"]
    # image_paths = sorted(glob.glob(os.path.join(args.feat_dir, 'images', '*.jpg')))
    #                      #key=lambda file_name: int(os.path.basename(file_name).split('/')[-1].split(".png")[0]))
    # selected_images = [image_paths[i] for i in sample_image_idx if i < len(image_paths)]
    # json_folder = args.feat_dir+"/labels"

    # #don't have to rename 
    # #path_mapping = rename_files_to_format(feat_dir[0], extension='.npy')
    # #for i in range(0,3):
    # rename_files_to_format(image_paths, feat_dir[1], extension='.npy')
    #image_paths_renamed = sorted(glob.glob(os.path.join(args.feat_dir+"_rename", '*.npy')), key=lambda file_name: int(os.path.basename(file_name).split('_')[-1].split(".npy")[0]))
    
    #create semantic labels for the selected images
    # for img_path in selected_images:
    #     rgb_image = cv2.imread(img_path)
    #     img_name = img_path.split('/')[-1].split('.')[0]
    #     mapped_class = f"semantic_class_{int(''.join(filter(str.isdigit, img_name)))}"
    #     seg_label = cv2.imread(args.seg_file_config.replace('semantic_config.yaml', 'semantic_class') 
    #                            + "/" + mapped_class + ".png", cv2.IMREAD_UNCHANGED).astype(np.int32)
        
    #     seg_label = cv2.resize(seg_label, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    #     frame_number = str(''.join(filter(str.isdigit, img_name[-5:])))
    #     output_json = json_folder+"/"+img_name+".json"
    #     save_json_labels(args.seg_file_config, seg_label, output_json, img_name, user_label_names)
    #     cv2.imwrite(json_folder+"/"+img_name+".png", rgb_image)

    #rename_files_to_format(feat_dir[0], extension='.npy')

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    evaluate_per_image(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh,
              args.encoder_dims, args.decoder_dims, 
              logger, args.clip_dim, code_size=args.code_size)