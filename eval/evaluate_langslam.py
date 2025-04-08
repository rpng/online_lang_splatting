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
from language.autoencoder.model import AutoencoderLight
import torch.nn.functional as F
import re
import shutil

import sys
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import colormaps
import yaml
from replica_save_labels import save_json_labels, get_top_labels
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
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0])# - 1 
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
        avg_filtered = np.expand_dims(avg_filtered, axis=-1) # because we only have one channel
        
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
    image_name = Path(output_path) / f'{img_name}_{eval_index:0>5}' #image_name = Path(output_path) / f'{eval_index+1:0>5}'
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
            if array.shape != image_shape:
                array = cv2.resize(array, (image_shape[1], image_shape[0]))
            compressed_sem_feats[i][j] = array
    
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
    c_iou_list, c_lvl = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann, 
                                        thresh=mask_thresh, colormap_options=colormap_options)
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
    model = AutoencoderLight(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model = model.load_from_checkpoint(ae_ckpt_path,
                                        encoder_hidden_dims=encoder_hidden_dims,
                                        decoder_hidden_dims=decoder_hidden_dims)
    model.to(device)
    model.eval()

    # Statistics containers
    chosen_iou_all, acc_num_all = [], []

    # Process each eval index
    counter = 0
    for eval_index in tqdm(eval_index_list):
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
    
if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default="room0")
    parser.add_argument("--clip_model_type", type=str, default="convnext_large_d_320")
    parser.add_argument("--clip_dim", type=int, default=768)
    parser.add_argument("--root_dir", type=str, help="/media/room_0/2025-03-23-12-56-41/psnr/before_opt")
    parser.add_argument('--label_name', type=str, default="label")
    #parser.add_argument("--output_dir", type=str, default=feat_dir+"/eval_results")
    parser.add_argument("--ae_ckpt_dir", type=str)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[384, 192, 96, 48, 24, 15],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[24, 48, 96, 192, 384, 384, 768],
                        )
    parser.add_argument('--code_size', type=int, default=15)
    
    args = parser.parse_args()

    feat_dir = [os.path.join(args.root_dir, "lang")]
    label_folder = os.path.join(args.root_dir, args.label_name)
    output_folder = os.path.join(args.root_dir, "eval_results")
    # feat_dir = [os.path.join(args.feat_dir)]
    dataset_name = args.dataset_name
    output_path = os.path.join(output_folder, dataset_name)
    os.makedirs(output_path, exist_ok=True)
    mask_thresh = args.mask_thresh
    ae_ckpt_path = args.ae_ckpt_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)
    
    evaluate_per_image(feat_dir, output_path, ae_ckpt_path, label_folder, mask_thresh,
              args.encoder_dims, args.decoder_dims, 
              logger, args.clip_dim, code_size=args.code_size)