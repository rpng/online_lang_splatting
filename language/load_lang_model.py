from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer

import sys
import os
#get location of current file directory
currentdir = os.path.dirname(os.path.realpath(__file__))
#add current directory to path
sys.path.append(currentdir)
try:
    from language.sed import add_sed_config
except:
    from sed import add_sed_config

import numpy as np
import cv2
import argparse
import torch
import os

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    #get current path
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--config-file",
        default=currentdir+"/configs/convnextL_768.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    return parser

def get_lang_feat(inputs, model, is_lang=True):
    with torch.no_grad():
        _, dense_clip_viz = model(inputs)
        if is_lang:
            dense_clip_viz = dense_clip_viz["clip_vis_dense"]
            #clip_viz_dense = dense_clip_viz.permute(0,2,3,1)
        else:
            dense_clip_viz = dense_clip_viz
    return dense_clip_viz

def load_lang_model(model_path=None):
    if model_path is not None:
        model = torch.load(model_path, map_location="cuda:0")
    else:
        args = get_parser().parse_args()
        cfg = setup_cfg(args)
        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        model_state_path = "seg_clip_model_l.pth"
        torch.save(model, model_state_path)
    return model

