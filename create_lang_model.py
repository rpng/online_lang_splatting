from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from language.load_lang_model import load_lang_model
from language.sed import add_sed_config
import numpy as np
import cv2
import argparse
import torch
import os
import open_clip
import tqdm
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from argparse import ArgumentParser

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
    #get location of current file directory
    currentdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "language/configs/convnextL_768.yaml")
    print("currentdir: ", currentdir)
    parser.add_argument(
        "--config-file",
        default=currentdir,
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

def get_lang_feat(inputs, model):
    with torch.no_grad():
        _, dense_clip_viz = model(inputs)
        #clip_viz_dense = dense_clip_viz
    return dense_clip_viz

def get_user_embed():
    name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    tokenizer = open_clip.get_tokenizer(name)
    texts = tokenizer(["door"]).cuda()
    #print("texts shape: ", texts.shape)
    clip_model, _, _ = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrain,
            device="cuda",)
    text_embs = clip_model.encode_text(texts)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)
    print("text_embs shape: ", text_embs.shape)
    return text_embs

def perform_similarity(clip_viz_dense, text_embs):
    sims = clip_viz_dense @ text_embs.T
    sims = sims.squeeze()
    sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    return sim_norm

    
if __name__ == "__main__":
    #test_gaussian_lang_feat()

    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    t1 = time.time()
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model_state_path = "seg_clip_model_l.pth"
    torch.save(model, model_state_path)
    print(f"Model building time: {(time.time() - t1)*1000.0} ms")

