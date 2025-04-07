# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
import time
from einops import rearrange
import matplotlib.pyplot as plt

import open_clip
@META_ARCH_REGISTRY.register()
class SED(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        in_features,
        fast_inference: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        #print("SED init")
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        #self.train_class_json = train_class_json
        #self.test_class_json = test_class_json
        #torch.save(self.sem_seg_head.predictor.clip_model, 'sed_model_predictor_clip.pth') # this also same as open_clip or predictor
        # self.clip_finetune = clip_finetune
        # for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
        #     if "visual" in name:
        #         if clip_finetune == "prompt":
        #             params.requires_grad = True if "prompt" in name else False
        #         elif clip_finetune == "conv":
        #             params.requires_grad = True if "conv" in name or "position" in name else False
        #         elif clip_finetune == "full":
        #             params.requires_grad = True
        #         elif clip_finetune == "mlp":
        #             params.requires_grad = True if "mlp" in name or "position" in name else False
        #         elif clip_finetune == "full_res5":
        #             if "stages.3" in name:
        #                 params.requires_grad = True
        #             else:
        #                 params.requires_grad = False
        #         else:
        #             params.requires_grad = False
        #     else:
        #         params.requires_grad = False

        # if clip_finetune == "fast_infer":
        #     for name, params in self.sem_seg_head.predictor.transformer.named_parameters():
        #         if "head1" in name or "head2" in name or "head0" in name:
        #             params.requires_grad = True
        #         else:
        #             params.requires_grad = False
        # finetune_backbone = backbone_multiplier > 0.
        # for name, params in self.backbone.named_parameters():
        #     if "norm0" in name:
        #         params.requires_grad = False
        #     else:
        #         params.requires_grad = finetune_backbone

        #self.sliding_window = sliding_window
        # self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.clip_resolution = (768, 768)
        self.sequential = False
        del self.backbone
        self.in_features = in_features
        self.fast_inference = fast_inference
        self.clip_finetune = clip_finetune

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "in_features": cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES,
            "fast_inference": cfg.TEST.FAST_INFERENCE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        #print("SED model fwd")
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]

        self.size_divisibility = -1

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        clip_images = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
        # save clip_images as .npy file
        ##import numpy as np
        #np.save("/home/vision/Documents/clip_seg/SED/clip_images.npy", clip_images.cpu().numpy())
        # t1 = time.time()
        # name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
        # clip_model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrain, device='cuda',)
        # clip_model = clip_model.float()
        
        # clip_finetune = "full"
        # for name, params in clip_model.named_parameters():
        #     if "visual" in name:
        #         if clip_finetune == "prompt":
        #             params.requires_grad = True if "prompt" in name else False
        #         elif clip_finetune == "conv":
        #             params.requires_grad = True if "conv" in name or "position" in name else False
        #         elif clip_finetune == "full":
        #             params.requires_grad = True
        #         elif clip_finetune == "mlp":
        #             params.requires_grad = True if "mlp" in name or "position" in name else False
        #         elif clip_finetune == "full_res5":
        #             if "stages.3" in name:
        #                 params.requires_grad = True
        #             else:
        #                 params.requires_grad = False
        #         else:
        #             params.requires_grad = False
        #     else:
        #         params.requires_grad = False
        
        # clip_features = clip_model.encode_image(clip_images, dense=True)
        #torch.save(self.sem_seg_head.predictor.clip_model, "sem_model_fwd.pth")
        #t1 = time.time()
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        #elapsed_time = (time.time() - t1) * 1000
        #print(f"Extracted CLIP embeddings in {elapsed_time} ms")
        ##images_resized = F.interpolate(images.tensor, size=(384, 384), mode='bilinear', align_corners=False,)
        ## features = self.backbone(images_resized)
        #np.save("/home/vision/Documents/clip_seg/SED/clip_features.npy", clip_features["clip_vis_dense"].cpu().numpy())
        # test the text features
        #clip_vis_dense = clip_features["clip_vis_dense"]
        #clip_vis_dense /= clip_vis_dense.norm(dim=-1, keepdim=True)
        #print("clip_vis_dense shape: ", clip_vis_dense.shape)

        processed_results = []
        return processed_results, clip_features