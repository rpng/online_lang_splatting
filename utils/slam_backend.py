"""
Backend Module

This module implements the backend component of a Gaussian Splatting SLAM system.
The backend handles mapping, optimization, and scene representation learning.
It includes support for language-guided scene understanding through online training
of autoencoder models that compress visual language features.

Main responsibilities:
- Map building and optimization
- Gaussian point densification and pruning
- Language feature extraction and encoding
- Online autoencoder training for language features
- Keyframe management
"""

import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping

from language.autoencoder.model import AutoencoderLight, EncoderDecoderOnline
from language.load_lang_model import get_lang_feat
import cv2
import open_clip
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
import tensorboard
from language.supervisedNet import LangSupervisedNet
from torch import optim

class BackEnd(mp.Process):
    """
    SLAM Backend class responsible for mapping and optimization.
    
    The backend builds and refines the 3D Gaussian model of the scene based on
    keyframes provided by the frontend. It also handles language feature
    extraction and encoding using autoencoder models, with online training
    to adapt to specific environments.
    """
    def __init__(self, config: Dict[str, Any], lang_model: Optional[torch.nn.Module] = None):
        """
        Initialize the SLAM backend.
        
        Args:
            config: Configuration dictionary containing parameters for mapping,
                   optimization, and language feature extraction
            lang_model: Pre-trained language model for feature extraction
        """
        super().__init__()
        
        # Configuration
        self.config = config
        self.lang_model = lang_model
        
        # These will be set later
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        
        # System state
        self.live_mode = False
        self.map_counter = 0
        self.lamda_lang = 1.0
        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = False
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True  # RGB-D mode is always initialized
        self.keyframe_optimizers = None
        self.is_single_stage = config["language"]["single_stage_ae"]
        
        # Initialize language models and autoencoders if language training enabled
        self._init_language_models()

    def _init_language_models(self):
        """Initialize language models and autoencoders if language training is enabled."""
        # Initialize text embeddings for similarity calculation
        if self.config["language"]["language_train"]:
            with torch.no_grad():
                name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
                tokenizer = open_clip.get_tokenizer(name)
                texts = tokenizer(["table"]).cuda()
                clip_model, _, _ = open_clip.create_model_and_transforms(
                        name,
                        pretrained=pretrain,
                        device="cuda")
                self.text_embs = clip_model.encode_text(texts)
                self.text_embs /= self.text_embs.norm(dim=-1, keepdim=True)

        # Initialize autoencoder models
        if self.config["language"]["language_train"]:
            t1 = time.time()
            
            # Set up autoencoder dimensions based on configuration
            if self.is_single_stage:
                encoder_hidden_dims = [384, 192, 96, 48, 24, 15]
                decoder_hidden_dims = [24, 48, 96, 192, 384, 384, 768]
            else:
                encoder_hidden_dims = [512, 256, 128, 64, 32]
                decoder_hidden_dims = [192, 256, 384, 512, 768]
                # Initialize online autoencoder for dynamic adaptation
                self.online_auto = EncoderDecoderOnline().to("cuda")
            
            # Load pretrained autoencoder model
            ckpt_path = self.config["language"]["auto_ckpt_path"]
            self.auto_model = AutoencoderLight(
                encoder_hidden_dims, 
                decoder_hidden_dims, 
                768, 
                is_MLP=True
            ).to("cuda")
            
            self.auto_model = self.auto_model.load_from_checkpoint(
                ckpt_path, 
                encoder_hidden_dims=encoder_hidden_dims, 
                decoder_hidden_dims=decoder_hidden_dims,
                is_MLP=True
            )
            self.auto_model.to("cuda")
            self.auto_model.eval()
            
            Log(f"Autoencoder model loaded in {time.time() - t1:.2f} s")

            # Load high-resolution language model if specified
            if self.config["language"]["hr_model"]:
                hr_ckpt_path = self.config["language"]["hr_ckpt_path"]
                self.hr_model = LangSupervisedNet()
                self.hr_model = self.hr_model.load_from_checkpoint(hr_ckpt_path)
                self.hr_model = self.hr_model.to("cuda")
                self.hr_model.eval()

    def set_hyperparams(self):
        """Set hyperparameters from the configuration dictionary."""
        # Results saving parameters
        self.save_results = self.config["Results"]["save_results"]

        # Initialization parameters
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        
        # Mapping parameters
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        
        # System configuration
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def add_next_kf(self, frame_idx: int, viewpoint: Any, 
                   init: bool = False, scale: float = 2.0, 
                   depth_map: Optional[np.ndarray] = None):
        """
        Add the next keyframe to the map.
        
        Args:
            frame_idx: Index of the frame to add
            viewpoint: Camera viewpoint for the frame
            init: Whether this is the initial keyframe
            scale: Scale factor for Gaussian initialization
            depth_map: Depth map for the frame
        """
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    @torch.no_grad()
    def perform_similarity(self, clip_viz_dense: torch.Tensor) -> torch.Tensor:
        """
        Calculate normalized similarity between visual features and text embedding.
        
        Args:
            clip_viz_dense: Visual features from CLIP model
            
        Returns:
            Normalized similarity map
        """
        sims = clip_viz_dense @ self.text_embs.T.to("cuda")
        sims = sims.squeeze()
        sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
        return sim_norm
    
    @torch.no_grad()
    def visualize_similarity(self, recon_online: torch.Tensor, recon_coco: torch.Tensor):
        """
        Visualize similarity maps for debugging.
        
        Args:
            recon_online: Reconstructed features from online autoencoder
            recon_coco: Reconstructed features from base autoencoder
        """
        # Calculate similarity maps
        sim_norm_recon = self.perform_similarity(recon_online)
        sim_norm_recon_coco = self.perform_similarity(recon_coco)
        
        # Reshape to 2D maps (192x192)
        sim_norm_recon = sim_norm_recon.view(192, 192)
        sim_norm_recon_coco = sim_norm_recon_coco.view(192, 192)
    
        # Convert to uint8 for visualization
        sim_norm_recon = (sim_norm_recon.detach().cpu().numpy() * 255).astype(np.uint8)
        sim_norm_recon_coco = (sim_norm_recon_coco.detach().cpu().numpy() * 255).astype(np.uint8)

        # Apply TURBO colormap for visualization
        heatmap_recon = cv2.applyColorMap(sim_norm_recon, cv2.COLORMAP_TURBO)
        heatmap_recon_coco = cv2.applyColorMap(sim_norm_recon_coco, cv2.COLORMAP_TURBO)

        # Display visualization windows
        cv2.imshow("Sim Online", heatmap_recon)
        cv2.imshow("General Sim", heatmap_recon_coco)
        cv2.waitKey(10)

    def reset(self):
        """Reset the backend state for a new mapping session."""
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True  # RGB-D mode is always initialized
        self.keyframe_optimizers = None

        # Remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        
        # Clear the backend queue
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def train_online_autoencoder(self, 
                               features: torch.Tensor, 
                               online_auto_optimizer: torch.optim.Optimizer,
                               online_auto_scheduler: torch.optim.lr_scheduler._LRScheduler,
                               viz: bool = False) -> Tuple[float, torch.Tensor]:
        """
        Train the online autoencoder with the current batch of features.
        
        This function trains the online autoencoder to adapt to the current
        environment, improving the compression of language features.
        
        Args:
            features: Input features to train on
            online_auto_optimizer: Optimizer for online autoencoder
            online_auto_scheduler: Learning rate scheduler
            viz: Whether to visualize the results
            
        Returns:
            Tuple of (loss value, compressed features)
        """
        # Move features to device and detach from computation graph
        features = features.to(self.device).detach()
        
        # Set model to training mode
        self.online_auto.train()
        
        # Zero gradients
        online_auto_optimizer.zero_grad()
        
        # Forward pass
        comp_15 = self.online_auto.encode(features)
        recon_coco = self.online_auto.decode(comp_15)
        
        # Compute loss
        l1_loss_val = F.l1_loss(recon_coco, features)
        cos_loss = 1 - F.cosine_similarity(recon_coco, features, dim=1).mean()
        loss = l1_loss_val + 0.6 * cos_loss
        
        # Backward pass and optimization
        loss.backward()
        online_auto_optimizer.step()
        
        # Visualize if requested
        if viz:
            with torch.no_grad():
                self.online_auto.eval()
                comp_15 = self.online_auto.encode(features)
                recon_32 = self.online_auto.decode(comp_15)
                
                # Get reconstructions from both autoencoders
                recon_online = self.auto_model.decode(recon_32)
                recon_coco = self.auto_model.decode(features)
                
                # Visualize similarity maps
                self.visualize_similarity(recon_online, recon_coco)
        
        # Return loss value and compressed features
        return loss.item(), comp_15.cpu().detach()

    def initialize_map(self, cur_frame_idx: int, viewpoint: Any) -> Dict[str, Any]:
        """
        Initialize the map with the first keyframe.
        
        Args:
            cur_frame_idx: Index of the current frame
            viewpoint: Camera viewpoint for the current frame
            
        Returns:
            Rendering package from the last iteration
        """
        Log("Beginning to initialize map")
        
        # Initialize online autoencoder optimizer if using two-stage approach
        if not self.is_single_stage:
            online_auto_optimizer = optim.Adam(self.online_auto.parameters(), lr=1e-3)
            online_auto_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                online_auto_optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True
            )
            
        Log(f"Initializing from viewpoint {cur_frame_idx}")
        
        # Perform initial optimization iterations
        for mapping_iteration in tqdm(range(self.init_itr_num), desc="Mapping Iteration"):
            self.iteration_count += 1

            # Render the scene
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            
            # Extract render package components
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            depth = render_pkg["depth"]
            opacity = render_pkg["opacity"]
            n_touched = render_pkg["n_touched"]

            # Process language features if needed
            if self.config["language"]["language_train"] and not self.is_single_stage:
                height, width = image.shape[1], image.shape[2]
                
                # Extract language features
                with torch.no_grad():
                    inputs = [{
                        "image": (viewpoint.original_image*255.0).cuda(), 
                        "height": height, "width": width
                    }]
                    
                    if viewpoint.gt_lang_feat is None:
                        # Get language features using external model
                        clip_viz_dense = get_lang_feat(inputs, self.lang_model, is_lang=False)
                        
                        # Process through high-resolution model if configured
                        if self.config["language"]["hr_model"]:
                            high_res_lang = self.hr_model(
                                clip_viz_dense['clip_vis_dense'], 
                                clip_viz_dense['res3'], 
                                clip_viz_dense['res2']
                            )
                            clip_viz_dense = high_res_lang
                        else: 
                            clip_viz_dense = clip_viz_dense["clip_vis_dense"]
                        
                        # Reshape and encode using autoencoder
                        N, C, H, W = clip_viz_dense.shape
                        batch_reshape = clip_viz_dense.permute(0, 2, 3, 1)
                        high_res_lang_res = batch_reshape.view(-1, 768)
                        low_dim_coco = self.auto_model.encode(high_res_lang_res)
                        low_dim_coco = low_dim_coco.cpu().detach()
                
                # Train online autoencoder periodically
                if (mapping_iteration % 5 == 0):
                    t1 = time.time()
                    # NOTE: This is outside torch.no_grad() intentionally since we're training
                    loss, _ = self.train_online_autoencoder(
                        low_dim_coco, 
                        online_auto_optimizer,
                        online_auto_scheduler, 
                        viz=(mapping_iteration % 300 == 0)
                    )
                    Log(f"Online training time: {time.time() - t1:.3f}s")

            # Compute mapping loss and backward
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            # Update Gaussians with torch.no_grad()
            with torch.no_grad():
                # Update maximum 2D radii
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                
                # Add densification statistics
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                
                # Densify and prune Gaussians periodically
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                # Reset opacity at specific iterations
                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                # Step optimizer
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        # Store visibility information
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Map initialization completed")
        
        return render_pkg
    
    def map(self, 
           current_window: List[int], 
           prune: bool = False, 
           iters: int = 1, 
           lang_run: bool = False,
           is_key: bool = False) -> bool:
        """
        Perform mapping for the current keyframe window.
        
        This function performs optimization of the Gaussian model and
        camera poses for the current window of keyframes.
        
        Args:
            current_window: List of keyframe indices in the current window
            prune: Whether to prune Gaussians
            iters: Number of iterations to perform
            lang_run: Whether to process language features
            is_key: Whether this is a keyframe mapping (unused, kept for compatibility)
            
        Returns:
            Boolean indicating whether Gaussians were split/modified
        """
        if len(current_window) == 0:
            return False
            
        # Initialize online autoencoder optimizer for two-stage approach
        if not self.is_single_stage:
            online_auto_optimizer = optim.Adam(self.online_auto.parameters(), lr=1e-4)
            online_auto_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                online_auto_optimizer, mode='min', factor=0.5, patience=10, threshold=1e-5, verbose=True
            )

        # Prepare viewpoints
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        
        # Get random viewpoints outside current window
        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        # Mapping iterations
        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            # Initialize accumulators
            loss_mapping = 0
            viewspace_point_tensor_acm, visibility_filter_acm = [], []
            radii_acm, n_touched_acm = [], []
            keyframes_opt = []

            # Process each keyframe in the window
            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                
                # Render the scene
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )

                # Extract render package components
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                depth = render_pkg["depth"]
                opacity = render_pkg["opacity"]
                n_touched = render_pkg["n_touched"]

                # Language supervision block
                Ll1_feat = 0
                if self.config["language"]["language_train"] and lang_run:
                    height, width = image.shape[1], image.shape[2]

                    if not self.config["language"]["labels_from_file"]:
                        # Process language features if not already available
                        if viewpoint.gt_lang_feat is None:
                            # Extract language features (no gradients needed)
                            with torch.no_grad():
                                inputs = [{
                                    "image": (viewpoint.original_image * 255.0).cuda(),
                                    "height": height, "width": width
                                }]
                                
                                # Get language features using external model
                                clip_viz_dense = get_lang_feat(inputs, self.lang_model, is_lang=False)

                                # Process through high-resolution model if configured
                                if self.config["language"]["hr_model"]:
                                    clip_viz_dense = self.hr_model(
                                        clip_viz_dense["clip_vis_dense"],
                                        clip_viz_dense["res3"],
                                        clip_viz_dense["res2"]
                                    )
                                else:
                                    clip_viz_dense = clip_viz_dense["clip_vis_dense"]

                                # Reshape and encode using autoencoder
                                N, C, H, W = clip_viz_dense.shape
                                flattened = clip_viz_dense.permute(0, 2, 3, 1).view(-1, 768)
                                low_dim = self.auto_model.encode(flattened)
                            
                            # Train online autoencoder if using two-stage approach
                            if not self.is_single_stage:
                                # Store features for training
                                viewpoint.coco_lang_feat = low_dim.cpu().detach()
                                
                                # Train online autoencoder (outside torch.no_grad())
                                _, low_dim = self.train_online_autoencoder(
                                    viewpoint.coco_lang_feat,
                                    online_auto_optimizer,
                                    online_auto_scheduler,
                                    viz=(self.iteration_count % 3 == 0)
                                )

                            # Store language features
                            code_num = self.config["language"]["lang_code_size"]
                            viewpoint.gt_lang_feat = low_dim.T.view(code_num, 192, 192).cpu().detach()

                    # Resize ground truth features to match rendered size
                    gt_lang_feat_resize = F.interpolate(
                        viewpoint.gt_lang_feat.unsqueeze(0),
                        size=(height, width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                    # Compute L1 loss between predicted and ground truth features
                    language_feat = render_pkg["language"]
                    Ll1_feat = l1_loss(language_feat, gt_lang_feat_resize.to("cuda"))

                # Compute mapping loss
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )

                # Add language loss if enabled
                if self.config["language"]["language_train"] and lang_run:
                    loss_mapping += self.lamda_lang * Ll1_feat

                # Accumulate tensors
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            # Process random viewpoints for regularization
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                
                # Render the scene
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                
                # Extract render package components
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                depth = render_pkg["depth"]
                opacity = render_pkg["opacity"]
                n_touched = render_pkg["n_touched"]

                # Process language features if needed
                if self.config["language"]["language_train"] and lang_run:
                    language_feat = render_pkg["language"]
                    Ll1_feat = 0
                    height, width = image.shape[1], image.shape[2]
 
                    # Resize ground truth features
                    gt_lang_feat_resize = F.interpolate(
                        viewpoint.gt_lang_feat.unsqueeze(0), 
                        size=(height, width), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    
                    # Compute L1 loss
                    Ll1_feat = l1_loss(language_feat, gt_lang_feat_resize.to("cuda"))
                    
                    # Train online autoencoder to prevent forgetting
                    if not self.is_single_stage:
                        # NOTE: This is outside torch.no_grad() intentionally
                        self.train_online_autoencoder(
                            viewpoint.coco_lang_feat, 
                            online_auto_optimizer, 
                            online_auto_scheduler,
                            viz=(self.iteration_count % 3 == 0)
                        )
                    
                # Compute mapping loss
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )

                # Add language loss if enabled
                if self.config["language"]["language_train"] and lang_run:
                    loss_mapping += self.lamda_lang * Ll1_feat

                # Accumulate tensors
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            # Add isotropic loss to encourage more uniform Gaussians
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            # Backward pass
            loss_mapping.backward()

            # Update Gaussians with torch.no_grad()
            gaussian_split = False
            with torch.no_grad():
                # Update visibility information
                self.occ_aware_visibility = {}
                for idx in range(len(current_window)):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # Prune Gaussians if requested
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        
                        # Count observations for each Gaussian
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                            
                        # Determine which Gaussians to prune
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                        elif prune_mode == "slam":
                            # Only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                            
                        # Perform pruning
                        if to_prune is not None:
                            self.gaussians.prune_points(to_prune.cuda())
                            
                            # Update visibility information
                            for idx in range(len(current_window)):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                                
                    # Return early after pruning
                    return False
                
                # Update maximum 2D radii for each viewpoint
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                # Densify and prune Gaussians periodically
                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    Log("Densifying and pruning Gaussians")
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                # Reset opacity periodically
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                # Step optimizers
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                
                # Update camera poses if not using ground truth
                if self.config["Training"]["use_gt_pose"] is not True:
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
                        
        return gaussian_split

    def color_refinement(self):
        """
        Perform final color refinement on the Gaussian model.
        This improves the visual quality of the reconstruction.
        """
        Log("Starting color refinement")

        # Perform many iterations of color refinement
        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1), desc="Color Refinement"):
            # Select a random viewpoint
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            
            # Render the scene
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            
            # Extract render package components
            image = render_pkg["render"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            # Compute color refinement loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            
            # Backward pass
            loss.backward()
            
            # Update Gaussians
            with torch.no_grad():
                # Update maximum 2D radii
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                
                # Step optimizer
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
                
        Log("Map refinement completed")

    def push_to_frontend(self, tag: Optional[str] = None):
        """
        Send the current state to the frontend.
        
        Args:
            tag: Message tag to identify the type of update
        """
        self.last_sent = 0
        
        # Prepare keyframe information
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
            
        # Set default tag if none provided
        if tag is None:
            tag = "sync_backend"

        # Send message to frontend
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        """
        Main loop for the backend process.
        
        This method:
        1. Processes messages from the frontend
        2. Performs mapping and optimization
        3. Sends updates back to the frontend
        """
        while True:
            # Check for messages from frontend
            if self.backend_queue.empty():
                # Skip if paused or no keyframes
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                # Skip in single-thread mode
                if self.single_thread:
                    time.sleep(0.01)
                    continue
                    
                # Perform mapping
                self.map(self.current_window)
                
                # Periodically perform language mapping and send updates
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10, lang_run=True)
                    self.push_to_frontend()
            else:
                # Process message from frontend
                data = self.backend_queue.get()
                
                if data[0] == "stop":
                    # Exit loop
                    break
                elif data[0] == "pause":
                    # Pause processing
                    self.pause = True
                elif data[0] == "unpause":
                    # Resume processing
                    self.pause = False
                elif data[0] == "color_refinement":
                    # Perform color refinement and save models
                    if not self.is_single_stage:
                        # Save online autoencoder weights
                        torch.save(
                            self.online_auto.state_dict(), 
                            self.config["language"]["online_ckpt_path"]
                        )
                        Log("Saved online autoencoder weights")
                    self.push_to_frontend()
                elif data[0] == "init":
                    # Initialize the map with the first keyframe
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    
                    Log("Resetting the system")
                    self.reset()

                    # Add first keyframe
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    
                    # Initialize the map
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    # Process a new keyframe
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    # Store keyframe information
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    
                    # Add keyframe to map
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    # Setup optimizers for keyframes
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    
                    # Determine iterations per keyframe
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    
                    # Set up optimization parameters for each keyframe
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                            
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        
                        # Add rotation and translation parameters if within optimization window
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"] * 0.5,
                                    "name": f"rot_{viewpoint.uid}",
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"]["cam_trans_delta"] * 0.5,
                                    "name": f"trans_{viewpoint.uid}",
                                }
                            )
                            
                        # Add exposure parameters for all keyframes
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": f"exposure_a_{viewpoint.uid}",
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": f"exposure_b_{viewpoint.uid}",
                            }
                        )
                        
                    # Initialize keyframe optimizer
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    
                    Log(f"Mapping keyframe {cur_frame_idx}")
                    
                    # Perform mapping
                    self.map(self.current_window, iters=iter_per_kf, lang_run=True)
                    self.map(self.current_window, prune=True)
                    
                    # Update counter and send update to frontend
                    self.map_counter += 1
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception(f"Unprocessed data type: {data[0]}")
                    
        # Clean up queues before exiting
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()