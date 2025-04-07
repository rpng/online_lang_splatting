"""
RGB-D SLAM Frontend Module

This module implements the frontend component of a Gaussian Splatting SLAM system for RGB-D sensors.
The frontend handles frame processing, camera tracking, and keyframe selection.

Main responsibilities:
- Tracking camera poses for each new frame
- Selecting keyframes based on visibility and motion criteria
- Managing the sliding window of active keyframes
- Communicating with the backend for mapping and optimization
"""

import time
from typing import Dict, List, Tuple, Optional, Set, Any, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.optim import Adam

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth


class FrontEnd(mp.Process):
    """
    SLAM Frontend class responsible for tracking and keyframe selection.
    
    The frontend tracks the camera pose for each new frame and determines when to
    create new keyframes based on tracking quality and scene coverage. It maintains
    a sliding window of active keyframes and communicates with the backend for
    mapping and optimization.
    """
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.config = config
        
        # These will be set later
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None
        self.dataset = None
        
        # State tracking
        self.initialized = False
        self.kf_indices = []
        self.monocular = False  # Using RGB-D only
        self.use_gt_pose = config["Training"]["use_gt_pose"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        self.median_depth = 1.0  # Default median depth
        
        # Control flags
        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1
        self.pause = False
        
        # System components
        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        
        # These will be set in set_hyperparams
        self.save_dir = None
        self.save_results = None
        self.save_trj = None
        self.save_trj_kf_intv = None
        self.tracking_itr_num = None
        self.kf_interval = None
        self.window_size = None
        self.single_thread = None

    def set_hyperparams(self):
        """
        Set hyperparameters from the configuration dictionary.
        This should be called after initializing the frontend.
        """
        # Result saving parameters
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        # Tracking and mapping parameters
        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx: int, depth: Optional[torch.Tensor] = None, 
                        opacity: Optional[torch.Tensor] = None, init: bool = False) -> np.ndarray:
        """
        Add a new keyframe to the system.
        
        Args:
            cur_frame_idx: Index of the current frame to add as keyframe
            depth: Optional depth map from rendering
            opacity: Optional opacity map from rendering
            init: Whether this is the initial keyframe
            
        Returns:
            Depth map for the new keyframe as numpy array
        """
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        
        # Find valid RGB regions
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        # Use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx: int, viewpoint: Camera):
        """
        Initialize the SLAM system with the first frame.
        
        Args:
            cur_frame_idx: Index of the current frame
            viewpoint: Camera object for the current frame
        """
        # Set initialized state (always True for RGB-D)
        self.initialized = True
        
        # Reset state
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        
        # Clear backend queue
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialize the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        # Add first keyframe
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx: int, viewpoint: Camera) -> Dict[str, Any]:
        """
        Perform camera tracking for the current frame.
        
        This function optimizes the camera pose by minimizing photometric error
        between the rendered image and the observed image.
        
        Args:
            cur_frame_idx: Index of the current frame
            viewpoint: Camera object for the current frame
            
        Returns:
            Rendering package with rendered image, depth, etc.
        """
        # Initialize pose from previous frame
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        viewpoint.update_RT(prev.R, prev.T)

        # Setup optimizer parameters
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": f"rot_{viewpoint.uid}",
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": f"trans_{viewpoint.uid}",
            }
        )
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

        # Create optimizer
        pose_optimizer = Adam(opt_params)
        
        # Iterative tracking optimization
        for tracking_itr in range(self.tracking_itr_num):
            # Render the scene from current pose estimate
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            
            # Compute tracking loss and update pose
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                
                # Use ground truth pose if enabled
                if self.use_gt_pose:
                    viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                    converged = (tracking_itr == 50)
                else:
                    converged = update_pose(viewpoint)

            # Periodically update visualization
            if tracking_itr % 10 == 0:
                # Handle language features if available
                if self.gaussians.is_language:
                    gtlang = None
                    if self.config["language"]["labels_from_file"]:
                        gtlang = viewpoint.gt_lang_feat.to("cuda")
                    
                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            current_frame=viewpoint,
                            gtcolor=viewpoint.original_image,
                            gtdepth=viewpoint.depth,
                            gtlangauge=gtlang
                        )
                    )
                
                # Regular visualization update
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth,
                        gtlangauge=None
                    )
                )
                
            # Exit early if converged
            if converged:
                break

        # Update median depth for keyframe selection
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg
    
    def is_keyframe(self, cur_frame_idx: int, last_keyframe_idx: int, 
                  cur_frame_visibility_filter: torch.Tensor, 
                  occ_aware_visibility: Dict[int, torch.Tensor]) -> bool:
        """
        Determine if the current frame should be a keyframe.
        
        This function uses several criteria:
        1. Translation distance from the last keyframe
        2. Point visibility overlap between current frame and last keyframe
        
        Args:
            cur_frame_idx: Index of the current frame
            last_keyframe_idx: Index of the last keyframe
            cur_frame_visibility_filter: Visibility mask for current frame
            occ_aware_visibility: Dict mapping frame indices to visibility masks
            
        Returns:
            Boolean indicating whether the current frame should be a keyframe
        """
        # Get keyframe selection parameters
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        # Get camera poses
        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        
        # Compute relative pose
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        
        # Compute translation distance
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        
        # Distance-based checks
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        # Visibility overlap check
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        
        point_ratio_2 = intersection / union
        
        # Return true if either condition is met
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(self, cur_frame_idx: int, cur_frame_visibility_filter: torch.Tensor, 
                     occ_aware_visibility: Dict[int, torch.Tensor], 
                     window: List[int]) -> Tuple[List[int], Optional[int]]:
        """
        Add the current frame to the keyframe window and manage window size.
        
        This function:
        1. Adds the current frame to the window
        2. Removes frames with little overlap with the current frame
        3. Ensures the window size doesn't exceed the maximum
        
        Args:
            cur_frame_idx: Index of the current frame
            cur_frame_visibility_filter: Visibility mask for current frame
            occ_aware_visibility: Dict mapping frame indices to visibility masks
            window: Current keyframe window
            
        Returns:
            Tuple of (updated window, removed frame index or None)
        """
        # Don't touch the first N frames in the window
        N_dont_touch = 2
        
        # Add current frame to the front of the window
        window = [cur_frame_idx] + window
        
        # Get current frame camera
        curr_frame = self.cameras[cur_frame_idx]
        
        # Find frames with little overlap to potentially remove
        to_remove = []
        removed_frame = None
        
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            
            # Calculate overlap
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            
            point_ratio_2 = intersection / denom
            
            # Get cutoff threshold
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            
            # If overlap is below threshold, mark for removal
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        # Remove a frame with low overlap if any found
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
            
        # Compute inverse of current frame pose
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        # If window is too large, remove a frame
        if len(window) > self.config["Training"]["window_size"]:
            # Find the keyframe to remove based on relative poses
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                
                # Compute distances to other keyframes
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                
                # Compute distance to current frame
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            # Remove frame with maximum inverse distance score
            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx: int, viewpoint: Camera, 
                        current_window: List[int], depthmap: np.ndarray):
        """
        Request the backend to process a new keyframe.
        
        Args:
            cur_frame_idx: Index of the current frame
            viewpoint: Camera object for the current frame
            current_window: Current keyframe window
            depthmap: Depth map for the current frame
        """
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx: int, viewpoint: Camera):
        """
        Request the backend to perform mapping.
        
        Args:
            cur_frame_idx: Index of the current frame
            viewpoint: Camera object for the current frame
        """
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx: int, viewpoint: Camera, depth_map: np.ndarray):
        """
        Request the backend to initialize the map.
        
        Args:
            cur_frame_idx: Index of the current frame
            viewpoint: Camera object for the current frame
            depth_map: Depth map for the current frame
        """
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data: List[Any]):
        """
        Synchronize state with the backend.
        
        Args:
            data: Data from backend containing gaussians, visibility, and keyframes
        """
        # Unpack data from backend
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        # Update keyframe poses
        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx: int):
        """
        Clean up resources for a frame.
        
        Args:
            cur_frame_idx: Index of the frame to clean up
        """
        # Clean camera resources
        self.cameras[cur_frame_idx].clean()
        
        # Periodically clear CUDA cache
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        """
        Main loop for the frontend process.
        
        This method:
        1. Processes frames sequentially
        2. Tracks camera poses
        3. Selects keyframes
        4. Communicates with the backend and visualization
        """
        # Initialize variables
        cur_frame_idx = 0
        
        # Create projection matrix
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        
        # Setup timing events
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        # Main processing loop
        while True:
            # Check for pause/unpause commands from GUI
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            # Process frame if no messages from backend
            if self.frontend_queue.empty():
                tic.record()
                
                # Check if all frames have been processed
                if cur_frame_idx >= len(self.dataset):
                    # Final evaluation if saving results
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=False,  # RGB-D only
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                # Wait states
                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                # Initialize camera for current frame
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)
                self.cameras[cur_frame_idx] = viewpoint

                # Initialize the map if needed
                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                # Update initialization status
                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Perform tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                # Prepare data for visualization
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                # Send data to visualization
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                # Skip to next frame if already waiting for backend
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                # Check keyframe criteria
                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                
                # Determine if a new keyframe should be created
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                
                # Special case for small window
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                    
                # Apply additional constraints for single thread mode
                if self.single_thread:
                    create_kf = check_time and create_kf
                    
                # Process new keyframe if needed
                if create_kf:
                    # Update keyframe window
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    
                    # Create new keyframe
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    
                    # Send keyframe to backend
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    # Clean up without creating keyframe
                    self.cleanup(cur_frame_idx)
                    
                # Move to next frame
                cur_frame_idx += 1

                # Perform trajectory evaluation if needed
                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=False,  # RGB-D only
                    )
                    
                # Record frame processing time
                toc.record()
                torch.cuda.synchronize()
                
                # Rate limiting for keyframes
                if create_kf:
                    # Throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
                    
            # Handle messages from backend
            else:
                data = self.frontend_queue.get()
                
                if data[0] == "sync_backend":
                    # Synchronize with backend
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    # Process keyframe response
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    # Process initialization response
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    # Exit loop on stop command
                    Log("Frontend Stopped.")
                    break
