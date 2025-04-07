#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    """
    A class representing a 3D Gaussian model for rendering.
    """
    def __init__(self, sh_degree: int, config=None):
        """
        Initialize a Gaussian model.
        
        Args:
            sh_degree: Maximum spherical harmonics degree
            config: Configuration dictionary
        """
        # Spherical harmonics configuration
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # Initialize model parameters
        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self._language_feature = torch.empty(0, device="cuda")
        
        # Tracking
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")
        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()
        self.denom = torch.empty(0, device="cuda")

        # Activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.covariance_activation = self.build_covariance_from_scaling_rotation

        # Setup configuration
        self.config = config
        self.is_language = False
        if config and "language" in config and "language_train" in config["language"]:
            self.is_language = config["language"]["language_train"]
        
        self.ply_input = None
        self.optimizer = None
        self.isotropic = False

    # Property accessors
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_language_features(self):
        if self._language_feature is not None:
            return self._language_feature
        raise ValueError('No language feature')
        
    def get_covariance(self, scaling_modifier=1):
        """Calculate the covariance matrix from scaling and rotation."""
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        """Build covariance matrix from scaling and rotation."""
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def oneupSHdegree(self):
        """Increase the spherical harmonics degree if below maximum."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def init_lr(self, spatial_lr_scale):
        """Initialize learning rate scaling."""
        self.spatial_lr_scale = spatial_lr_scale

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        """
        Create a point cloud from an image and depth data.
        
        Args:
            cam_info: Camera information
            init: Whether this is initial creation
            scale: Scale factor for depth
            depthmap: Optional pre-computed depth map
            
        Returns:
            Point cloud data
        """
        cam = cam_info
        # Process image
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        
        # Process depth
        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                # Create synthetic depth for monocular sensors
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))
        
        # Create point cloud
        if self.is_language:
            return self.create_pcd_from_image_and_depth(cam, rgb, None, depth, init)
        
        return self.create_pcd_from_image_and_depth(cam, rgb, None, depth, init)

    def create_pcd_from_image_and_depth(self, cam, rgb, language_map, depth, init=False):
        """
        Create a point cloud from RGB and depth images.
        
        Args:
            cam: Camera information
            rgb: RGB image
            language_map: Optional language feature map
            depth: Depth image
            init: Whether this is initial creation
            
        Returns:
            Point cloud components
        """
        # Set downsampling factor based on initialization or not
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
            
        # Get point size from config
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
                
        # Create RGBD image from color and depth
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        # Transform to world space
        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        
        # Downsample point cloud
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        # Create basic point cloud
        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        # Convert to tensors
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        
        # Initialize features
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # Compute scaling
        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        # Initialize rotations and opacities
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        
        # Return components based on whether language features are used
        if self.is_language:
            return fused_point_cloud, features, None, scales, rots, opacities
        else:
            return fused_point_cloud, features, scales, rots, opacities

    def extend_from_pcd(
        self, fused_point_cloud, features, language_features, scales, rots, opacities, kf_id
    ):
        """
        Extend the model with new points from a point cloud.
        
        Args:
            fused_point_cloud: Point positions
            features: Point appearance features
            language_features: Optional language features
            scales: Point scale factors
            rots: Point rotations
            opacities: Point opacities
            kf_id: Keyframe ID
        """
        # Create parameters from tensors
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        
        # Initialize language features if needed
        if self.is_language:
            language_features = torch.zeros(
                (new_xyz.shape[0], self.config["language"]["lang_code_size"]), 
                device="cuda"
            )
            new_language_features = nn.Parameter(
                language_features.contiguous().requires_grad_(True)
            )
            
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        # Create tracking tensors for new points
        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        
        # Add new points to the model
        if self.is_language:
            self.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                new_language_features,
                new_opacity,
                new_scaling,
                new_rotation,
                new_kf_ids=new_unique_kfIDs,
                new_n_obs=new_n_obs,
            )
        else:
            self.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                None,
                new_opacity,
                new_scaling,
                new_rotation,
                new_kf_ids=new_unique_kfIDs,
                new_n_obs=new_n_obs,
            )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        """
        Extend model from a new point cloud based on camera input.
        
        Args:
            cam_info: Camera information
            kf_id: Keyframe ID
            init: Whether this is initial creation
            scale: Scale factor for depth
            depthmap: Optional pre-computed depth map
        """
        if self.is_language:
            fused_point_cloud, features, language_features, scales, rots, opacities = (
                self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
            )
            self.extend_from_pcd(
                fused_point_cloud, features, language_features, 
                scales, rots, opacities, kf_id
            )
        else:
            fused_point_cloud, features, scales, rots, opacities = (
                self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
            )
            self.extend_from_pcd(
                fused_point_cloud, features, None, 
                scales, rots, opacities, kf_id
            )

    def training_setup(self, training_args):
        """
        Set up the model for training.
        
        Args:
            training_args: Training arguments
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Create parameter groups for optimizer
        param_groups = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
        
        # Add language feature parameters if needed
        if self.is_language:
            lang_code_size = self.config["language"]["lang_code_size"]
            language_features = torch.zeros((self._xyz.shape[0], lang_code_size), device="cuda")
            self._language_feature = nn.Parameter(language_features.requires_grad_(True))
            param_groups.append(
                {
                    "params": [self._language_feature],
                    "lr": training_args.language_lr,
                    "name": "f_language",
                }
            )

        # Create optimizer
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        
        # Set up learning rate scheduler
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        # Store scheduler parameters
        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """
        Update learning rate based on iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Current learning rate
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        """
        Construct a list of attribute names for PLY export.
        
        Returns:
            List of attribute names
        """
        attr_list = ["x", "y", "z", "nx", "ny", "nz"]
        
        # Add DC feature attributes
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attr_list.append(f"f_dc_{i}")
            
        # Add rest feature attributes
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attr_list.append(f"f_rest_{i}")
            
        # Add language feature attributes if present
        if self.is_language:
            for i in range(self._language_feature.shape[1]):
                attr_list.append(f"f_language_{i}")
                
        # Add remaining attributes
        attr_list.append("opacity")
        for i in range(self._scaling.shape[1]):
            attr_list.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            attr_list.append(f"rot_{i}")
            
        return attr_list

    def save_ply(self, path):
        """
        Save the model to a PLY file.
        
        Args:
            path: Output file path
        """
        mkdir_p(os.path.dirname(path))

        # Collect all tensor data
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # Create attribute array
        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        # Combine all attributes
        if self.is_language:
            f_language = self._language_feature.detach().cpu().numpy()
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, f_language, opacities, scale, rotation),
                axis=1,
            )
        else:
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), 
                axis=1
            )
            
        # Create PLY file
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        """Reset all opacities to a low value."""
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(self, visibility_filters):
        """
        Reset opacity for only non-visible gaussians.
        
        Args:
            visibility_filters: List of visibility filters
        """
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self._opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """
        Load model from a PLY file.
        
        Args:
            path: Input file path
        """
        plydata = PlyData.read(path)

        # Helper function to load point cloud without color
        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        
        # Load position data
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # Load feature data
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # Find and load rest features
        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        # Reshape features
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        # Load scale and rotation data
        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Create model parameters
        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        
        # Initialize tracking tensors
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        Replace a tensor in the optimizer.
        
        Args:
            tensor: New tensor
            name: Parameter name
            
        Returns:
            Dictionary of updated tensors
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # Preserve optimizer state
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # Replace parameter
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        Prune optimizer parameters based on mask.
        
        Args:
            mask: Boolean mask for points to keep
            
        Returns:
            Dictionary of updated tensors
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # Update optimizer state
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # Replace parameter
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # Just update parameter without optimizer state
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        Remove points based on mask.
        
        Args:
            mask: Boolean mask for points to remove
        """
        # Invert mask to get points to keep
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # Update model parameters
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if self.is_language:
            self._language_feature = optimizable_tensors["f_language"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Update tracking tensors
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_language_features,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        if self.is_language:
            d.update({"f_language": new_language_features})

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if self.is_language:
            self._language_feature = optimizable_tensors["f_language"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        
        if self.is_language:
            new_language_features = self._language_feature[selected_pts_mask].repeat(N, 1)
        else:
            new_language_features = None
        
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_language_features,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        if self.is_language:
            new_language_features = self._language_feature[selected_pts_mask]
        else:
            new_language_features = None
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_language_features,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
