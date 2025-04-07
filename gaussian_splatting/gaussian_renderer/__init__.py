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

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    LanguageGaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh
from utils.camera_utils import Camera

def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if pc.is_language:
        return _language_render(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            scaling_modifier,
            override_color,
            mask,
        )
    else:
        return _render(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            scaling_modifier,
            override_color,
            mask,
        )

def _render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Handle empty model case
    if pc.get_xyz.shape[0] == 0:
        return None

    # Create zero tensor for gradient computation in screen space
    screenspace_points = torch.zeros_like(
        pc.get_xyz, 
        dtype=pc.get_xyz.dtype, 
        requires_grad=True, 
        device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    
    # Set up camera parameters for rasterization
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Configure rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    # Create rasterizer with settings
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Get model properties
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Prepare scales, rotations and covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # Handle isotropic vs anisotropic scaling
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # Prepare colors from spherical harmonics or precomputed values
    shs = None
    colors_precomp = None
    
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            # Precompute colors from spherical harmonics
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Apply mask if provided
    if mask is not None:
        rendered_image, radii, depth, opacity = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask] if shs is not None else None,
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask] if scales is not None else None,
            rotations=rotations[mask] if rotations is not None else None,
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
        n_touched = None
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Return rendering results with the same structure as the original code
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }

def _language_render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render language-conditioned Gaussians.
    
    Background tensor (bg_color) must be on GPU!
    """
    # Handle empty model case
    if pc.get_xyz.shape[0] == 0:
        return None
    
    # Create zero tensor for gradient computation in screen space
    screenspace_points = torch.zeros_like(
        pc.get_xyz, 
        dtype=pc.get_xyz.dtype, 
        requires_grad=True, 
        device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up camera parameters for rasterization
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Configure rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    # Create language rasterizer with settings
    rasterizer = LanguageGaussianRasterizer(raster_settings=raster_settings)

    # Get model properties
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Prepare scales, rotations and covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # Handle isotropic vs anisotropic scaling
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # Prepare colors from spherical harmonics or precomputed values
    shs = None
    colors_precomp = None
    
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            # Precompute colors from spherical harmonics
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # Get language features
    language_precomp = pc.get_language_features
    
    # Apply mask if provided
    if mask is not None:
        (
            rendered_image,
            language,
            radii,
            depth,
            opacity,
        ) = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask] if shs is not None else None,
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            language_precomp=language_precomp[mask] if language_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask] if scales is not None else None,
            rotations=rotations[mask] if rotations is not None else None,
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
        n_touched = None
    else:
        (
            rendered_image,
            language,
            radii,
            depth,
            opacity,
            n_touched,
        ) = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            language_precomp=language_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Return rendering results with the same structure as the original code
    return {
        "render": rendered_image,
        "language": language,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }
