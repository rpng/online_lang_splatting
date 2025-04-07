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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    )

def rasterize_language_gaussians(
    means3D,
    means2D,
    sh,
    #language_sh,
    colors_precomp,
    language_precomp,
    opacities,
    opacities_lang,
    scales,
    scales_lang,
    rotations,
    rotations_lang,
    cov3Ds_precomp,
    cov3Ds_precomp_lang,
    theta,
    rho,
    raster_settings,
):
    return _RasterizeLanguageGaussians.apply(
        means3D,
        means2D,
        sh,
        #language_sh,
        colors_precomp,
        language_precomp,
        opacities,
        opacities_lang,
        scales,
        scales_lang,
        rotations,
        rotations_lang,
        cov3Ds_precomp,
        cov3Ds_precomp_lang,
        theta,
        rho,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, n_touched = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, depth, opacity, n_touched = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth, opacity, n_touched

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_opacity, grad_n_touched):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.projmatrix_raw,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_depth,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_tau = _C.rasterize_gaussians_backward(*args)
        
        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)


        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,
        )

        return grads


class _RasterizeLanguageGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        #language_sh,
        colors_precomp,
        language_precomp,
        opacities,
        opacities_lang,
        scales,
        scales_lang,
        rotations,
        rotations_lang,
        cov3Ds_precomp,
        cov3Ds_precomp_lang,
        theta,
        rho,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            #raster_settings.bg,
            means3D,
            colors_precomp,
            language_precomp,
            opacities,
            opacities_lang,
            scales,
            scales_lang,
            rotations,
            rotations_lang,
            raster_settings.scale_modifier, #float
            cov3Ds_precomp,
            cov3Ds_precomp_lang,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx, #float
            raster_settings.tanfovy, #float
            raster_settings.image_height, #int
            raster_settings.image_width, #int
            sh,
            raster_settings.sh_degree, #int
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # print("raster_settings.bg: ", raster_settings.bg)
        # print("means3D: ", means3D.shape)
        # print("colors_precomp: ", colors_precomp.shape)
        # print("language_precomp: ", language_precomp.shape)
        # print("opacities: ", opacities.shape)
        # print("scales: ", scales.shape)
        # print(":raster_settings.scale_modifier: ", raster_settings.scale_modifier)
        # print("cov3Ds_precomp: ", cov3Ds_precomp.shape)
        # print("raster_settings.viewmatrix: ", raster_settings.viewmatrix.shape)
        # print("raster_settings.projmatrix: ", raster_settings.projmatrix.shape)
        # print("raster_settings.projmatrix_raw: ", raster_settings.projmatrix_raw.shape)
        # print("raster_settings.tanfovx: ", raster_settings.tanfovx)
        # print("raster_settings.tanfovy: ", raster_settings.tanfovy)
        # print("raster_settings.image_height: ", raster_settings.image_height)
        # print("raster_settings.image_width: ", raster_settings.image_width)
        # print("sh: ", sh.shape)
        # print("raster_settings.sh_degree: ", raster_settings.sh_degree)
        # print("raster_settings.campos: ", raster_settings.campos.shape)
        # print("raster_settings.prefiltered: ", raster_settings.prefiltered)
        # print("raster_settings.debug: ", raster_settings.debug)
        

        # # Invoke C++/CUDA rasterizer
        # if raster_settings.debug:
        #     cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        #     try:
        #         num_rendered, color, language, radii, language_geometry_buffer, binning_buffer, image_buffer, depth, opacity, n_touched = _C.rasterize_language_gaussians(*args)
        #     except Exception as ex:
        #         torch.save(cpu_args, "snapshot_fw.dump")
        #         print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
        #         raise ex
        # else:
            #print("language_sh", language_sh)
        num_rendered, num_rendered_lang, color, language, radii, radii_lang, language_geometry_buffer, binning_buffer, binning_buffer_lang, image_buffer, depth, opacity, opacity_lang, n_touched, n_touched_lang = _C.rasterize_language_gaussians(*args)
        # print("d",color.shape)
        # print("w",color.max())
        # print("r",color.min())
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_rendered_lang = num_rendered_lang
        ctx.save_for_backward(colors_precomp, language_precomp, means3D, scales, scales_lang, 
                              rotations, rotations_lang, cov3Ds_precomp, cov3Ds_precomp_lang, radii, radii_lang, sh, #language_sh, 
                              language_geometry_buffer, binning_buffer, binning_buffer_lang, image_buffer)
        return color, language, radii, radii_lang, depth, opacity, opacity_lang, n_touched, n_touched_lang

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_language, grad_out_radii, grad_out_radii_lang, grad_out_depth, grad_out_opacity, grad_out_opacity_lang, grad_n_touched, grad_n_touched_lang):

        # print('-----=--')
        # print(grad_out_color)
        # print(grad_out_language)
        # print(grad_out_radii)
        # print(grad_out_radii_lang)
        # print(grad_out_depth)
        # print(grad_out_opacity)
        # print(grad_out_opacity_lang)
        # print(grad_n_touched, grad_n_touched_lang)
        # print('============')
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_rendered_lang = ctx.num_rendered_lang
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            language_precomp,
            means3D,
            scales,
            scales_lang,
            rotations,
            rotations_lang,
            cov3Ds_precomp,
            cov3Ds_precomp_lang,
            radii,
            radii_lang,
            sh,
            #language_sh,
            language_geometry_buffer,
            binning_buffer,
            binning_buffer_lang,
            image_buffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            #raster_settings.bg,
            means3D,
            radii,
            radii_lang,
            colors_precomp,
            language_precomp,
            scales,
            scales_lang,
            rotations,
            rotations_lang,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            cov3Ds_precomp_lang,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_language,
            grad_out_depth,
            sh,
            #language_sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            language_geometry_buffer,
            num_rendered,
            num_rendered_lang,
            binning_buffer,
            binning_buffer_lang,
            image_buffer,
            raster_settings.debug,
        )
        # print("E",scales_lang.shape)
        # print("E",scales_lang.grad)
        # print("E",scales_lang.min())
        # print("E",scales_lang.max())

        # print("F",rotations_lang.shape)
        # print("E",rotations_lang.grad)
        # print("F",rotations_lang.min())
        # print("F",rotations_lang.max())

        (
            grad_means2D,
            #grad_means2D_lang,
            grad_colors_precomp,
            grad_language_precomp,
            grad_opacities,
            grad_opacities_lang,
            grad_means3D,
            #grad_means3D_lang,
            grad_cov3Ds_precomp,
            grad_cov3Ds_precomp_lang,
            grad_sh,
            #grad_language_sh,
            grad_scales,
            grad_scales_lang,
            grad_rotations,
            grad_rotations_lang,
            grad_tau,
        ) = _C.rasterize_language_gaussians_backward(*args)

        #grad_means2D += grad_means2D_lang
        #grad_means3D += grad_means3D_lang
        
        # print("a",grad_colors_precomp.shape)
        # print("a",grad_colors_precomp.min())
        # print("a",grad_colors_precomp.max())

        # print("b",grad_language_precomp.shape)
        # print("b",grad_language_precomp.min())
        # print("b",grad_language_precomp.max())

        #print("c",grad_opacities.shape)
        # print("c",grad_opacities.min())
        # print("c",grad_opacities.max())
        
        # print("d",grad_scales_lang.shape)
        # print("d",grad_scales_lang.min())
        # print("d",grad_scales_lang.max())

        #print("y",grad_cov3Ds_precomp.shape)
        # print("y",grad_cov3Ds_precomp.min())
        # print("y",grad_cov3Ds_precomp.max())

        # print("p",grad_rotations.shape)
        # print("p",grad_rotations.min())
        # print("p",grad_rotations.max())

        # print("ii",grad_means3D.shape)
        # print("ii",grad_means3D.min())
        # print("ii",grad_means3D.max())
        
        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            #grad_language_sh,
            grad_colors_precomp,
            grad_language_precomp,
            grad_opacities,
            grad_opacities_lang,
            grad_scales,
            grad_scales_lang,
            grad_rotations,
            grad_rotations_lang,
            grad_cov3Ds_precomp,
            grad_cov3Ds_precomp_lang,
            grad_theta,
            grad_rho,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    #background_language: torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    projmatrix_raw : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, theta=None, rho=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])
        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings, 
        )

class LanguageGaussianRasterizer(nn.Module):
    def __init__(
            self,
            raster_settings: GaussianRasterizationSettings,
    ):
        super().__init__()
        self.raster_settings = raster_settings
    
    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible
    
    def forward(
        self,
        means3D,
        means2D,
        opacities,
        opacities_lang,
        shs=None,
        #language_shs=None,
        colors_precomp=None,
        language_precomp=None,
        scales=None,
        scales_lang=None,
        rotations=None,
        rotations_lang=None,
        cov3D_precomp=None,
        cov3D_precomp_lang=None,
        theta=None,
        rho=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception("Please provide excatly one of either SHs or precomputed colors!")
        # if (language_shs is None and language_precomp is None) or (
        #     language_shs is not None and language_precomp is not None
        # ):
        #     raise Exception("Please provide excatly one of either SHs or precomputed colors!")
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )
        if ((scales_lang is None or rotations_lang is None) and cov3D_precomp_lang is None) or (
            (scales_lang is not None or rotations_lang is not None) and cov3D_precomp_lang is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance for language!"
            )
        
        if shs is None:
            shs = torch.Tensor([])
        # if language_shs is None:
        #     language_shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if language_precomp is None:
            language_precomp = torch.Tensor([])
        
        if scales is None:
            scales = torch.Tensor([])
        if scales_lang is None:
            scales_lang = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if rotations_lang is None:
            rotations_lang = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if cov3D_precomp_lang is None:
            cov3D_precomp_lang = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])
        
        # Invoke C++/CUDA rasterization routine
        colors, language, radii, radii_lang, depth, opacity, opacity_lang, n_touched, n_touched_lang = (
            rasterize_language_gaussians(
                means3D,
                means2D,
                shs,
                #language_shs,
                colors_precomp,
                language_precomp,
                opacities,
                opacities_lang,
                scales,
                scales_lang,
                rotations,
                rotations_lang,
                cov3D_precomp,
                cov3D_precomp_lang,
                theta,
                rho,
                raster_settings,
            )
        )

        return (
            colors,
            language,
            radii,
            radii_lang,
            depth,
            opacity,
            opacity_lang,
            n_touched,
            n_touched_lang
        )
