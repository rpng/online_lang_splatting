/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
    const torch::Tensor& projmatrix_raw,
    const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
    const bool prefiltered,
	const bool debug);

std::tuple<int, int,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor,
			torch::Tensor>
RasterizeLanguageGaussiansCUDA(const torch::Tensor& background_color,
                               //const torch::Tensor& background_language,
                               const torch::Tensor& means3D,
                               const torch::Tensor& colors,
                               const torch::Tensor& language,
                               const torch::Tensor& opacity,
							   const torch::Tensor& opacity_lang,
                               const torch::Tensor& scales,
							   const torch::Tensor& scales_lang,
                               const torch::Tensor& rotations,
                               const torch::Tensor& rotations_lang,
                               const float scale_modifier,
                               const torch::Tensor& cov3D_precomp,
                               const torch::Tensor& cov3D_precomp_lang,
							   const torch::Tensor& viewmatrix,
                               const torch::Tensor& projmatrix,
                               const torch::Tensor& projmatrix_raw,
                               const float tan_fovx,
                               const float tan_fovy,
                               const int image_height,
                               const int image_width,
                               const torch::Tensor& sh,
                               //const torch::Tensor& language_sh,
                               const int degree,
                               const torch::Tensor& campos,
                               const bool prefiltered,
                               const bool debug);


std::tuple<torch::Tensor, 
			torch::Tensor, 
			torch::Tensor, 
			torch::Tensor, 
			torch::Tensor, 
			torch::Tensor, 
			torch::Tensor, 
			torch::Tensor, 
			torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& projmatrix_raw,
    const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           //torch::Tensor,
           torch::Tensor,
		   torch::Tensor,
		   torch::Tensor,
		   torch::Tensor,
		   torch::Tensor,
		   //torch::Tensor,
		   //torch::Tensor,
           torch::Tensor>
RasterizeLanguageGaussiansBackwardCUDA(
	const torch::Tensor& background_color,
	//const torch::Tensor& background_language,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& radii_lang,
	const torch::Tensor& colors,
	const torch::Tensor& language,
	const torch::Tensor& scales,
	const torch::Tensor& scales_lang,
	const torch::Tensor& rotations,
	const torch::Tensor& rotations_lang,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& cov3D_precomp_lang,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& projmatrix_raw,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_language,
	const torch::Tensor& dL_dout_depth,
	const torch::Tensor& sh,
	//const torch::Tensor& language_sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& language_geometry_buffer,
	const int R,
	const int R_lang,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& binningBuffer_lang,
	const torch::Tensor& imageBuffer,
	const bool debug);

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);