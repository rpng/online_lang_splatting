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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	void language_preprocess(
		int num_gaussians,
		int D,
		int M,
		int language_M,
		const float* orig_points,
		const glm::vec3* scales,
		const glm::vec3* scales_lang,
		const float scale_modifier,
		const glm::vec4* rotations,
		const glm::vec4* rotations_lang,
		const float* opacities,
		const float* opacities_lang,
		const float* shs,
		//const float* language_shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* cov3D_precomp_lang,
		const float* colors_precomp,
		const float* language_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		int* radii_lang,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* cov3Ds_lang,
		float* colors,
		float* language,
		float4* conic_opacity,
		float4* conic_opacity_lang,
		const dim3 grid,
		uint32_t* tiles_touched,
		uint32_t* tiles_touched_lang,
		bool prefiltered);
	
	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		const float* depth,
	    float* out_depth,
		float* out_opacity,
		int* n_touched);
	
	void language_render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint2* ranges_lang,
		const uint32_t* point_list,
		const uint32_t* point_list_lang,
		int W, int H,
		const float2* means2D,
		const float* colors,
		const float* language,
		const float4* conic_opacity,
		const float4* conic_opacity_lang,
		float* final_T,
		float* final_T_lang,
		uint32_t* n_contrib,
		uint32_t* n_contrib_lang,
		const float* bg_color,
		//const float* bg_language,
		float* out_color,
		float* out_language,
		const float* depth,
		float* out_depth,
		float* out_opacity,
		float* out_opacity_lang,
		int* n_touched,
		int* n_touched_lang);
}


#endif