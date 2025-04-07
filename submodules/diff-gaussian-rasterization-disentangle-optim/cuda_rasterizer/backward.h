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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dpixels_depth,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_ddepths);
	
	void language_render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint2* ranges_lang,
		const uint32_t* point_list,
		const uint32_t* point_list_lang,
		int W, int H,
		const float* bg_color,
		//const float* bg_language,
		const float2* means2D,
		const float4* conic_opacity,
		const float4* conic_opacity_lang,
		const float* colors,
		const float* language,
		const float* depths,
		const float* final_Ts,
		const float* final_Ts_lang,
		const uint32_t* n_contrib,
		const uint32_t* n_contrib_lang,
		const float* dL_dpixels,
		const float* dL_dpixels_language,
		const float* dL_dpixels_depth,
		float3* dL_dmean2D,
		//float3* dL_dmean2D_lang,
		float4* dL_dconic2D,
		float4* dL_dconic2D_lang,
		float* dL_dopacity,
		float* dL_dopacity_lang,
		float* dL_dcolors,
		float* dL_dlanguage,
		float* dL_ddepths);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float* proj_raw,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_ddepth,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		float* dL_dtau);
	
	void language_preprocess(
		int P, int D, int M,
		int language_M,
		const float3* means,
		const int* radii,
		const int* radii_lang,
		const float* shs,
		//const float* language_shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec3* scales_lang,
		const glm::vec4* rotations,
		const glm::vec4* rotations_lang,
		const float scale_modifier,
		const float* cov3Ds,
		const float* cov3Ds_lang,
		const float* view_matrix,
		const float* projection_matrix,
		const float* projection_matirx_raw,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		//const float3* dL_dmean2D_lang,
		const float* dL_dconics,
		const float* dL_dconics_lang,
		glm::vec3* dL_dmeans,
		//glm::vec3* dL_dmeans_lang,
		float* dL_dcolor,
		float* dL_dlanguage,
		float* dL_ddepth,
		float* dL_dcov3D,
		float* dL_dcov3D_lang,
		float* dL_dsh,
		//float* dL_dlanguage_sh,
		glm::vec3* dL_dscale,
		glm::vec3* dL_dscale_lang,
		glm::vec4* dL_drot,
		glm::vec4* dL_drot_lang,
		float* dL_dtau
		//float* dL_dtau_lang
		);
}

#endif