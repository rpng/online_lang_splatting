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

#include "forward.h"
#include "auxiliary.h"
#include "helper_math.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Perform initial steps for each Gaussian prior to rasterization
template<int COLOR_CHANNELS, int LANGUAGE_CHANNELS>
__global__ void languagePreprocessCUDA(int num_gaussians,
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
	const int W, const int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	int* radii_lang,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* cov3Ds_lang,
	float* rgb,
	float* language,
	float4* conic_opacity,
	float4* conic_opacity_lang,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint32_t* tiles_touched_lang,
	bool prefiltered) 
{
	// RIGHT now same as preprocessCUDA
	auto idx = cg::this_grid().thread_rank();
	if (idx >= num_gaussians)
		return;
							
	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	radii_lang[idx] = 0;
	tiles_touched[idx] = 0;
	tiles_touched_lang[idx] = 0;
							
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
							
	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], 
		orig_points[3 * idx + 1], 
		orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
							
	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	const float* cov3D_lang;
	if (cov3D_precomp_lang != nullptr)
	{
		cov3D_lang = cov3D_precomp_lang + idx * 6;
	}
	else
	{
		computeCov3D(scales_lang[idx], scale_modifier, rotations_lang[idx], cov3Ds_lang + idx * 6);
		cov3D_lang = cov3Ds_lang + idx * 6;
	}
							
	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
	float3 cov_lang = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D_lang, viewmatrix);
							
	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	bool det_condition = (det == 0.0f);
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
	
	float det_lang = (cov_lang.x * cov_lang.z - cov_lang.y * cov_lang.y);
	bool det_condition_lang = (det_lang == 0.0f);
	if (det_condition && det_condition_lang) return;

	float det_inv_lang = 1.f / det_lang;
	float3 conic_lang = { cov_lang.z * det_inv_lang, -cov_lang.y * det_inv_lang, cov_lang.x * det_inv_lang };
							
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	bool zero_condition = ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0);

	float mid_lang = 0.5f * (cov_lang.x + cov_lang.z);
	float lambda1_lang = mid_lang + sqrt(max(0.1f, mid_lang * mid_lang - det_lang));
	float lambda2_lang = mid_lang - sqrt(max(0.1f, mid_lang * mid_lang - det_lang));
	float my_radius_lang = ceil(3.f * sqrt(max(lambda1_lang, lambda2_lang)));
	uint2 rect_min_lang, rect_max_lang;
	getRect(point_image, my_radius_lang, rect_min_lang, rect_max_lang, grid);
	bool zero_condition_lang = ((rect_max_lang.x - rect_min_lang.x) * (rect_max_lang.y - rect_min_lang.y) == 0);
							
	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (zero_condition && zero_condition_lang)
	{
		return;
	}

	if (colors_precomp == nullptr)
	{
		if (zero_condition)
		{
			rgb[idx * COLOR_CHANNELS + 0] = 0;
			rgb[idx * COLOR_CHANNELS + 1] = 0;
			rgb[idx * COLOR_CHANNELS + 2] = 0;
		}
		else
		{
			glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
			rgb[idx * COLOR_CHANNELS + 0] = result.x;
			rgb[idx * COLOR_CHANNELS + 1] = result.y;
			rgb[idx * COLOR_CHANNELS + 2] = result.z;
		}		
		
	}

	// Store some useful helper data for the next steps.

	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;

	
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	

	radii_lang[idx] = my_radius_lang;
	conic_opacity_lang[idx] = { conic_lang.x, conic_lang.y, conic_lang.z, opacities_lang[idx] };
	tiles_touched_lang[idx] = (rect_max_lang.y - rect_min_lang.y) * (rect_max_lang.x - rect_min_lang.x);
	
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <int COLOR_CHANNELS, int LANGUAGE_CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
language_renderCUDA(const uint2* __restrict__ ranges,
					const uint2* __restrict__ ranges_lang,
					const uint32_t* __restrict__ point_list,
					const uint32_t* __restrict__ point_list_lang,
					int W, int H,
					const float2* __restrict__ points_xy_image,
					const float* __restrict__ features,
					const float* __restrict__ language_features,
					const float4* __restrict__ conic_opacity,
					const float4* __restrict__ conic_opacity_lang,
					float* __restrict__ final_T,
					float* __restrict__ final_T_lang,
					uint32_t* __restrict__ n_contrib,
					uint32_t* __restrict__ n_contrib_lang,
					const float* __restrict__ bg_color,
					float* __restrict__ out_color,
					float* __restrict__ out_language_feature,
					const float* __restrict__ depth,
					float* __restrict__ out_depth,
					float* __restrict__ out_opacity,
					float* __restrict__ out_opacity_lang,
					int* __restrict__ n_touched,
					int* __restrict__ n_touched_lang) {
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;
	bool done_lang = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	uint2 range_lang = ranges_lang[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	const int rounds_lang = ((range_lang.y - range_lang.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	int toDo_lang = range_lang.y - range_lang.x;

	// Allocate storage for batches of collectively fetched data.
	// [TODO]: Duplicate the collected id or not
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ int collected_id_lang[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float2 collected_xy_lang[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity_lang[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	float T_lang = 1.0f;
	uint32_t contributor = 0;
	uint32_t contributor_lang = 0;
	uint32_t last_contributor = 0;
	uint32_t last_contributor_lang = 0;
	float C[COLOR_CHANNELS] = { 0 };
	float L[LANGUAGE_CHANNELS] = { 0 };
	float D = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
		
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;
			
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) {
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * COLOR_CHANNELS + ch] * alpha * T;
			}
			D += collected_depth[j] * alpha * T;
			// Keep track of how many pixels touched this Gaussian.
			if (test_T > 0.5f) {
				atomicAdd(&(n_touched[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	for (int i = 0; i < rounds_lang; i++, toDo_lang -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done_lang = __syncthreads_count(done_lang);
		if (num_done_lang == BLOCK_SIZE)
			break;
		
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range_lang.x + progress < range_lang.y)
		{
			int coll_id_lang = point_list_lang[range_lang.x + progress];
			collected_id_lang[block.thread_rank()] = coll_id_lang;
			collected_xy_lang[block.thread_rank()] = points_xy_image[coll_id_lang];
			collected_conic_opacity_lang[block.thread_rank()] = conic_opacity_lang[coll_id_lang];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done_lang && j < min(BLOCK_SIZE, toDo_lang); j++)
		{
			// Keep track of current position in range
			contributor_lang++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy_lang[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o_lang = collected_conic_opacity_lang[j];
			float power_lang = -0.5f * (con_o_lang.x * d.x * d.x + con_o_lang.z * d.y * d.y) - con_o_lang.y * d.x * d.y;
			if (power_lang > 0.0f)
				continue;
			
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha_lang = min(0.99f, con_o_lang.w * exp(power_lang));
			if (alpha_lang < 1.0f / 255.0f) {
				continue;
			}
			float test_T_lang = T_lang * (1 - alpha_lang);
			if (test_T_lang < 0.0001f)
			{
				done_lang = true;
				continue;
			}
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < LANGUAGE_CHANNELS; ch++) {
				L[ch] += language_features[collected_id_lang[j] * LANGUAGE_CHANNELS + ch] * alpha_lang * T_lang;
			}
			// Keep track of how many pixels touched this Gaussian.
			if (test_T_lang > 0.5f) {
				atomicAdd(&(n_touched_lang[collected_id_lang[j]]), 1);
			}
			T_lang = test_T_lang;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor_lang = contributor_lang;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		final_T_lang[pix_id] = T_lang;
		n_contrib[pix_id] = last_contributor;
		n_contrib_lang[pix_id] = last_contributor_lang;
		for (int ch = 0; ch < COLOR_CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		for (int ch = 0; ch < LANGUAGE_CHANNELS; ch++) {
			out_language_feature[ch * H * W + pix_id] = L[ch] ;//+ T * bg_language[ch]; 
			//TODO: L[ch] or L[ch] + T * bg_language[ch];
		}
		out_depth[pix_id] = D;
		out_opacity[pix_id] = 1 - T;
		out_opacity_lang[pix_id] = 1 - T_lang;
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	float* __restrict__ out_depth, 
	float* __restrict__ out_opacity,
	int * __restrict__ n_touched)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// uint32_t horizontal_blocks = gridDim.x; # TODO Maybe it's different?
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) {
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}
			D += collected_depth[j] * alpha * T;
			// Keep track of how many pixels touched this Gaussian.
			if (test_T > 0.5f) {
				atomicAdd(&(n_touched[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_depth[pix_id] = D;
		out_opacity[pix_id] = 1 - T;
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depth,
	float* out_depth, 
	float* out_opacity,
	int* n_touched)
{
	renderCUDA<NUM_COLOR_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth,
		out_depth,
		out_opacity,
		n_touched);
}

void FORWARD::language_render(
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
	float* out_color,
	float* out_language,
	const float* depth,
	float* out_depth,
	float* out_opacity,
	float* out_opacity_lang,
	int* n_touched,
	int* n_touched_lang) 
{
	language_renderCUDA<NUM_COLOR_CHANNELS, NUM_LANGUAGE_CHANNELS> << <grid, block >> > (
		ranges,
		ranges_lang,
		point_list,
		point_list_lang,
		W, H,
		means2D,
		colors,
		language,
		conic_opacity,
		conic_opacity_lang,
		final_T,
		final_T_lang,
		n_contrib,
		n_contrib_lang,
		bg_color,
		out_color,
		out_language,
		depth,
		out_depth,
		out_opacity,
		out_opacity_lang,
		n_touched,
		n_touched_lang);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_COLOR_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

void FORWARD::language_preprocess(
	int num_gaussians,
	int D,
	int M,
	int language_M,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* cov3Ds_lang,
	float* rgb,
	float* language,
	float4* conic_opacity,
	float4* conic_opacity_lang,
	const dim3 grid,
	uint32_t* tiles_touched,
	uint32_t* tiles_touched_lang,
	bool prefiltered)
{
	languagePreprocessCUDA<NUM_COLOR_CHANNELS, NUM_LANGUAGE_CHANNELS>
	<<<(num_gaussians + 255) / 256, 256>>>(num_gaussians,
											D, 
											M, 
											language_M,
											means3D,
											scales,
											scales_lang,
											scale_modifier,
											rotations,
											rotations_lang,
											opacities,
											opacities_lang,
											shs,
											//language_shs,
											clamped,
											cov3D_precomp,
											cov3D_precomp_lang,
											colors_precomp,
											language_precomp,
											viewmatrix,
											projmatrix,
											cam_pos,
											W, H,
											tan_fovx, tan_fovy,
											focal_x, focal_y,
											radii,
											radii_lang,
											means2D,
											depths,
											cov3Ds,
											cov3Ds_lang,
											rgb,
											language,
											conic_opacity,
											conic_opacity_lang,
											grid,
											tiles_touched,
											tiles_touched_lang,
											prefiltered);
}