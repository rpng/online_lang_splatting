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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}


CudaRasterizer::LanguageGeometryState CudaRasterizer::LanguageGeometryState::fromChunk(char*& chunk, size_t P) {
	LanguageGeometryState language_geom;
	obtain(chunk, language_geom.depths, P, 128);
	obtain(chunk, language_geom.clamped, P * 3, 128);
	obtain(chunk, language_geom.internal_radii, P, 128);
	obtain(chunk, language_geom.internal_radii_lang, P, 128);
	obtain(chunk, language_geom.means2D, P, 128);
	obtain(chunk, language_geom.cov3D, P * 6, 128);
	obtain(chunk, language_geom.cov3D_lang, P * 6, 128);
	obtain(chunk, language_geom.conic_opacity, P, 128);
	obtain(chunk, language_geom.conic_opacity_lang, P, 128);
	obtain(chunk, language_geom.rgb, P * 3, 128);
	obtain(chunk, language_geom.language, P * NUM_LANGUAGE_CHANNELS, 128);
	obtain(chunk, language_geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, language_geom.scan_size, language_geom.tiles_touched, language_geom.tiles_touched, P);
	obtain(chunk, language_geom.tiles_touched_lang, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, language_geom.scan_size, language_geom.tiles_touched_lang, language_geom.tiles_touched_lang, P);
	obtain(chunk, language_geom.scanning_space, language_geom.scan_size, 128);
	obtain(chunk, language_geom.point_offsets, P, 128);
	obtain(chunk, language_geom.point_offsets_lang, P, 128);
	return language_geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::LanguageImageState CudaRasterizer::LanguageImageState::fromChunk(char*& chunk, size_t N)
{
	LanguageImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.accum_alpha_lang, N, 128);
	obtain(chunk, img.n_contrib_lang, N, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.ranges_lang, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	float* out_opacity,
	int* radii,
	int* n_touched,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_COLOR_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		out_depth, 
		out_opacity,
		n_touched
    ), debug)

	return num_rendered;
}

std::tuple<int, int>
CudaRasterizer::LanguageRasterizer::forward(
	std::function<char*(size_t)> language_geometry_buffer,
	std::function<char*(size_t)> binning_buffer,
	std::function<char*(size_t)> binning_buffer_lang,
	std::function<char*(size_t)> image_buffer,
	const int P, int D, int M,
	int language_M,
	const float* background_color,
	//const float* background_language,
	const int width, int height,
	const float* means3D,
	const float* shs,
	//const float* language_shs,
	const float* colors_precomp,
	const float* language_precomp,
	const float* opacities,
	const float* opacities_lang,
	const float* scales,
	const float* scales_lang,
	const float scale_modifier,
	const float* rotations,
	const float* rotations_lang,
	const float* cov3D_precomp,
	const float* cov3D_precomp_lang,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_language,
	float* out_depth,
	float* out_opacity,
	float* out_opacity_lang,
	int* radii,
	int* radii_lang,
	int* n_touched,
	int* n_touched_lang,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<LanguageGeometryState>(P);
	char* chunkptr = language_geometry_buffer(chunk_size);
	LanguageGeometryState language_geom_state = 
		LanguageGeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr) {
		radii = language_geom_state.internal_radii;
	}
	// TODO here copy one internal radii as the radii_lang
	if (radii_lang == nullptr) {
		radii_lang = language_geom_state.internal_radii_lang;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<LanguageImageState>(width * height);
	char* img_chunkptr = image_buffer(img_chunk_size);
	LanguageImageState imgState = LanguageImageState::fromChunk(img_chunkptr, width * height);
	
	if (NUM_COLOR_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	CHECK_CUDA(FORWARD::language_preprocess(
		P, D, M,
		language_M,
		means3D,
		(glm::vec3*)scales,
		(glm::vec3*)scales_lang,
		scale_modifier,
		(glm::vec4*)rotations,
		(glm::vec4*)rotations_lang,
		opacities,
		opacities_lang,
		shs,
		//language_shs,
		language_geom_state.clamped,
		cov3D_precomp,
		cov3D_precomp_lang,
		colors_precomp,
		language_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		radii_lang,
		language_geom_state.means2D,
		language_geom_state.depths,
		language_geom_state.cov3D,
		language_geom_state.cov3D_lang,
		language_geom_state.rgb,
		language_geom_state.language,
		language_geom_state.conic_opacity,
		language_geom_state.conic_opacity_lang,
		tile_grid,
		language_geom_state.tiles_touched,
		language_geom_state.tiles_touched_lang,
		prefiltered),
		debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(language_geom_state.scanning_space, language_geom_state.scan_size, language_geom_state.tiles_touched, language_geom_state.point_offsets, P), debug)
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(language_geom_state.scanning_space, language_geom_state.scan_size, language_geom_state.tiles_touched_lang, language_geom_state.point_offsets_lang, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	int num_rendered_lang;
	CHECK_CUDA(cudaMemcpy(&num_rendered, language_geom_state.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	CHECK_CUDA(cudaMemcpy(&num_rendered_lang, language_geom_state.point_offsets_lang + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	size_t binning_chunk_size_lang = required<BinningState>(num_rendered_lang);
	char* binning_chunkptr = binning_buffer(binning_chunk_size);
	char* binning_chunkptr_lang = binning_buffer_lang(binning_chunk_size_lang);
	BinningState binning_state =
		BinningState::fromChunk(binning_chunkptr, num_rendered);
	BinningState binning_state_lang =
		BinningState::fromChunk(binning_chunkptr_lang, num_rendered_lang);
	
	// For each instance to be rendered, produce adequate [ tile | depth ] key
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		language_geom_state.means2D,
		language_geom_state.depths,
		language_geom_state.point_offsets,
		binning_state.point_list_keys_unsorted,
		binning_state.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		language_geom_state.means2D,
		language_geom_state.depths,
		language_geom_state.point_offsets_lang,
		binning_state_lang.point_list_keys_unsorted,
		binning_state_lang.point_list_unsorted,
		radii_lang,
		tile_grid)
	CHECK_CUDA(, debug)
	
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binning_state.list_sorting_space,
		binning_state.sorting_size,
		binning_state.point_list_keys_unsorted, binning_state.point_list_keys,
		binning_state.point_list_unsorted, binning_state.point_list,
		num_rendered, 0, 32 + bit), debug)
	
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binning_state_lang.list_sorting_space,
		binning_state_lang.sorting_size,
		binning_state_lang.point_list_keys_unsorted, binning_state_lang.point_list_keys,
		binning_state_lang.point_list_unsorted, binning_state_lang.point_list,
		num_rendered_lang, 0, 32 + bit), debug)
	
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
	CHECK_CUDA(cudaMemset(imgState.ranges_lang, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binning_state.point_list_keys,
			imgState.ranges);
	
	if (num_rendered_lang > 0)
		identifyTileRanges << <(num_rendered_lang + 255) / 256, 256 >> > (
			num_rendered_lang,
			binning_state_lang.point_list_keys,
			imgState.ranges_lang);
	
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr 
									? colors_precomp 
									: language_geom_state.rgb;

	const float* language_feature_ptr = language_precomp != nullptr 
											? language_precomp 
											: language_geom_state.language;
	
	CHECK_CUDA(FORWARD::language_render(
		tile_grid, 
		block,
		imgState.ranges,
		imgState.ranges_lang,
		binning_state.point_list,
		binning_state_lang.point_list,
		width, height,
		language_geom_state.means2D,
		feature_ptr,
		language_feature_ptr,
		language_geom_state.conic_opacity,
		language_geom_state.conic_opacity_lang,
		imgState.accum_alpha,
		imgState.accum_alpha_lang,
		imgState.n_contrib,
		imgState.n_contrib_lang,
		background_color,
		//background_language,
		out_color,
		out_language,
		language_geom_state.depths,
		out_depth,
		out_opacity,
		out_opacity_lang,
		n_touched,
		n_touched_lang), debug)
	
	return std::make_tuple(num_rendered, num_rendered_lang);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
    const float* projmatrix_raw,
    const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_dpix_depth,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dtau,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    const float* depth_ptr = geomState.depths;

	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		depth_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_depth,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth
    ), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
        projmatrix_raw,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		dL_dtau), debug)
}

void CudaRasterizer::LanguageRasterizer::backward(const int P,
	int D, int M, int language_M, int R, int R_lang,
	const float* background_color,
	//const float* background_language,
	const int width, int height,
	const float* means3D,
	const float* shs,
	//const float* language_shs,
	const float* colors_precomp,
	const float* language_precomp,
	const float* scales,
	const float* scales_lang,
	const float scale_modifier,
	const float* rotations,
	const float* rotations_lang,
	const float* cov3D_precomp,
	const float* cov3D_precomp_lang,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_raw,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	const int* radii_lang,
	char* language_geomentry_buffer,
	char* binning_buffer,
	char* binning_buffer_lang,
	char* image_buffer,
	const float* dL_dpix,
	const float* dL_dpix_language,
	const float* dL_dpix_depth,
	float* dL_dmean2D,
	//float* dL_dmean2D_lang,
	float* dL_dconic,
	float* dL_dconic_lang,
	float* dL_dopacity,
	float* dL_dopacity_lang,
	float* dL_dcolor,
	float* dL_dlanguage,
	float* dL_ddepths,
	float* dL_dmean3D,
	//float* dL_dmean3D_lang,
	float* dL_dcov3D,
	float* dL_dcov3D_lang,
	float* dL_dsh,
	//float* dL_dlanguage_sh,
	float* dL_dscale,
	float* dL_dscale_lang,
	float* dL_drot,
	float* dL_drot_lang,
	float* dL_dtau,
	//float* dL_dtau_lang,
	bool debug) 
{
	
	LanguageGeometryState langauge_geometry_state = 
		LanguageGeometryState::fromChunk(language_geomentry_buffer, P);
	BinningState binning_state = BinningState::fromChunk(binning_buffer, R);
	BinningState binning_state_lang = BinningState::fromChunk(binning_buffer_lang, R_lang);
	LanguageImageState image_state = LanguageImageState::fromChunk(image_buffer, width * height);

	if (radii == nullptr) {
		radii = langauge_geometry_state.internal_radii;
	}
	if (radii_lang == nullptr) {
		radii_lang = langauge_geometry_state.internal_radii_lang;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : langauge_geometry_state.rgb;
	const float* language_ptr = (language_precomp != nullptr) ? language_precomp : langauge_geometry_state.language;
	const float* depth_ptr = langauge_geometry_state.depths;

	CHECK_CUDA(BACKWARD::language_render(
		tile_grid,
		block,
		image_state.ranges,
		image_state.ranges_lang,
		binning_state.point_list,
		binning_state_lang.point_list,
		width, height,
		background_color,
		//background_language,
		langauge_geometry_state.means2D,
		langauge_geometry_state.conic_opacity,
		langauge_geometry_state.conic_opacity_lang,
		color_ptr,
		language_ptr,
		depth_ptr,
		image_state.accum_alpha,
		image_state.accum_alpha_lang,
		image_state.n_contrib,
		image_state.n_contrib_lang,
		dL_dpix,
		dL_dpix_language,
		dL_dpix_depth,
		(float3*)dL_dmean2D,
		//(float3*)dL_dmean2D_lang,
		(float4*)dL_dconic,
		(float4*)dL_dconic_lang,
		dL_dopacity,
		dL_dopacity_lang,
		dL_dcolor,
		dL_dlanguage,
		dL_ddepths
	), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : langauge_geometry_state.cov3D;
	const float* cov3D_ptr_lang = (cov3D_precomp_lang != nullptr) ? cov3D_precomp_lang : langauge_geometry_state.cov3D_lang;

	CHECK_CUDA(BACKWARD::language_preprocess(
		P, D, M,
		language_M,
		(float3*)means3D,
		radii,
		radii_lang,
		shs,
		//language_shs,
		langauge_geometry_state.clamped,
		(glm::vec3*)scales,
		(glm::vec3*)scales_lang,
		(glm::vec4*)rotations,
		(glm::vec4*)rotations_lang,
		scale_modifier,
		cov3D_ptr,
		cov3D_ptr_lang,
		viewmatrix,
		projmatrix,
		projmatrix_raw,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		//(float3*)dL_dmean2D_lang,
		dL_dconic,
		dL_dconic_lang,
		(glm::vec3*)dL_dmean3D,
		//(glm::vec3*)dL_dmean3D_lang,
		dL_dcolor,
		dL_dlanguage,
		dL_ddepths,
		dL_dcov3D,
		dL_dcov3D_lang,
		dL_dsh,
		//dL_dlanguage_sh,
		(glm::vec3*)dL_dscale,
		(glm::vec3*)dL_dscale_lang,
		(glm::vec4*)dL_drot,
		(glm::vec4*)dL_drot_lang,
		dL_dtau
		//dL_dtau_lang
	), debug)
}