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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
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
			int* radii = nullptr,
			int* n_touched = nullptr,
			bool debug = false);

		static void backward(
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
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dpix_depth,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_ddepths,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dtau,
			bool debug);
	};
	class LanguageRasterizer 
	{
	public:
		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static std::tuple<int, int> forward(
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
			int* radii = nullptr,
			int* radii_lang = nullptr,
			int* n_touched = nullptr,
			int* n_touched_lang = nullptr,
			bool debug = false);
		
		static void backward(const int P,
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
							bool debug);
	};

}; // namespace CudaRasterizer

#endif