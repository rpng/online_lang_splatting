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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_COLOR_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor n_touched = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_opaticy = torch::full({1, H, W}, 0.0, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_depth.contiguous().data<float>(),
		out_opaticy.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		n_touched.contiguous().data<int>(),
        debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_depth, out_opaticy, n_touched);
}

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
							const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	
	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_COLOR_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor out_language_feature = torch::full({NUM_LANGUAGE_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor radii_lang = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor n_touched = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor n_touched_lang = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor out_opaticy = torch::full({1, H, W}, 0.0, float_opts);
	torch::Tensor out_opaticy_lang = torch::full({1, H, W}, 0.0, float_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor language_geometry_buffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer_lang = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> language_geometry_func = resizeFunctional(language_geometry_buffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> binningFunc_lang = resizeFunctional(binningBuffer_lang);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int num_rendered = 0;
	int num_rendered_lang = 0;
	if (P != 0) {
		int M = 0;
		if (sh.size(0) != 0) {
			M = sh.size(1);
		}
		int language_M = 0;

		std::tuple<int, int> result = CudaRasterizer::LanguageRasterizer::forward(
			language_geometry_func,
			binningFunc,
			binningFunc_lang,
			imgFunc,
			P, degree, M, language_M,
			background_color.contiguous().data<float>(),
			//background_language.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			//language_sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(),
			language.contiguous().data<float>(),
			opacity.contiguous().data<float>(),
			opacity_lang.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scales_lang.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			rotations_lang.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			cov3D_precomp_lang.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_language_feature.contiguous().data<float>(),
			out_depth.contiguous().data<float>(),
			out_opaticy.contiguous().data<float>(),
			out_opaticy_lang.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			radii_lang.contiguous().data<int>(),
			n_touched.contiguous().data<int>(),
			n_touched_lang.contiguous().data<int>(),
			debug);
		
		num_rendered = std::get<0>(result);
		num_rendered_lang = std::get<1>(result);
	}
	return std::make_tuple(num_rendered,
		num_rendered_lang,
		out_color,
		out_language_feature,
		radii,
		radii_lang,
		language_geometry_buffer,
		binningBuffer,
		binningBuffer_lang,
		imgBuffer,
		out_depth,
		out_opaticy,
		out_opaticy_lang,
		n_touched,
		n_touched_lang);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
	const torch::Tensor& dL_dout_depths,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_COLOR_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dtau = torch::zeros({P,6}, means3D.options());

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
      projmatrix_raw.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_depths.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_ddepths.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
      dL_dtau.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dtau);
}

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
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0) {
		M = sh.size(1);
	}
	int language_M = 0;

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	//torch::Tensor dL_dmeans3D_lang = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	//torch::Tensor dL_dmeans2D_lang = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_COLOR_CHANNELS}, means3D.options());
	torch::Tensor dL_dlanguage = torch::zeros({P, NUM_LANGUAGE_CHANNELS}, means3D.options());
	torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dconic_lang = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dopacity_lang = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dcov3D_lang = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	//torch::Tensor dL_dlanguage_sh = torch::zeros({P, language_M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dscales_lang = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_drotations_lang = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dtau = torch::zeros({P, 6}, means3D.options());
	//torch::Tensor dL_dtau_lang = torch::zeros({P, 6}, means3D.options());

	if (P != 0) {
		CudaRasterizer::LanguageRasterizer::backward(
			P, degree, M, language_M, R, R_lang,
			background_color.contiguous().data<float>(),
			//background_language.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data<float>(),
			//language_sh.contiguous().data<float>(),
			colors.contiguous().data<float>(),
			language.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			scales_lang.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			rotations_lang.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(),
			cov3D_precomp_lang.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			projmatrix_raw.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			radii.contiguous().data_ptr<int>(),
			radii_lang.contiguous().data_ptr<int>(),
			reinterpret_cast<char*>(language_geometry_buffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer_lang.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data_ptr<float>(),
			dL_dout_language.contiguous().data_ptr<float>(),
			dL_dout_depth.contiguous().data_ptr<float>(),
			dL_dmeans2D.contiguous().data_ptr<float>(),
			//dL_dmeans2D_lang.contiguous().data_ptr<float>(),
			dL_dconic.contiguous().data_ptr<float>(),
			dL_dconic_lang.contiguous().data_ptr<float>(),
			dL_dopacity.contiguous().data_ptr<float>(),
			dL_dopacity_lang.contiguous().data_ptr<float>(),
			dL_dcolors.contiguous().data_ptr<float>(),
			dL_dlanguage.contiguous().data_ptr<float>(),
			dL_ddepths.contiguous().data_ptr<float>(),
			dL_dmeans3D.contiguous().data_ptr<float>(),
			//dL_dmeans3D_lang.contiguous().data_ptr<float>(),
			dL_dcov3D.contiguous().data_ptr<float>(),
			dL_dcov3D_lang.contiguous().data_ptr<float>(),
			dL_dsh.contiguous().data_ptr<float>(),
			//dL_dlanguage_sh.contiguous().data_ptr<float>(),
			dL_dscales.contiguous().data_ptr<float>(),
			dL_dscales_lang.contiguous().data_ptr<float>(),
			dL_drotations.contiguous().data_ptr<float>(),
			dL_drotations_lang.contiguous().data_ptr<float>(),
			dL_dtau.contiguous().data_ptr<float>(),
			//dL_dtau_lang.contiguous().data_ptr<float>(),
			debug);
	}

	return std::make_tuple(dL_dmeans2D,
						//dL_dmeans2D_lang,
						dL_dcolors,
						dL_dlanguage,
						dL_dopacity,
						dL_dopacity_lang,
						dL_dmeans3D,
						//dL_dmeans3D_lang,
						dL_dcov3D,
						dL_dcov3D_lang,
						dL_dsh,
						//dL_dlanguage_sh,
						dL_dscales,
						dL_dscales_lang,
						dL_drotations,
						dL_drotations_lang,
						dL_dtau);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}