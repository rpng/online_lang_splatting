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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_COLOR_CHANNELS 3 // Default 3, RGB
#define NUM_LANGUAGE_CHANNELS 3 // Default 3, language
#define BLOCK_X 16
#define BLOCK_Y 16

// #define CUDA_DEVICE_INDEX 7
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#endif