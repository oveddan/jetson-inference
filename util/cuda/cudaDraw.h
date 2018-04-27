#ifndef __CUDA_DRAW_H__
#define __CUDA_DRAW_H__

#include "cudaUtility.h"

cudaError_t cudaDrawCircle( float4* input, float4* output,
    int width, int height, int cx, int cy, float r, float3 color);

#endif
