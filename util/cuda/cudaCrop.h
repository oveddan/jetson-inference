#ifndef __CUDA_CROP_H__
#define __CUDA_CROP_H__


#include "cudaUtility.h"


/**
 * Function for cropping an image on the GPU.
 * @ingroup util
 */
cudaError_t cudaCropRGBA( float4* input,  size_t inputWidth,  size_t inputHeight,
				        float4* output, size_t left, size_t top, size_t cropWidth, size_t cropHeight);

#endif

