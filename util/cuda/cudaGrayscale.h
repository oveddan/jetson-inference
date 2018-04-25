#ifndef __CUDA_GRAYSCALE_H__
#define __CUDA_GRAYSCALE_H__

#include "cudaUtility.h"
#include <stdint.h>

cudaError_t rgbaToGreyscaleCuda(uchar4 * const d_rgbaImage,
    unsigned char* const d_greyImage, const size_t numRows, const size_t numCols);

#endif


