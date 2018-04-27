#ifndef __CUDA_FACE_GRID__
#define __CUDA_FACE_GRID__

#include "cudaUtility.h"

cudaError_t cudaFaceGrid(
    float* output, long frameW, long frameH, long gridW, long gridH,
    long labelFaceX, long labelFaceY, long labelFaceW, long labelFaceH);

#endif
