#include "cudaFaceGrid.h"

__global__ void gpuFaceGridFromFaceRect(float* output, 
    int gridW, int xLo, int xHi, int yLo, int yHi) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  bool isFilled = x >= xLo && x <= xHi && y >= yLo && y <= yHi;

  output[y*gridW+x] = isFilled ? 1. : 0.;
}

// gpuNormalizeFaceGridRGBA
__global__ void gpuNormalizeFaceGrid(float* input, float* output, int gridW)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= gridW || y >= gridW )
		return;

  int pixel = y * gridW + x;

  float value = input[pixel];

  float normalized = value * 255.0f;

  output[pixel] = normalized;
}



// Given face detection data, generate face grid data.
//
// Input Parameters:
// - frameW/H: The frame in which the detections exist
// - gridW/H: The size of the grid (typically same aspect ratio as the
//     frame, but much smaller)
// - labelFaceX/Y/W/H: The face detection (x and y are 0-based image
//     coordinates)
// - parameterized: Whether to actually output the grid or just the
//     [x y w h] of the 1s square within the gridW x gridH grid.
cudaError_t cudaFaceGrid(
    float* output, long frameW, long frameH, long gridW, long gridH,
    long labelFaceX, long labelFaceY, long labelFaceW, long labelFaceH) {
  float scaleX = gridW * 1.0f / frameW;
  float scaleY = gridH * 1.0f / frameH;

  // Use one-based image coordinates.
  long xLo = round(labelFaceX * scaleX);
  long yLo = round(labelFaceY * scaleY);
  long w = round(labelFaceW * scaleX);
  long h = round(labelFaceH * scaleY);

  long zero = 0;

  long xHi = xLo + w;
  long yHi = yLo + h;

  xLo = min(gridW, max(zero, xLo));
  xHi = min(gridW, max(zero, xHi));
  yLo = min(gridH, max(zero, yLo));
  yHi = min(gridH, max(zero, yHi));

	const dim3 blockDim(5, 5);
	const dim3 gridDim(iDivUp(gridW,blockDim.x), iDivUp(gridH,blockDim.y));

  gpuFaceGridFromFaceRect<<<gridDim, blockDim>>>(output, gridW, xLo, xHi, yLo, yHi);

  return CUDA(cudaGetLastError());
}
