#include "cudaDraw.h"

// gpuCrop
__global__ void gpuDrawCircle( float4* input, float4* output, int width, int height, int cx, int cy, float r, float3 color)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;


  float2 vec;
  
  vec.x = x-cx;
  vec.y = y-cy;

  bool renderColor = abs(vec.x) <= r && abs(vec.y) <= r;
  int pixel = y * width + x;

  if (renderColor)
    output[pixel] = make_float4(color.x, color.y, color.z, 255.);
  else
    output[pixel] = input[pixel];
}

// cudaDrawCircle
cudaError_t cudaDrawCircle( float4* input, float4* output, int width, int height, int cx, int cy, float r, float3 color)
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0)
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuDrawCircle<<<gridDim, blockDim>>>(input, output, width, height, cx, cy, r, color);

	return CUDA(cudaGetLastError());
}
