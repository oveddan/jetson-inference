#include "cudaCrop.h"

// gpuCrop
template <typename T>
__global__ void gpuCrop( int left, int top, T* input, int iWidth, T* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

  const int dx = left + x;
	const int dy = top + y;

	const T px = input[ dy * iWidth + dx ];

	output[y*oWidth+x] = px;
}

// cudaCropRGBA
cudaError_t cudaCropRGBA( float4* input,  size_t inputWidth,  size_t inputHeight,
				        float4* output, size_t left, size_t top, size_t cropWidth, size_t cropHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || cropWidth == 0 || inputHeight == 0 || cropHeight == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(cropWidth,blockDim.x), iDivUp(cropHeight,blockDim.y));

	gpuCrop<float4><<<gridDim, blockDim>>>(left, top, input, inputWidth, output, cropWidth, cropHeight);

	return CUDA(cudaGetLastError());
}
