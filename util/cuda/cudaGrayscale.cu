#include "cudaGrayscale.h"

__global__ void rgbaToGreyscaleCudaKernel(const uchar4* const rgbaImage,
    unsigned char* const greyImage,
    const int numRows, const int numCols) {
  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset
  const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;

  if(pointIndex<numRows*numCols) { // this is necessary only if too many threads are started
    uchar4 const imagePoint = rgbaImage[pointIndex];
    greyImage[pointIndex] = .299f*imagePoint.x + .587f*imagePoint.y  + .114f*imagePoint.z;
  }
}

// Parallel implementation for running on GPU using multiple threads.
cudaError_t rgbaToGreyscaleCuda(uchar4 * const d_rgbaImage,
    unsigned char* const d_greyImage, const size_t numRows, const size_t numCols)
{
  const int blockThreadSize = 256;
  const int numberOfBlocks= 1 + ((numRows*numCols - 1) / blockThreadSize); // a/b rounded up
  const dim3 blockSize(blockThreadSize, 1, 1);
  const dim3 gridSize(numberOfBlocks, 1, 1);
  rgbaToGreyscaleCudaKernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

	return CUDA(cudaGetLastError());
}


