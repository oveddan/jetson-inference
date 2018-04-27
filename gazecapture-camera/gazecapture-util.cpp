#include "gazecapture-util.h"
#include "cudaMappedMemory.h"
#include "cudaCrop.h"
#include "cudaResize.h"

using namespace dlib;

void cropAndResize(void* imgRGBA, long imageWidth, long imageHeight, void* imgCropped,
    rectangle cropBox, float cropBoxScale, void* imgScaled, long resizeWidth, long resizeHeight) {

  cudaCropRGBA((float4*)imgRGBA, imageWidth, imageHeight,
    (float4*)imgCropped,
    cropBox.left() / cropBoxScale, cropBox.top() / cropBoxScale, cropBox.width() / cropBoxScale, cropBox.height() / cropBoxScale);

  cudaResizeRGBA((float4*)imgCropped, cropBox.width() / cropBoxScale,
    cropBox.height() / cropBoxScale, (float4*)imgScaled, resizeWidth, resizeHeight);
}



