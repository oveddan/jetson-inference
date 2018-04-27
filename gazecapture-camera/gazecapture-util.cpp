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

float maxWidthCm = 10;
float maxHeightCm = 5;
int w = 1920;
int h = 1080;

float mapValue(float value, float istart, float istop, float ostart, float ostop) {
  return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}

int mapGazeX(float gazeX) {
    return round(mapValue(gazeX, -maxWidthCm, maxWidthCm, 0, w));
}

int mapGazeY(float gazeY) {
    return round(mapValue(gazeY, -2.5, -20, 0, h));
}

float2 toGazeCoords(float* gazeCm) {
  return make_float2(mapGazeX(gazeCm[0]), mapGazeY(gazeCm[1]));
}
