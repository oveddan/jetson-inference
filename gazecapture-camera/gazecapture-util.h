#ifndef __GAZE_CAPTURE_UTIL_H__
#define __GAZE_CAPTURE_UTIL_H__

#include "cudaMappedMemory.h"
#include <dlib/image_processing.h>

void cropAndResize(void* imgRGBA, long imageWidth, long imageHeight, void* imgCropped,
    dlib::rectangle cropBox, float cropBoxScale, void* imgScaled, long resizeWidth, long resizeHeight);

float2 toGazeCoords(float* gazeCm);

#endif

