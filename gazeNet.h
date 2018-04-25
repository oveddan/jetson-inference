/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __GAZE_NET_H__
#define __GAZE_NET_H__


#include "tensorNet.h"

const int GAZE_IMAGE_DIM = 224;
const int GAZE_FACE_GRID_DIM = 25;

/**
 * Image recognition with GoogleNet/Alexnet or custom models, using TensorRT.
 * @ingroup deepVision
 */
class gazeNet : public tensorNet
{
public:
	/**
	 * Load a new network instance
	 */
	static gazeNet* Create(uint32_t maxBatchSize);

	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto (can be NULL)
	 * @param class_info File path to list of class name labels
	 * @param input Name of the input layer blob.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static gazeNet* Create(const char* prototxt_path, const char* model_path, const char* mean_face_binary,
   const char* mean_left_binary, const char* mean_right_binary, uint32_t maxBatchSize );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static gazeNet* Create( int argc, char** argv );

	/**
	 * Destroy
	 */
	virtual ~gazeNet();

  inline uint32_t GetMaxGazes() {
    return maxBatchSize;
  };

  bool Detect( float* faceImage, float* leftEyeImage, float* rightEyeImage,
      float* faceGrid, float* gaze);

protected:
  gazeNet();

	bool init(const char* prototxt_path, const char* model_path, const char* mean_face_binary,
  const char* mean_left_binary, const char* mean_right_binary, uint32_t maxBatchSize );

  uint32_t maxBatchSize;
};


#endif
