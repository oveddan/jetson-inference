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

#include "gazeNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"
#include "commandLine.h"


const char* DEFAULT_PROTOTXT = "networks/gaze-capture/itracker_deploy.prototxt";
const char* DEFAULT_MODEL = "networks/gaze-capture/itracker25x_iter_92000.caffemodel";
const char* DEFAULT_MEAN_FACE = "networks/gaze-capture/mean_face_224.mat";
const char* DEFAULT_MEAN_LEFT = "networks/gaze-capture/mean_left_224.mat";
const char* DEFAULT_MEAN_RIGHT = "networks/gaze-capture/mean_right_224.mat";

#define INPUT_FACE 0
#define INPUT_LEFT_EYE 1
#define INPUT_RIGHT_EYE 2
#define INPUT_FACE_GRID 3
#define OUTPUT_GAZE 0

// constructor
gazeNet::gazeNet() : tensorNet()
{
}


// destructor
gazeNet::~gazeNet()
{

}

// Create
gazeNet* gazeNet::Create( const char* prototxt_path, const char* model_path, const char* mean_face_binary,
   const char* mean_left_binary, const char* mean_right_binary, uint32_t maxBatchSize )
{
  gazeNet* net = new gazeNet();
  net->maxBatchSize = maxBatchSize;

  if( !net )
    return NULL;

  if( !net->init(prototxt_path, model_path, mean_face_binary, mean_left_binary, mean_right_binary, maxBatchSize) )
  {
    printf("gazeNet -- failed to initialize.\n");
    return NULL;
  }

  return net;
}

// Create
gazeNet* gazeNet::Create(uint32_t maxBatchSize)
{
  return gazeNet::Create(DEFAULT_PROTOTXT, DEFAULT_MODEL, DEFAULT_MEAN_FACE, DEFAULT_MEAN_LEFT, DEFAULT_MEAN_RIGHT, maxBatchSize);
}


// Create
gazeNet* gazeNet::Create( int argc, char** argv )
{
  commandLine cmdLine(argc, argv);

  int maxBatchSize = cmdLine.GetInt("batch_size");

  if( maxBatchSize < 1 ) {
    maxBatchSize = 2;
  }

  // create from pretrained model
  return gazeNet::Create(maxBatchSize);
}

// init
bool gazeNet::init(const char* prototxt_path, const char* model_path, const char* mean_face_path, const char* mean_left_path, const char* mean_right_path, uint32_t maxBatchSize )
{
  if( !prototxt_path || !model_path || !mean_face_path || !mean_left_path || !mean_right_path)
    return false;

  printf("\n");
  printf("gazeNet -- loading classification network model from:\n");
  printf("         -- prototxt     %s\n", prototxt_path);
  printf("         -- model        %s\n", model_path);
  printf("         -- batch_size   %u\n\n", maxBatchSize);

  const std::string input3 = "image_face";
  const std::string input1 = "image_left";
  const std::string input2 = "image_right";
  const std::string input4 = "flatten";

  std::vector<std::string> inputs;

  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  const std::string output = "fc3";

  std::vector<std::string> outputs;
  outputs.push_back(output);

  /*
   * load and parse googlenet network definition and model file
   */
  if( !tensorNet::LoadNetwork( prototxt_path, model_path, inputs, outputs, maxBatchSize ) )
  {
    printf("failed to load %s\n", model_path);
    return false;
  }

  printf(LOG_GIE "%s loaded\n", model_path);

  return true;
}


// from gazeNet.cu
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight);

bool gazeNet::Detect( float* faceImage, float* leftEyeImage, float* rightEyeImage,
      float* faceGrid, float* gaze) {


  if(CUDA_FAILED(cudaPreImageNet((float4*)faceImage, 244, 244, mInputs[INPUT_FACE].CUDA, 244, 244))) {
    printf("gazeNet::Detect() -- cudaPreImageNet failed\n");
    return false;
  }
  if(CUDA_FAILED(cudaPreImageNet((float4*)leftEyeImage, 244, 244, mInputs[INPUT_LEFT_EYE].CUDA, 244, 244))) {
    printf("gazeNet::Detect() -- cudaPreImageNet failed\n");
    return false;
  }
  if(CUDA_FAILED(cudaPreImageNet((float4*)rightEyeImage, 244, 244, mInputs[INPUT_RIGHT_EYE].CUDA, 244, 244))) {
    printf("gazeNet::Detect() -- cudaPreImageNet failed\n");
    return false;
  }
  if(CUDA_FAILED(cudaMemcpy((float4*)mInputs[INPUT_FACE_GRID].CUDA, faceGrid, 25*25*sizeof(float), cudaMemcpyDeviceToDevice))) {
    printf("gazeNet::Detect() -- cudaPreImageNet failed\n");
    return false;
  }

  void* inferenceBuffers[] = {
    mInputs[INPUT_FACE].CUDA,
    mInputs[INPUT_LEFT_EYE].CUDA,
    mInputs[INPUT_RIGHT_EYE].CUDA,
    mInputs[INPUT_FACE_GRID].CUDA,
    mOutputs[OUTPUT_GAZE].CUDA
  };

  if (!mContext->execute(1, inferenceBuffers)) {
    printf(LOG_GIE "gazeCapture::Detect() -- failed to execute tensorRT context\n");

    return false;
  }

  PROFILER_REPORT();

  gaze = mOutputs[OUTPUT_GAZE].CPU;

  printf("predicted gaze %f, %f\n", gaze[0], gaze[1]);

  return true;
}


// Classify
// int gazeNet::Classify( float* rgba, uint32_t width, uint32_t height, float* confidence )
// {
  // if( !rgba || width == 0 || height == 0 )
  // {
    // printf("gazeNet::Classify( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
    // return -1;
  // }


  // for(int i = 0; i < mInputs.size(); i++) {
    // // downsample and convert to band-sequential BGR
    // if( CUDA_FAILED(cudaPreImageNetMean((float4*)rgba, width, height, mInputs[i].CUDA, mInputs[i].width, mInputs[i].height,
                   // make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f))) )
    // {
      // printf("gazeNet::Classify() -- cudaPreImageNetMean failed\n");
      // return -1;
    // }
  // }


  // // process with GIE
  // void* inferenceBuffers[] = { mInputs[0].CUDA, mInputs[1].CUDA, mInputs[2].CUDA, mInputs[3].CUDA, mOutputs[0].CUDA };

  // mContext->execute(1, inferenceBuffers);

  // //CUDA(cudaDeviceSynchronize());
  // PROFILER_REPORT();


  // // determine the maximum class
  // // todo: extract outputs
  // //printf("\nmaximum class:  #%i  (%f) (%s)\n", classIndex, classMax, mClassDesc[classIndex].c_str());
  // return 0;
// }

