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

#include "gstCamera.h"

#include <dlib/image_processing.h>

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
// #include "cudaGrayscale.h"
#include "cudaOverlay.h"
#include "cudaCrop.h"
#include "cudaRGB.h"
#include "cudaFaceGrid.h"
#include "cudaResize.h"
#include "cudaFont.h"
#include "cudaDraw.h"
#include "featureExtractor.h"
#include "gazecapture-util.h"
#include "gazeNet.h"

#define DEFAULT_CAMERA 0	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)

bool signal_recieved = false;
using namespace dlib;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

#define FACE_GRID_SIZE 25

int main( int argc, char** argv )
{
	printf("gazenet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);

	printf("\n\n");


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(1920, 1080, DEFAULT_CAMERA);

	if( !camera )
	{
		printf("\ngazenet-camera:  failed to initialize video device\n");
		return 0;
	}

	printf("\ngazenet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());


	/*
	 * create gazeNet
	 */
	gazeNet* net = gazeNet::Create(argc, argv);

	if( !net )
	{
		printf("gazenet-console:   failed to initialize gazeNet\n");
		return 0;
	}

	/*
	 * allocate memory for output gazes
	 */
	const uint32_t maxGazes= net->GetMaxGazes();		printf("maximum gazes:  %u\n", maxGazes);

  float* gazesCPU    = NULL;
  void* imgCropped = NULL;
  void* imgFace = NULL;
  void* imgLeftEyeCropped = NULL;
  void* imgLeftEye = NULL;
  void* imgRightEyeCropped = NULL;
  void* imgRightEye = NULL;
  void* faceGrid = NULL;
  void* faceGridCPU = NULL;
  void* outputGrayscale = NULL;
  void* outputRGBA = NULL;

  long cameraWidth = camera->GetWidth();
  long cameraHeight = camera->GetHeight();
  long cropImageSize = cameraWidth * cameraHeight * sizeof(float4);
  long resizeImageSize = 244 * 244 * sizeof(float4);
  int faceGridSize = FACE_GRID_SIZE * FACE_GRID_SIZE * sizeof(float1);

  if(CUDA_FAILED(cudaHostAlloc((void**)&gazesCPU, maxGazes*sizeof(float2), 0))) {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }
  if(CUDA_FAILED(cudaMalloc((void**)&imgCropped, cropImageSize)))
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }
  if(CUDA_FAILED(cudaMalloc((void**)&imgFace, resizeImageSize)))
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }
  if(CUDA_FAILED(cudaMalloc((void**)&imgLeftEye, resizeImageSize)))
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }
  if(CUDA_FAILED(cudaMalloc((void**)&imgRightEye, resizeImageSize)))
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }
  if( !cudaAllocMapped(&faceGridCPU, &faceGrid, faceGridSize) )
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }
  if(CUDA_FAILED(cudaMalloc((void**)&outputGrayscale, 244 * 244 * sizeof(float))))
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }
  if(CUDA_FAILED(cudaMalloc((void**)&outputRGBA, 244 * 244 * sizeof(float4))))
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }

  /*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	glTexture* faceTexture = NULL;

	if( !display ) {
		printf("\ngazenet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("gazenet-camera:  failed to create openGL texture\n");

    faceTexture = glTexture::Create(244, 244, GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !faceTexture )
			printf("gazenet-camera:  failed to create openGL texture\n");

	}


	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();


	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\ngazenet-camera:  failed to open camera for streaming\n");
		return 0;
	}

	printf("\ngazenet-camera:  camera open for streaming\n");


	/*
	 * processing loop
	 */
	float confidence = 0.0f;

  std::vector<rectangle> face_boxes;
  std::vector<rectangle> left_eye_boxes;
  std::vector<rectangle> right_eye_boxes;

  FeatureExtractor featureExtractor;

	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;

		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\ngazenet-camera:  failed to capture frame\n");
    // else
      // printf("gazenet-camera:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);

		// convert from YUV to RGBA
		void* imgRGBA = NULL;

		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA, true) )
			printf("gazenet-camera:  failed to convert from NV12 to RGBA\n");

    long width = camera->GetWidth();
    long height = camera->GetHeight();

    face_boxes.clear();
    left_eye_boxes.clear();
    right_eye_boxes.clear();
    featureExtractor.extract(height, width, imgCPU,
        face_boxes, left_eye_boxes, right_eye_boxes);

    bool gazeDetected = false;

    if(face_boxes.size() > 0) {
      int numGazes = maxGazes;

      rectangle face_box = face_boxes[0];

      cropAndResize(imgRGBA, width, height, imgCropped,
              face_boxes[0], detectionScale, imgFace, 244, 244);
      cropAndResize(imgRGBA, width, height, imgCropped,
              left_eye_boxes[0], detectionScale, imgLeftEye, 244, 244);
      cropAndResize(imgRGBA, width, height, imgCropped,
              right_eye_boxes[0], detectionScale, imgRightEye, 244, 244);


      point center = dlib::center(face_box);
      cudaFaceGrid(
        (float*)faceGrid, width, height, FACE_GRID_SIZE, FACE_GRID_SIZE,
        center.x() / detectionScale, center.y() / detectionScale,
        face_box.width() / detectionScale, face_box.height() / detectionScale);

      // print out face grid. todo: move to function
      if(false) {
        for(int i = 0; i < 25 * 25; i++) {
          float* iptr = (float*) faceGridCPU;
          std::cout << iptr[i] << " ";
          if (i % 25 == 0)
            std::cout << "\n";
        }
      }

      // classify image
      if(net->Detect((float*)imgFace, (float*)imgLeftEye, (float*)imgRightEye, (float*)faceGrid,
            gazesCPU)) {
        gazeDetected = true;
          printf("gaze detected");
      }
    }

   if( display != NULL )
    {
      char str[256];
      sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
      display->SetTitle(str);
    }

		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
        if (gazeDetected){
          float2 gazeCoords = toGazeCoords((float*)gazesCPU);

          printf("Gaze coords: %f %f \n", gazeCoords.x, gazeCoords.y);

          cudaDrawCircle((float4*)imgRGBA, (float4*)imgRGBA,
              width, height, gazeCoords.x, gazeCoords.y, 5.0f,
              make_float3(255.0, 0.0, 0.0));
        }

				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f),
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f),
		 						   width, height));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

        if( tex_map != NULL )
        {
          cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
          texture->Unmap();
        }

        texture->Render(100,100);
      }

      if(faceTexture != NULL && face_boxes.size() > 0) {
        void* tex_map2;

        // RENDER LEFT_EYE
        CUDA(cudaNormalizeRGBA((float4*)imgLeftEye, make_float2(0.0f, 255.0f),
                   (float4*)imgLeftEye, make_float2(0.0f, 1.0f),
                   244, 244));

        tex_map2 = faceTexture->MapCUDA();

        if( tex_map2 != NULL )
        {
          cudaMemcpy(tex_map2, imgLeftEye, faceTexture->GetSize(), cudaMemcpyDeviceToDevice);
          faceTexture->Unmap();
        }

        // draw the texture
        faceTexture->Render(100,100);

        // RENDER FACE
        // rescale image pixel intensities of face for display
        CUDA(cudaNormalizeRGBA((float4*)imgFace, make_float2(0.0f, 255.0f),
                   (float4*)imgFace, make_float2(0.0f, 1.0f),
                   244, 244));


        tex_map2 = faceTexture->MapCUDA();

        if( tex_map2 != NULL )
        {
          cudaMemcpy(tex_map2, imgFace, faceTexture->GetSize(), cudaMemcpyDeviceToDevice);
          faceTexture->Unmap();
        }


        // draw the texture
        faceTexture->Render(100 + 244,100);

       // RENDER RIGHT EYE
        CUDA(cudaNormalizeRGBA((float4*)imgRightEye, make_float2(0.0f, 255.0f),
                   (float4*)imgRightEye, make_float2(0.0f, 1.0f),
                   244, 244));


        tex_map2 = faceTexture->MapCUDA();

        if( tex_map2 != NULL )
        {
          cudaMemcpy(tex_map2, imgRightEye, faceTexture->GetSize(), cudaMemcpyDeviceToDevice);
          faceTexture->Unmap();
        }

        // draw the texture
        faceTexture->Render(100+244*2,100);


        // RENDER FACE GRID
        CUDA(cudaResize((float*)faceGrid, 25, 25,
                   (float*)outputGrayscale, 244, 244));

       CUDA(cudaGrayscaleRGBAf((float*)outputGrayscale,
          (float4*)outputRGBA, 244, 244));

        tex_map2 = faceTexture->MapCUDA();

        if( tex_map2 != NULL )
        {
          cudaMemcpy(tex_map2, outputRGBA, faceTexture->GetSize(), cudaMemcpyDeviceToDevice);
          faceTexture->Unmap();
        }

        // draw the texture
        faceTexture->Render(100+244*3,100);
      }

			display->EndRender();
		}
	}

	printf("\ngazenet-camera:  un-initializing video device\n");


	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}

	printf("gazenet-camera:  video device has been un-initialized.\n");
	printf("gazenet-camera:  this concludes the test of the video device.\n");
	return 0;
}

