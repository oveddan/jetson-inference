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

#include "opencv2/opencv.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include <dlib/image_processing/render_face_detections.h>
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
#include "cudaResize.h"
#include "cudaFont.h"
#include "gazeNet.h"

#define DEFAULT_CAMERA 0	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)

using namespace dlib;

rectangle get_bounding_box(const full_object_detection& d, unsigned long first_line, unsigned long last_line) {
  long left = 5000;
  long top = -5000;
  long right = -5000;
  long bottom = 5000;

  point pnt;
  for(unsigned long i = first_line; i <= last_line; i++) {
    pnt = d.part(i);

    left = std::min(left, pnt.x());
    right = std::max(right, pnt.x());
    top = std::max(top, pnt.y());
    bottom = std::min(bottom, pnt.y());
  }
  return rectangle(left, top, right, bottom);
};

rectangle to_square(const rectangle& rect) {
  point center = dlib::center(rect);

  long width = rect.right() - rect.left();
  long height = rect.top() - rect.bottom();
  long size = std::max(width, height);

  return dlib::centered_rect(
    center.x(), center.y(), size, size
  );
}

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


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
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);

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
	float* gazesCUDA   = NULL;
  void* imgFaceCropped = NULL;
  void* imgFaceCroppedCPU = NULL;
  void* imgFace = NULL;
  void* imgFaceCPU = NULL;
  float* imgLeftEye = NULL;
  float* imgRightEye = NULL;
  float* faceGrid = NULL;

 if( !cudaAllocMapped((void**)&imgFaceCPU, (void**)&imgFace, 244 * 244 * sizeof(float4)) )
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }
 if( !cudaAllocMapped((void**)&imgFaceCroppedCPU, (void**)&imgFaceCropped, 244 * 244 * sizeof(float4)) )
  {
    printf("gazecapture-camera:  failed to alloc output memory\n");
    return 0;
  }


	// if( !cudaAllocMapped((void**)&gazesCPU, (void**)&gazesCUDA, maxGazes * sizeof(float2)) )
	// {
		// printf("gazecapture-camera:  failed to alloc output memory\n");
		// return 0;
	// }

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

  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor pose_model;
  deserialize("networks/shape_predictor_68_face_landmarks.dat") >> pose_model;

  // cv::namedWindow( "frame", CV_WINDOW_AUTOSIZE );

  image_window win;

  float detectionScale = 0.5;

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

    clock_t begin_time = clock();
    cv::Mat matPrevRGB = cv::Mat(camera->GetHeight(), camera->GetWidth(), CV_8UC3, (char*)imgCPU, cv::Mat::AUTO_STEP);
    cv::Mat res;
    cv::resize(matPrevRGB, res, cv::Size(), detectionScale, detectionScale);
    std::time_t e = std::time(0);
    printf("New size, time: %f %f %f\n", float( clock () - begin_time ) / CLOCKS_PER_SEC, camera->GetHeight()*.5, camera->GetWidth()*.5);

    begin_time = clock();

    cv_image<rgb_pixel> cimg(res);
    std::vector<rectangle> faces = detector(cimg);
    std::vector<full_object_detection> shapes;
    for(unsigned long i = 0; i < faces.size(); i++) {
      shapes.push_back(pose_model(cimg, faces[i]));
    }
    printf("faces, detect time: %lu %f\n", faces.size(), float( clock () - begin_time ) / CLOCKS_PER_SEC);
    if (false) {
      win.clear_overlay();
      win.set_image(cimg);
    }
    // win.add_overlay(render_face_detections(shapes));

    std::vector<rectangle> face_boxes;
    std::vector<rectangle> left_eye_boxes;
    std::vector<rectangle> right_eye_boxes;
    std::vector<image_window::overlay_line> lines;
    const rgb_pixel color = rgb_pixel(0,0,255);

    for(unsigned long i = 0; i < shapes.size(); i++) {
      const full_object_detection& d = shapes[i];

      rectangle left_eye_box = to_square(get_bounding_box(d, 36, 41));
      rectangle right_eye_box = to_square(get_bounding_box(d, 42, 47));

      face_boxes.push_back(to_square(faces[i]));
      left_eye_boxes.push_back(left_eye_box);
      right_eye_boxes.push_back(right_eye_box);

      if (false) {
        win.add_overlay(to_square(faces[i]), color);
        win.add_overlay(left_eye_box, color);
        win.add_overlay(right_eye_box, color);
      }
    }

    long width = camera->GetWidth();
    long height = camera->GetHeight();
    // cv::cuda::GpuMat faceGpuMat;
    if(face_boxes.size() > 0) {
      int numGazes = maxGazes;


      rectangle face_box = face_boxes[0];
      point center = dlib::center(face_box);

      // printf("%lu %lu %lu %lu", center.x(), center.y() , face_box.width(), face_box.height());
      // cv::Rect crop(center.x() / detectionScale, center.y()/ detectionScale, face_box.width()/ detectionScale, face_box.height()/ detectionScale);

      // printf("Cropping");
      // cv::Mat cropped(matPrevRGB, crop);

      // cv::Mat resized(cv::Size(244, 244), CV_8UC3, imgFaceCPU);

      // printf("resizing");

      // cv::resize(cropped, resized, cv::Size(244, 244));
      // cudaAllocMapped(resized.data, imgFace, 244*244*sizeof(float3));
      // cv::cuda::GpuMat cropped = cv::cuda::GpuMat(cvRgb,
          // crop);

      // cv::cuda::GpuMat output;

      // cv::cuda::remap(cropped, output, cv::Size(244, 244));

      //

       cudaCropRGBA((float4*)imgRGBA, width, height, (float4*)imgFaceCropped,
              face_box.left() / detectionScale, face_box.top() / detectionScale, face_box.width() / detectionScale, face_box.height() / detectionScale);

       cudaResizeRGBA((float4*)imgFaceCropped, face_box.width() / detectionScale, face_box.height() / detectionScale, (float4*)imgFace, 244, 244);
              // face_box.left(), face_box.bottom(), face_box.width(), face_box.height()))
      // if(CUDA_FAILED(cudaCropRGBA((float4*)imgRGBA, width, height, (float4*)imgFace,
              // face_box.left(), face_box.top(), 244, 244 [>face_box.width(), face_box.height()<]))) {
        // printf("cameraNet::cropFace failed");
      // }


      // classify image
      if(false && net->Detect((float*)imgFace, (float*)imgLeftEye, (float*)imgRightEye, (float*)faceGrid,
            gazesCPU)) {
          printf("gaze detected");
      }
    }

		// if( img_class >= 0 )
		// {
			// if( display != NULL )
			// {
				// char str[256];
				// sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				// //sprintf(str, "TensorRT build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				// display->SetTitle(str);
			// }
		// }


    // int numFaceBoxes = faces.size();
    // for( int n=0; n < numFaceBoxes; n++ )
    // {
      // // rectangle face = faces[n];
      // // float4 bb((float)face.left(), (float)face.bottom(), (float)face.right(), (float)face.top());
      // // printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb.x, bb.y, bb.z, bb.w, bb.z - bb.x, bb.w - bb.y);

      // if( CUDA_FAILED(cudaRectOutlineOverlay((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
                                // (float)face.left(), (float)face.bottom(), (float)face.right(), (float)face.top()) )) {
        // printf("detectnet-console:  failed to draw boxes\n");

        // // lastClass = nc;
        // // lastStart = n;

        // CUDA(cudaDeviceSynchronize());
      // }
    // }


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f),
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f),
		 						   camera->GetWidth(), camera->GetHeight()));

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

        // rescale image pixel intensities of face for display
        CUDA(cudaNormalizeRGBA((float4*)imgFace, make_float2(0.0f, 255.0f),
                   (float4*)imgFace, make_float2(0.0f, 1.0f),
                   244, 244));


        void* tex_map2 = faceTexture->MapCUDA();

        if( tex_map2 != NULL )
        {
          cudaMemcpy(tex_map2, imgFace, faceTexture->GetSize(), cudaMemcpyDeviceToDevice);
          faceTexture->Unmap();
        }

        // draw the texture
        faceTexture->Render(500,500);

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

