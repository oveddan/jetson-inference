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

#include "cudaNormalize.h"
// #include "cudaGrayscale.h"
#include "cudaOverlay.h"
#include "cudaFont.h"
#include "gazeNet.h"

#define DEFAULT_CAMERA 0	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)

using namespace dlib;


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
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;

	if( !display ) {
		printf("\ngazenet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
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

  cv::namedWindow( "frame", CV_WINDOW_AUTOSIZE );

  image_window win;

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
    cv::resize(matPrevRGB, res, cv::Size(), 0.5, .5);
    std::time_t e = std::time(0);
    printf("New size, time: %f %f %f\n", float( clock () - begin_time ) / CLOCKS_PER_SEC, camera->GetHeight()*.5, camera->GetWidth()*.5);

    // cv::Mat matPrevRGB;
    // cv::cvtColor(matPrev, matPrevRGB, CV_RGB2BGR);

    // // cv::Mat mat2;
    // cv::cvtColor(matRGBA, mat2, CV_RGB2BGR);

    // void* imgGrayscale = NULL;
    begin_time = clock();

    cv_image<rgb_pixel> cimg(res);
    std::vector<rectangle> faces = detector(cimg);
    std::vector<full_object_detection> shapes;
     // Display it all on the screen
    for(unsigned long i = 0; i < faces.size(); i++) {
      shapes.push_back(pose_model(cimg, faces[i]));
    }
    printf("faces, detect time: %lu %f\n", faces.size(), float( clock () - begin_time ) / CLOCKS_PER_SEC);
    win.clear_overlay();
    win.set_image(cimg);
    win.add_overlay(render_face_detections(shapes));

    std::vector<image_window::overlay_line> lines;
    const rgb_pixel color = rgb_pixel(0,0,255);
    for(unsigned long i = 0; i < shapes.size(); i++) {
      const full_object_detection& d = shapes[i];

      // left eye
      for(unsigned long i = 37; i <= 41; i++) {
        lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
      }

      // right eye
      for(unsigned long i = 43; i <= 47; i++) {
        lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
      }
    }
    win.add_overlay(lines);

   // cv::imshow("frame", matPrevRGB);
    // cv::waitKey(0);

    // if( CUDA_FAILED(rgbaToGreyscaleCuda((uchar4*)imgRGBA, &imgGrayscale, camera->GetWidth(), camera->GetHeight())));
    // {
      // printf("CUDA convert to grayscale failed\n");
      // return -1;
    // }

		// classify image
		const int img_class = net->Classify((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), &confidence);

		if( img_class >= 0 )
		{
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "TensorRT build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);
			}
		}


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

				// draw the texture
				texture->Render(100,100);
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

