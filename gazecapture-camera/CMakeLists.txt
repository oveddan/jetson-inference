
file(GLOB gazeCaptureCameraSources *.cpp)
file(GLOB gazeCaptureCameraIncludes *.h )

find_package(OpenCV REQUIRED)

add_subdirectory("../libs/dlib" dlib_build)

cuda_add_executable(gazeCapture-camera ${gazeCaptureCameraSources})
target_link_libraries(gazeCapture-camera nvcaffe_parser dlib::dlib nvinfer jetson-inference ${OpenCV_LIBS})
