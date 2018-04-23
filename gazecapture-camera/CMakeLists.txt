
file(GLOB gazeCaptureCameraSources *.cpp)
file(GLOB gazeCaptureCameraIncludes *.h )

cuda_add_executable(gazeCapture-camera ${gazeCaptureCameraSources})
target_link_libraries(gazeCapture-camera nvcaffe_parser nvinfer jetson-inference)
