#include "featureExtractor.h"

#include "opencv2/opencv.hpp"
#include "dlib/opencv.h"
#include <dlib/image_processing/render_face_detections.h>

using namespace dlib;
using namespace std;

rectangle get_bounding_box(const full_object_detection &d, unsigned long first_line, unsigned long last_line) {
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

rectangle to_square(const rectangle &rect) {
  point center = dlib::center(rect);

  long width = rect.right() - rect.left();
  long height = rect.top() - rect.bottom();
  long size = std::max(width, height);

  return dlib::centered_rect(
    center.x(), center.y(), size, size
  );
}

FeatureExtractor::FeatureExtractor(){
  detector = get_frontal_face_detector();
  deserialize("networks/shape_predictor_68_face_landmarks.dat") >> pose_model;
}

void FeatureExtractor::extract(long height, long width, void* imgCPU,
  std::vector<dlib::rectangle> &face_boxes,
  std::vector<dlib::rectangle> &left_eye_boxes,
  std::vector<dlib::rectangle> &right_eye_boxes
) {
  clock_t begin_time = clock();
  cv::Mat matPrevRGB = cv::Mat(height, width, CV_8UC3, (char*)imgCPU, cv::Mat::AUTO_STEP);
  cv::Mat res;
  cv::resize(matPrevRGB, res, cv::Size(), detectionScale, detectionScale);
  std::time_t e = std::time(0);
  // printf("resize time: %f\n", float( clock () - begin_time ) / CLOCKS_PER_SEC);

  begin_time = clock();

  cv_image<rgb_pixel> cimg(res);
  std::vector<rectangle> faces = detector(cimg);
  std::vector<full_object_detection> shapes;
  for(unsigned long i = 0; i < faces.size(); i++) {
    shapes.push_back(pose_model(cimg, faces[i]));
  }
  // printf("faces, detect time: %lu %f\n", faces.size(), float( clock () - begin_time ) / CLOCKS_PER_SEC);

  const rgb_pixel color = rgb_pixel(0,0,255);

  for(unsigned long i = 0; i < shapes.size(); i++) {
    const full_object_detection& d = shapes[i];

    rectangle left_eye_box = to_square(get_bounding_box(d, 36, 41));
    rectangle right_eye_box = to_square(get_bounding_box(d, 42, 47));

    face_boxes.push_back(to_square(faces[i]));
    left_eye_boxes.push_back(left_eye_box);
    right_eye_boxes.push_back(right_eye_box);
  }
}
