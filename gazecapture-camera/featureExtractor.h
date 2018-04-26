#ifndef __FEATURE_EXTRACTOR_H__
#define __FEATURE_EXTRACTOR_H__

#include <dlib/image_processing.h>
#include "dlib/image_processing/frontal_face_detector.h"

const float detectionScale = 0.25;

class FeatureExtractor {
  public:
    FeatureExtractor();

    void extract(
      long height, long width, void* imgCPU,
      std::vector<dlib::rectangle> &face_boxes,
      std::vector<dlib::rectangle> &left_eye_boxes,
      std::vector<dlib::rectangle> &right_eye_boxes
    );

  protected:
    dlib::frontal_face_detector detector;
    dlib::shape_predictor pose_model;
};

#endif


