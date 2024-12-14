#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "observer/core.h"

namespace rknn_yolo_inference {

using std::string;
using std::vector;

struct DetectItem {
  cv::Rect2f box;
  int class_index;
  std::string class_label;
  float conf;
};

struct DetectResult {
  vector<DetectItem> items;
  cv::Mat img;
};

class IYolo : public treasure_chest::pattern::Subject {
 public:
  IYolo() = default;

  virtual DetectResult Detect(cv::Mat img0) = 0;

  virtual string ToString() { return "I`m IYolo."; }
};
}  // namespace rknn_yolo_inference
