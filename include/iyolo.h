#include <map>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace rknn_yolo_inference {

using std::string;
using std::vector;

struct DetectResult {
  cv::Rect2f box;
  int class_index;
  std::string class_label;
};

class IYolo {
 public:
  IYolo() = default;

  virtual vector<DetectResult> Detect(cv::Mat img0) = 0;

  virtual string ToString() { return "I`m IYolo."; }
};
}  // namespace rknn_yolo_inference
