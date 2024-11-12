#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>

namespace yolo_cvdnn_inference {

using std::string;
using std::vector;

struct DetectResult {
  cv::Rect2f box;
  int class_index;
  std::string class_label;
};

class Yolo {
 public:
  Yolo() = default;

  virtual bool Load(vector<int> input_size, vector<string> classes,
                    string model_path) = 0;

  virtual vector<DetectResult> Detect(cv::Mat img0) = 0;

  // scale the image using letterbox algorithm
  static cv::Mat mat2letterbox(cv::Mat &img, cv::Size new_shape,
                               cv::Scalar color, bool _auto, bool scaleFill,
                               bool scaleup, int stride);

 protected:
  vector<string> labels_;
  vector<int> input_size_;
  vector<int> output_size_;
  cv::dnn::Net net_;
};
}  // namespace yolo_cvdnn_inference