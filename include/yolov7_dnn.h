#include <string>
#include <vector>

#include <opencv2/dnn.hpp>

#include "iyolo.h"
#include "observer/core.h"

namespace yolo_cvdnn_inference {
class Yolov7 : public rknn_yolo_inference::IYolo {
 public:
  Yolov7() : IYolo() {}

  bool Load(std::vector<int> input_size, std::vector<std::string> classes,
            std::string model_path);
  rknn_yolo_inference::DetectResult Detect(cv::Mat img0) override;

  // scale the image using letterbox algorithm
  static cv::Mat mat2letterbox(cv::Mat &img, cv::Size new_shape,
                               cv::Scalar color, bool _auto, bool scaleFill,
                               bool scaleup, int stride);

 private:
  std::vector<std::string> labels_;
  std::vector<int> input_size_;
  std::vector<int> output_size_;
  cv::dnn::Net net_;
};
}  // namespace yolo_cvdnn_inference
