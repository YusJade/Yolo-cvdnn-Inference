
#include <string>

#include "iyolo.h"
#include "third_party/utils/common.h"
#include "third_party/utils/image_utils.h"
#include "third_party/yolo11/cpp/yolo11.h"

namespace rknn_yolo_inference {

// Yolo11 version implement, using rknn.
class Yolo11 : public IYolo {
 public:
  Yolo11(std::string model);

  vector<DetectResult> Detect(cv::Mat img0) override;

  string ToString() override { return "I`m Yolo11."; }

 private:
  rknn_app_context_t rknn_app_ctx_;
};
}  // namespace rknn_yolo_inference
