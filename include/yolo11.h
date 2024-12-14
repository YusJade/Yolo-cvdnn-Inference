
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include "iyolo.h"
#include "observer/core.h"
#include "third_party/utils/common.h"
#include "third_party/utils/image_utils.h"
#include "third_party/yolo11/cpp/yolo11.h"

namespace rknn_yolo_inference {

using treasure_chest::pattern::Subject;

// Yolo11 version implement, using rknn.
class Yolov7 : public IYolo {
 public:
  Yolov7(std::string model);

  DetectResult Detect(cv::Mat img0) override;

  string ToString() override { return "I`m Yolo11."; }

  bool Save();

 private:
  rknn_app_context_t rknn_app_ctx_;
};
}  // namespace rknn_yolo_inference
