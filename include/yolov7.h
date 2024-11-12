#include "yolo.h"

namespace yolo_cvdnn_inference {
class Yolov7 : public Yolo {
 public:
  Yolov7() : Yolo() {}

  bool Load(vector<int> input_size, vector<string> classes,
            string model_path) override;
  vector<DetectResult> Detect(cv::Mat img0) override;

 private:
};
}  // namespace yolo_cvdnn_inference