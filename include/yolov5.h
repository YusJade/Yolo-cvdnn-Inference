#pragma once

#include <opencv2/core/mat.hpp>

#include "iyolo.h"
#include "rknn_api.h"

namespace rknn_yolo_inference {
class Yolov5 : public IYolo {
 public:
  Yolov5(std::string model, std::string label_path);

  DetectResult Detect(cv::Mat img0) override;

 private:
  typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;  // 是否为量化模型（性能需求比浮点模型更小）
  } rknn_app_context_t;

  /**
   * @brief 初始化和装载 rknn_app_context_t
   * @param model_path
   */
  void InitModel(std::string model_path);
  int InitPostProcess(std::string label_path);

  static cv::Mat Letterbox(const cv::Mat& src, const cv::Size& target_size);

  rknn_app_context_t rknn_app_ctx_;
  std::vector<std::string> labels_;
};
}  // namespace rknn_yolo_inference