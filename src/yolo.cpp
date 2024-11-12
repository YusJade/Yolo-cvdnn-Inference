#include "yolo.h"

#include <opencv2/imgproc.hpp>

using namespace yolo_cvdnn_inference;

cv::Mat Yolo::mat2letterbox(cv::Mat &img, cv::Size new_shape, cv::Scalar color,
                            bool _auto, bool scaleFill, bool scaleup,
                            int stride) {
  float width = img.cols;
  float height = img.rows;
  float r = std::min(new_shape.width / width, new_shape.height / height);
  if (!scaleup)
    r = std::min(r, 1.0f);
  int new_unpadW = int(round(width * r));
  int new_unpadH = int(round(height * r));
  int dw = new_shape.width - new_unpadW;
  int dh = new_shape.height - new_unpadH;
  if (_auto) {
    dw %= stride;
    dh %= stride;
  }
  dw /= 2, dh /= 2;
  cv::Mat dst;
  cv::resize(img, dst, cv::Size(new_unpadW, new_unpadH), 0, 0,
             cv::INTER_LINEAR);
  int top = int(round(dh - 0.1));
  int bottom = int(round(dh + 0.1));
  int left = int(round(dw - 0.1));
  int right = int(round(dw + 0.1));
  cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT,
                     color);
  return dst;
}