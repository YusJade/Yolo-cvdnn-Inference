#include "yolov7.h"

#include <cassert>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/dnn/dnn.hpp>
#include <spdlog/spdlog.h>

using namespace yolo_cvdnn_inference;

bool Yolov7::Load(vector<int> input_size, vector<string> classes,
                  string model_path) {
  try {
    net_ = cv::dnn::readNetFromONNX(model_path);
  } catch (cv::Exception e) {
    SPDLOG_ERROR("failed to load yolov7 onnx model: {}", e.what());
    assert(false);
  }

  SPDLOG_INFO("success to load yolov7 onnx model");
  return true;
}

vector<DetectResult> Yolov7::Detect(cv::Mat img0) {
  cv::Mat img =
      mat2letterbox(img0, cv::Size(input_size_[2], input_size_[3]),
                    cv::Scalar(114, 114, 114), false, false, true, 32);
  cv::Mat blob;
  cv::dnn::blobFromImage(img, blob, 1 / 255.0f, cv::Size(img.cols, img.rows),
                         cv::Scalar(0, 0, 0), true, false);
  net_.setInput(blob);

  std::vector<std::string> outlayer_names{"output"};
  std::vector<cv::Mat> result;
  net_.forward(result, outlayer_names);

  cv::Mat out(result[0].size[1], result[0].size[2], CV_32F, result[0].data);

  std::vector<DetectResult> res;
  std::vector<cv::Rect> boxes;
  std::vector<int> indices;
  std::vector<float> scores;
  std::vector<int> class_indexes;

  for (int r = 0; r < out.size[0]; r++) {
    float cx = out.at<float>(r, 0);
    float cy = out.at<float>(r, 1);
    float w = out.at<float>(r, 2);
    float h = out.at<float>(r, 3);
    float sc = out.at<float>(r, 4);
    cv::Mat confs = out.row(r).colRange(5, output_size_[4]);  // 85
    confs *= sc;
    double min_val = 0, max_val = 0;
    // 必须是二维的
    int min_idx[2] = {-1};
    int max_idx[2] = {-1};
    minMaxIdx(confs, &min_val, &max_val, min_idx, max_idx);
    scores.push_back(max_val);
    boxes.push_back(cv::Rect(cx - w / 2, cy - h / 2, w, h));
    indices.push_back(r);
    class_indexes.push_back(max_idx[1]);
  }

  cv::dnn::NMSBoxes(boxes, scores, 0.25f, 0.45f, indices);
  for (auto &ind : indices) {
    int cx = boxes[ind].x;
    int cy = boxes[ind].y;
    int w = boxes[ind].width;
    int h = boxes[ind].height;

    // if (scores[ind] > 0.6)
    //   res.push_back(
    //       {{cx, cy, w, h}, (float)scores[ind],
    //       class_list_[class_indexs[ind]]});
  }
  return vector<DetectResult>();
}