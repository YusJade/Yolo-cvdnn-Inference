#include "camera.h"

#include <string>
#include <thread>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

using namespace arm_face_id;

Camera::Camera(const Camera::Settings& settings,
               treasure_chest::pattern::SyncQueue<cv::Mat>& queue)
    : treasure_chest::pattern::Producer<cv::Mat>(queue), settings_(settings) {
  spdlog::info(
      "camera settings: [cam_url={}, cam_index={}, "
      "enable_net_cam={}].",
      settings.cam_url, settings.cam_index, settings.enable_net_cam);
}

void Camera::Start() {
  if (!cap_.isOpened()) {
    spdlog::error("failed to open camera.");
    return;
  }
  is_running_ = true;
  cv::Mat frame;
  while (true) {
    // if (!cap_.read(frame)) {
    //   // SPDLOG_INFO("camera can`t grab more frames from source, stop...");
    //   is_running_ = true;
    //   // break;
    // }
    cap_ >> frame;
    if (frame.empty()) break;
    // 向任务队列发送任务
    PutTask(frame.clone());
    // 通知观察者新的画面帧
    Notify(frame);
    // OnNotify(frame);
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_msec_));
  }
  is_running_ = false;
  SPDLOG_INFO("grabbed all frames from source...");
}

void Camera::Close() {
  is_running_ = false;
  cap_.release();
}

bool Camera::Open() {
  // bool is_success = false;
  if (settings_.enable_net_cam) {
    cap_.open(settings_.cam_url);
    if (!cap_.isOpened()) {
      spdlog::warn("failed to open network camera :< [cam_url:{}]",
                   settings_.cam_url);
    }
  } else {
    spdlog::info("disabled network camera :O [enable_net_cam:]",
                 settings_.enable_net_cam);
  }

  if (cap_.isOpened()) {
    spdlog::info("opened network camera :) [cam_url:{}]", settings_.cam_url);
  } else if (cap_.open(settings_.cam_index)) {
    spdlog::info("opened native camera :) [cam_index:{}]", settings_.cam_index);
  } else {
    spdlog::error("failed to open camera :( [cam_index:{}]",
                  settings_.cam_index);
    return false;
  }

  return true;
}

void Camera::OnNotify(const cv::Mat& mat) {
  if (!writer_ && !mat.empty()) {
    int frame_width = mat.cols;
    int frame_height = mat.rows;
    int fps = 30;
    std::string name = "result.avi";
    auto fourcc = cv::VideoWriter::fourcc('D', 'I', 'V', 'X');
    writer_ =
        new cv::VideoWriter(name, fourcc, fps, {frame_width, frame_height});
    if (!writer_->isOpened()) {
      SPDLOG_ERROR("failed to create {}!", name);
    } else {
      SPDLOG_INFO("save to {}", name);
    }
  } else {
    SPDLOG_INFO("try to write.");
    if (!mat.empty()) writer_->write(mat);
  }
}