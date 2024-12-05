#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/internal/flag.h>
#include <absl/flags/internal/parse.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "camera.h"
#include "sync_queue/core.h"
#include "yolo11.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

ABSL_FLAG(std::string, serial_port, "ttyS3", "use which serial port");
ABSL_FLAG(std::string, classes, "classes.txt", "the classes.txt file");
ABSL_FLAG(std::string, model, "", "model path");
ABSL_FLAG(bool, gui, false, "open with gui");
ABSL_FLAG(int, camera_index, 0, "native camera device index");
ABSL_FLAG(std::string, camera_url, "http://localhost:4747/video",
          "network camera url");
ABSL_FLAG(std::string, src, "", "video");
ABSL_FLAG(std::string, log_level, "info",
          "decide which level of log will be printed.");

using arm_face_id::Camera;
using rknn_yolo_inference::DetectResult;
using treasure_chest::pattern::SyncQueue;

std::vector<std::string> load_classes(std::string file) {
  std::ifstream file_stream(file);
  std::string class_name;
  std::vector<std::string> classes;
  while (std::getline(file_stream, class_name)) {
    classes.push_back(class_name);
  }
  return std::move(classes);
}

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] [%s:%#] %v");
  std::string log_level = absl::GetFlag(FLAGS_log_level);
  if (log_level == "info")
    spdlog::set_level(spdlog::level::info);
  else if (log_level == "debug")
    spdlog::set_level(spdlog::level::debug);

  SPDLOG_DEBUG("initialized logger, log level is: {}", log_level);
  SPDLOG_DEBUG("this is a script to inference with yolo.");

  rknn_yolo_inference::Yolo11 yolo(absl::GetFlag(FLAGS_model));

  // std::string src = absl::GetFlag(FLAGS_src);
  // bool is_picture = src.find(".jpg") || src.find(".png");
  // if (is_picture) {
  //   cv::Mat pic = cv::imread(src);
  //   DetectResult result = yolo.Detect(pic);
  //   cv::imwrite("res.jpg", result.img);

  //   SPDLOG_INFO("finished inference.");

  //   return 0;
  // }
  // cv::Mat test_img = cv::imread("test_img.jpg");
  // yolo11.Detect(test_img);
  // std::vector<std::string> classes =
  // load_classes(absl::GetFlag(FLAGS_classes)); Yolov7 yolo; yolo.Load({1, 3,
  // 640, 640}, classes, absl::GetFlag(FLAGS_model));

  SyncQueue<cv::Mat> detect_task_queue(200);
  Camera::Settings camera_settings;
  camera_settings.cam_index = absl::GetFlag(FLAGS_camera_index);
  camera_settings.cam_url = absl::GetFlag(FLAGS_camera_url);
  camera_settings.enable_net_cam = true;
  std::shared_ptr<Camera> cam =
      std::make_shared<Camera>(camera_settings, detect_task_queue);
  yolo.AddObserver<cv::Mat>(cam);
  cam->SetReadInterval(0);
  cam->Open();
  std::thread cam_thread([&] { cam->Start(); });
  std::thread detect_thread([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    SPDLOG_INFO("start to infer");
    int frame_count = 0;
    while (detect_task_queue.TaskQuantity() != 0 || cam->IsRunning()) {
      SPDLOG_INFO("cam_is_running:{}, task_quantity:{}", cam->IsRunning(),
                  detect_task_queue.TaskQuantity());
      cv::Mat mat = detect_task_queue.Dequeue();
      frame_count++;
      if (frame_count < 5) {
        continue;
      }

      frame_count = 0;
      auto start = std::chrono::high_resolution_clock::now();
      rknn_yolo_inference::DetectResult res = yolo.Detect(mat);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;
      if (!res.items.empty()) {
        SPDLOG_INFO("detected one frame, fps: {}, target: {}",
                    1 / duration.count(), res.items.size());
      }
    }
    SPDLOG_INFO("finished inference.");
    // cam->Close();
  });

  cam_thread.join();
  detect_thread.join();
  SPDLOG_INFO("saving result...");

  return 0;
}
