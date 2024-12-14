#include <chrono>
#include <cstdlib>
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
#include "iyolo.h"
#include "sync_queue/core.h"
#include "yolo11.h"
#include "yolov7_dnn.h"

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

void DetectVideo(rknn_yolo_inference::IYolo& yolo) {
  // 考虑到开发板内存有限, 同步队列的容量不宜过大.
  SyncQueue<cv::Mat> detect_task_queue(200);
  Camera::Settings camera_settings;
  camera_settings.cam_index = absl::GetFlag(FLAGS_camera_index);
  camera_settings.cam_url = absl::GetFlag(FLAGS_camera_url);
  camera_settings.enable_net_cam = true;
  std::shared_ptr<Camera> cam =
      std::make_shared<Camera>(camera_settings, detect_task_queue);

  // 由 yolo 绘制结果帧并通知 cam 保存为视频.
  yolo.AddObserver<cv::Mat>(cam);
  cam->SetReadInterval(0);
  cam->Open();
  std::thread cam_thread([&] { cam->Start(); });
  std::thread detect_thread([&] {
    // 等待队列任务抵达, 避免提前结束.
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
}

void DetectImage(rknn_yolo_inference::IYolo& yolo, std::string img_path) {
  cv::Mat img = cv::imread(img_path);
  auto result = yolo.Detect(img);
  cv::imwrite("result.jpg", result.img);
  spdlog::info("Result of image inference saved as result.jpg");
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] [%s:%#] %v");
  std::string log_level = absl::GetFlag(FLAGS_log_level);
  if (log_level == "info")
    spdlog::set_level(spdlog::level::info);
  else if (log_level == "debug")
    spdlog::set_level(spdlog::level::debug);

  SPDLOG_DEBUG("Initialized logger, log level is: {}", log_level);
  SPDLOG_DEBUG("This is a script to inference with yolo.");

  std::string model = absl::GetFlag(FLAGS_model);
  rknn_yolo_inference::IYolo* yolo = nullptr;
  // 如果是 onnx 模型, 则加载 dnn 推理.
  if (model.find(".onnx") != std::string::npos) {
    spdlog::info("Loading onnx inference unit~");
    auto yolov7_dnn = new yolo_cvdnn_inference::Yolov7();
    yolov7_dnn->Load({1, 3, 640, 640},
                     load_classes(absl::GetFlag(FLAGS_classes)),
                     absl::GetFlag(FLAGS_model));
    yolo = yolov7_dnn;
  } else if (model.find(".rknn") != std::string::npos) {
    spdlog::info("Loading rknn inference unit~");
    yolo = new rknn_yolo_inference::Yolov7(absl::GetFlag(FLAGS_model));

  } else {
    spdlog::error("Not supported model format :<");
    return 0;
  }

  std::string src_path = absl::GetFlag(FLAGS_src);
  if (!src_path.empty()) {
    spdlog::info("Execute image inference.");
    DetectImage(*yolo, src_path);
  } else {
    spdlog::info("Execute video inference.");
    DetectVideo(*yolo);
  }

  return 0;
}
