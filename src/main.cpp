#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/internal/flag.h>
#include <absl/flags/internal/parse.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/strings/str_format.h>
#include <experimental/filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/fmt/fmt.h>

#include "camera.h"
#include "iyolo.h"
#include "sync_queue/core.h"
// #include "yolo11.h"
#include "timer.h"
#include "yolov5-lite.h"
#include "yolov5.h"
#include "yolov7_dnn.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

ABSL_FLAG(std::string, serial, "ttyS3", "which serial port to open.");
ABSL_FLAG(std::string, label, "classes.txt", "the location of label file.");
ABSL_FLAG(std::string, model, "",
          "the location of model file (.rknn, .onnx supported).");
// ABSL_FLAG(bool, gui, false, "open with gui");
// ABSL_FLAG(int, camera_index, 0, "native camera device index");
// ABSL_FLAG(std::string, camera_url, "http://localhost:4747/video",
//           "network camera url");
ABSL_FLAG(std::string, src, "",
          "the source of input: video, image or camera (native or network "
          "supported).");
ABSL_FLAG(std::string, log_level, "info",
          "decide which level of log will be printed, set to \"debug\" for "
          "detail of inference.");

using arm_face_id::Camera;
using namespace rknn_yolo_inference;
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

std::ofstream CreateRunsFile(std::string model_path = "model") {
  absl::Time current_time = absl::Now();
  std::string date_time_str =
      absl::FormatTime("%Y-%m-%d_%H-%M-%S", current_time, absl::UTCTimeZone());
  std::string dir_name = "runs/" + date_time_str;
  std::experimental::filesystem::path mpath(model_path);
  if (std::experimental::filesystem::create_directory(dir_name)) {
    std::cout << "directory created: " << dir_name << std::endl;
  } else {
    std::cout << "failed to create directory or directory already exists: "
              << dir_name << std::endl;
  }
  std::string fps_file_name =
      fmt::format("{}/{}.fps", dir_name, mpath.filename().string());
  std::ofstream file(fps_file_name, std::ios::out);
  if (!file.is_open()) {
    spdlog::error("fail create fps file:{}", fps_file_name);
  }
  return file;
}

void DetectVideo(rknn_yolo_inference::IYolo& yolo, std::ofstream& fps_file) {
  // 考虑到开发板内存有限, 同步队列的容量不宜过大.
  SyncQueue<cv::Mat> detect_task_queue(200);
  std::string src = absl::GetFlag(FLAGS_src);
  int camera_index = 0;
  std::string camera_url = "http://127.0.0.1/";
  if (src.find("://")) {
    camera_url = src;
  } else {
    try {
      camera_index = std::stoi(src);
    } catch (std::exception e) {
      spdlog::error("invalid native camera {}, fallback to 0, {}", src,
                    e.what());
    }
  }
  Camera::Settings camera_settings{camera_url, camera_index};
  std::shared_ptr<Camera> cam =
      std::make_shared<Camera>(camera_settings, detect_task_queue);

  // 由 yolo 绘制结果帧并通知 cam 保存为视频.
  yolo.AddObserver<cv::Mat>(cam);
  cam->SetReadInterval(0);
  cam->Open();

  Timer timer;
  std::thread cam_thread([&] { cam->Start(); });
  std::thread detect_thread([&] {
    // 等待队列任务抵达, 避免提前结束.
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    SPDLOG_INFO("start inference");
    int frame_count = 0;
    while (detect_task_queue.TaskQuantity() != 0 || cam->IsRunning()) {
      SPDLOG_INFO("cam_is_running:{}, task_quantity:{}", cam->IsRunning(),
                  detect_task_queue.TaskQuantity());
      cv::Mat mat = detect_task_queue.Dequeue();

      frame_count++;
      if (frame_count < 5) continue;
      frame_count = 0;

      timer.Tik();
      DetectResult res = yolo.Detect(mat);
      timer.Tok();
      SPDLOG_INFO("detected one frame, fps: {}, target: {}", timer.FPS(),
                  res.items.size());
      fps_file << timer.FPS() << std::endl;
    }
    SPDLOG_INFO("finished inference.");
    // cam->Close();
  });

  cam_thread.join();
  detect_thread.join();
  SPDLOG_INFO("saving result...");
}

void DetectImage(rknn_yolo_inference::IYolo& yolo, std::string img_path,
                 std::ofstream& fps_file) {
  cv::Mat img = cv::imread(img_path);
  auto result = yolo.Detect(img);
  cv::imwrite("result.jpg", result.img);
  spdlog::info("Result of image inference saved as result.jpg");
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
  std::string log_level = absl::GetFlag(FLAGS_log_level);
  if (log_level == "info")
    spdlog::set_level(spdlog::level::info);
  else if (log_level == "debug")
    spdlog::set_level(spdlog::level::debug);

  spdlog::info("initialized logger, log level is: {}", log_level);
  spdlog::info("this is a script to inference with yolo.");

  std::string model = absl::GetFlag(FLAGS_model);
  std::string label_path = absl::GetFlag(FLAGS_label);
  rknn_yolo_inference::IYolo* yolo = nullptr;
  // 如果是 onnx 模型, 则加载 dnn 推理.
  if (model.find(".onnx") != std::string::npos) {
    spdlog::info("loading onnx inference");
    auto yolov7_dnn = new yolo_cvdnn_inference::Yolov7();
    yolov7_dnn->Load({1, 3, 640, 640}, label_path, absl::GetFlag(FLAGS_model));
    yolo = yolov7_dnn;
  } else if (model.find(".rknn") != std::string::npos) {
    spdlog::info("loading rknn inference");
    // // yolo = new rknn_yolo_inference::Yolov7(absl::GetFlag(FLAGS_model));
    // yolo = new rknn_yolo_inference::Yolov5(model, label_path);
    yolo = new rknn_yolo_inference::Yolov5Lite(model, label_path);
    // return 0;
  } else {
    spdlog::error("not supported model format :<");
    return 0;
  }

  std::string src_path = absl::GetFlag(FLAGS_src);
  std::ofstream fps_file;
  if (src_path.find(".mp4") == std::string::npos &&
      src_path.find(".avi") == std::string::npos) {
    spdlog::info("run image inference.");
    fps_file = CreateRunsFile(model);
    DetectImage(*yolo, src_path, fps_file);
    fps_file.close();
  } else {
    spdlog::info("run video inference.");
    fps_file = CreateRunsFile(model);

    DetectVideo(*yolo, fps_file);
    fps_file.close();
  }

  return 0;
}
