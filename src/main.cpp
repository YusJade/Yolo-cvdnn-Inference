#include <iostream>

#include <absl/flags/flag.h>
#include <absl/flags/internal/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <opencv2/core.hpp>

#include "yolov7.h"

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

ABSL_FLAG(std::string, serial_port, "ttyS3", "use which serial port");
ABSL_FLAG(std::string, classes, "classes.txt", "the classes.txt file");
ABSL_FLAG(std::string, model, "yolov5.onnx", "onnx model path");
ABSL_FLAG(bool, gui, false, "open with gui");
ABSL_FLAG(int, camera_index, 0, "native camera device index");
ABSL_FLAG(std::string, camera_url, "http://localhost:4747/video",
          "network camera url");

using yolo_cvdnn_inference::Yolov7;

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] [%s:%#] %v");
  spdlog::set_level(spdlog::level::debug);
  SPDLOG_DEBUG("initialized logger");
  SPDLOG_DEBUG("this is a script to inference with yolo.");

  Yolov7 yolo;
  yolo.Load({1, 3, 640, 640}, {}, absl::GetFlag(FLAGS_model));

  return 0;
}