cmake_minimum_required(VERSION 3.10.2)

project(Yolo-cvdnn-Inference)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_AUTOMOC ON)

find_package(OpenCV CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Qt6 COMPONENTS Core Widgets SerialPort CONFIG)

set(SRCS src/main.cpp src/yolo.cpp src/yolov7.cpp src/camera.cpp)

if(${Qt6_FOUND})
  set(Qt_LIBS Qt6::Core Qt6::Widgets Qt6::SerialPort)
  qt_wrap_cpp(SRCS)
else()
  find_package(
    Qt5
    COMPONENTS Core Widgets SerialPort
    REQUIRED CONFIG)
  set(Qt_LIBS Qt5::Core Qt5::Widgets Qt5::SerialPort)
  qt5_wrap_cpp(SRCS)
endif()

add_executable(Yolo-cvdnn-Inference ${SRCS})

target_link_libraries(
  Yolo-cvdnn-Inference
  ${Qt_LIBS}
  # Qt6::Core Qt6::Widgets
  spdlog::spdlog
  absl::flags
  absl::flags_parse
  absl::flags_usage
  ${OpenCV_LIBS})

target_include_directories(
  Yolo-cvdnn-Inference
  PUBLIC include ${OpenCV_INCLUDE_DIRS} third_party
         ${TREASURE_CHEST_INCLUDE_DIRS} third_party/treasure_chest)

add_subdirectory(third_party)
