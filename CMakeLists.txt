cmake_minimum_required(VERSION 3.10.2)

project(rknn-yolo-inference)

add_subdirectory(third_party)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_AUTOMOC ON)

find_package(OpenCV CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Qt6 COMPONENTS Core Widgets SerialPort CONFIG)

set(SRCS src/main.cpp src/yolo11.cpp src/camera.cpp)

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

add_executable(rknn-yolo-inference ${SRCS})

target_link_libraries(
  rknn-yolo-inference
  ${Qt_LIBS}
  # Qt6::Core Qt6::Widgets
  spdlog::spdlog
  absl::flags
  absl::flags_parse
  absl::flags_usage
  ${OpenCV_LIBS}
  yolo11
  )

target_include_directories(
  rknn-yolo-inference PUBLIC 
  include 
  ${OpenCV_INCLUDE_DIRS} 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TREASURE_CHEST_INCLUDE_DIRS} 
  third_party
  third_party/treasure_chest
  third_party/yolo11/cpp
  third_party/rknpu2/include
)


