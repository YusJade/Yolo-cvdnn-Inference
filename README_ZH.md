# RKNN YOLO ZOO

`RKNN YOLO ZOO` 基于 RKNPU SDK 工具链开发的，提供了 YOLO 模型的部署示例。包括使用 CAPI 推理 YOLO 的 `.RKNN` 模型的过程。

- ⚠️ 仅支持 `rk3566`。

## 环境配置

对于 Python 环境的配置（使用 Python API 将模型转换为 RKNN 格式），请参考官方仓库🐈‍⬛。

<a href="https://github.com/airockchip/rknn_model_zoo/?tab=readme-ov-file">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=airockchip&repo=rknn_model_zoo&theme=default" alt="airockchip/rknn_model_zoo" />
</a>

C++ 开发环境使用了 `abseil` 和 `spdlog`。

## 模型支持计划

 - [x]  YOLOv11n
 - [x]  YOLOv11s
 - [x]  YOLOv7n
 - [x]  YOLOv5n

## 仓库中的一些工具

 - `scripts/build_rk3566.sh`：使用 `CMake` 构建本项目。
 - `scripts/optimize_graph.sh`：将某些特定算子的输出转换为模型的输出，并舍弃其后的所有算子。
 - `scripts/run.sh`：运行推理。