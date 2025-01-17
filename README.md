# RKNN YOLO ZOO

`RKNN YOLO ZOO` is developed based on the RKNPU SDK toolchain and provides deployment examples for yolo. Include the process of using CAPI to infer the yolo\`s `.RKNN` model.

- ⚠️ support `rk3566` only.

## Enviroment

For python environment configuration (using Python API convert your model to rknn format), plz refer to the official repository🐈‍⬛. 

 <a href="https://github.com/airockchip/rknn_model_zoo/?tab=readme-ov-file">
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=airockchip&repo=rknn_model_zoo&theme=default" alt="airockchip/rknn_model_zoo" />
</a>


## Model support plan

 - [x]  yolo11
 - [x]  yolov7
 - [x]  yolov5


## Some utils in repo

 - `scripts/build_rk3566.sh`: use `CMake` to build this project.
 - `scripts/optimize_graph.sh`: convert the output of some specific operator into model`s output, and discard all operators that follow it.
 - `scripts/run.sh`: run inference. 