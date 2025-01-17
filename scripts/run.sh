#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./
./rknn_yolo_inference \
  --model=./yolov5.rknn \
  --class=./ --src=