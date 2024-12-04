#! /bin/bash

cmake -S . -B rk3566-build \
    -DTARGET_SOC=rk356x \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_ASAN=off \
    -DDISABLE_RGA=off \
    -DCMAKE_INSTALL_PREFIX=./rknn_yolo11_infer_patch \
    -DCMAKE_TOOLCHAIN_FILE=/home/yu/aarch64.cmake 

cmake --build ./rk3566-build --target rknn-yolo-inference