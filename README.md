# 从 rknn_model_zone 移植构建流程

直接使用 CMake 构建例程是无法直接在 rk3566 上运行的，rknn_model_zone 提供了顶层的构建脚本 build-linux.sh ，下面通过注释分析一下脚本的功能：

> 阅读 .sh 前的 Tips:
 - `if [[ -z ${xxx} ]]`: 检查该变量是否为空

```sh
#!/bin/bash

# 为 rk3566 构建 yolo11 例程，可以这样使用脚本：
# ./build-linux.sh -t rk356x -a aarch64 -d yolo11

set -e

# 配置交叉编译工具
GCC_COMPILER=/opt/gcc-aarch64-linux-gnu/bin/aarch64-linux-gnu

# 打印出当前脚本的文件名以及传递给脚本的所有参数
echo "$0 $@"
# 处理传入的选项并获取选项后面跟着的参数值
while getopts ":t:a:d:b:m:r" opt; do
  case $opt in
    t)
      TARGET_SOC=$OPTARG
      ;;
    a)
      TARGET_ARCH=$OPTARG
      ;;
    b)
      BUILD_TYPE=$OPTARG
      ;;
    m)
      ENABLE_ASAN=ON
      export ENABLE_ASAN=TRUE
      ;;
    d)
      BUILD_DEMO_NAME=$OPTARG
      ;;
    r)
      DISABLE_RGA=ON
      ;;
    :) # 缺失某个选项
      echo "Option -$OPTARG requires an argument." 
      exit 1
      ;;
    ?) # 有未知选项
      echo "Invalid option: -$OPTARG index:$OPTIND"
      ;;
  esac
done

if [ -z ${TARGET_SOC} ] || [ -z ${BUILD_DEMO_NAME} ]; then
  echo "$0 -t <target> -a <arch> -d <build_demo_name> [-b <build_type>] [-m]"
  echo ""
  echo "    -t : target (rk356x/rk3588/rk3576/rv1106/rk1808/rv1126)"
  echo "    -a : arch (aarch64/armhf)"
  echo "    -d : demo name"
  echo "    -b : build_type(Debug/Release)"
  echo "    -m : enable address sanitizer, build_type need set to Debug"
  echo "    -r : disable rga, use cpu resize image"
  echo "such as: $0 -t rk3588 -a aarch64 -d mobilenet"
  echo "Note: 'rk356x' represents rk3562/rk3566/rk3568, 'rv1106' represents rv1103/rv1106, 'rv1126' represents rv1109/rv1126"
  echo "Note: 'disable rga option is invalid for rv1103/rv1103b/rv1106"
  echo ""
  exit -1
fi

if [[ -z ${GCC_COMPILER} ]];then
    if [[ ${TARGET_SOC} = "rv1106"  || ${TARGET_SOC} = "rv1103" ]];then
        echo "Please set GCC_COMPILER for $TARGET_SOC"
        echo "such as export GCC_COMPILER=~/opt/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf"
        exit
    elif [[ ${TARGET_SOC} = "rv1109" || ${TARGET_SOC} = "rv1126" ]];then
        GCC_COMPILER=arm-linux-gnueabihf
    else
        GCC_COMPILER=aarch64-linux-gnu
    fi
fi
echo "$GCC_COMPILER"
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

if command -v ${CC} >/dev/null 2>&1; then
    :
else
    echo "${CC} is not available"
    echo "Please set GCC_COMPILER for $TARGET_SOC"
    echo "such as export GCC_COMPILER=~/opt/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf"
    exit
fi

# Debug / Release
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

# Build with Address Sanitizer for memory check, BUILD_TYPE need set to Debug
if [[ -z ${ENABLE_ASAN} ]];then
    ENABLE_ASAN=OFF
fi

if [[ -z ${DISABLE_RGA} ]];then
    DISABLE_RGA=OFF
fi

# 找到并获取目标例程的 cpp 目录路径
for demo_path in `find examples -name ${BUILD_DEMO_NAME}`
do
    if [ -d "$demo_path/cpp" ]
    then
        BUILD_DEMO_PATH="$demo_path/cpp"
        break;
    fi
done

if [[ -z "${BUILD_DEMO_PATH}" ]]
then
    echo "Cannot find demo: ${BUILD_DEMO_NAME}, only support:"

    for demo_path in `find examples -name cpp`
    do
        if [ -d "$demo_path" ]
        then
            dname=`dirname "$demo_path"`
            name=`basename $dname`
            echo "$name"
        fi
    done
    echo "rv1106_rv1103 only support: mobilenet and yolov5/6/7/8/x"
    exit
fi

case ${TARGET_SOC} in
    rk356x)
        ;;
    rk3588)
        ;;
    rv1106)
        ;;
    rv1103)
        TARGET_SOC="rv1106"
        ;;
    rk3566)
        TARGET_SOC="rk356x"
        ;;
    rk3568)
        TARGET_SOC="rk356x"
        ;;
    rk3562)
        TARGET_SOC="rk356x"
        ;;
    rk3576)
        TARGET_SOC="rk3576"
        ;;
    rk1808):
        TARGET_SOC="rk1808"
        ;;
    rv1109)
        ;;
    rv1126)
        TARGET_SOC="rv1126"
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3562,rk3566,rk3568,rk3588,rk3576,rv1106,rv1103,rk1808,rv1109,rv1126"
        exit -1
        ;;
esac

TARGET_SDK="rknn_${BUILD_DEMO_NAME}_demo"

TARGET_PLATFORM=${TARGET_SOC}_linux
if [[ -n ${TARGET_ARCH} ]];then
TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
fi
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
INSTALL_DIR=${ROOT_PWD}/install/${TARGET_PLATFORM}/${TARGET_SDK}
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_SDK}_${TARGET_PLATFORM}_${BUILD_TYPE}

echo "==================================="
echo "BUILD_DEMO_NAME=${BUILD_DEMO_NAME}"
echo "BUILD_DEMO_PATH=${BUILD_DEMO_PATH}"
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "ENABLE_ASAN=${ENABLE_ASAN}"
echo "DISABLE_RGA=${DISABLE_RGA}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}
cmake ../../${BUILD_DEMO_PATH} \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_ASAN=${ENABLE_ASAN} \
    -DDISABLE_RGA=${DISABLE_RGA} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j4
make install

# Check if there is a rknn model in the install directory
suffix=".rknn"
shopt -s nullglob
if [ -d "$INSTALL_DIR" ]; then
    files=("$INSTALL_DIR/model/"/*"$suffix")
    shopt -u nullglob

    if [ ${#files[@]} -le 0 ]; then
        echo -e "\e[91mThe RKNN model can not be found in \"$INSTALL_DIR/model\", please check!\e[0m"
    fi
else
    echo -e "\e[91mInstall directory \"$INSTALL_DIR\" does not exist, please check!\e[0m"
fi

```

## 分析 CMake 构建时必要的参数

```sh
cd ${BUILD_DIR}
cmake ../../${BUILD_DEMO_PATH} \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_ASAN=${ENABLE_ASAN} \
    -DDISABLE_RGA=${DISABLE_RGA} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
make -j4
make install
```

- `-DTARGET_SOC=${TARGET_SOC}: rk356x`
- `-DCMAKE_SYSTEM_NAME=Linux`
- `-DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH}: arrch64`
- `-DENABLE_ASAN=${ENABLE_ASAN}: off`
- `-DDISABLE_RGA=${DISABLE_RGA}: off`

`CMake_xxx`的变量为 CMake 提供的内置变量，这里暂不关注，下面根据变量值，取出 `CMakeLists.txt` 中条件成立的部分：

**${TARGET_SOC}: rk356x**:

这部分代码反应了例程所依赖的库和目标，在移植时要注意将依赖的目标一同移植过来。

```cmake
# Currently zero copy only supports rknpu2, v1103/rv1103b/rv1106 supports zero copy by default
if (NOT (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103" OR TARGET_SOC STREQUAL "rk1808" 
    OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126" OR TARGET_SOC STREQUAL "rv1103b"))
    add_executable(${PROJECT_NAME}_zero_copy
        main.cc
        postprocess.cc
        rknpu2/yolo11_zero_copy.cc
    )

    target_compile_definitions(${PROJECT_NAME}_zero_copy PRIVATE ZERO_COPY)

    target_link_libraries(${PROJECT_NAME}_zero_copy
        imageutils
        fileutils
        imagedrawing    
        ${LIBRKNNRT}
        dl
    )

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        target_link_libraries(${PROJECT_NAME}_zero_copy Threads::Threads)
    endif()

    target_include_directories(${PROJECT_NAME}_zero_copy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBRKNNRT_INCLUDES}
        ${LIBTIMER_INCLUDES}
    )
    install(TARGETS ${PROJECT_NAME}_zero_copy DESTINATION .)
endif()
```

**${ENABLE_ASAN}: off**

```cmake
if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()
```

**${DISABLE_RGA}: off**

```cmake
# 无相关代码
```

## 移植 Tips

**构建指令:**

```sh
cmake ../../${BUILD_DEMO_PATH} \
    -DTARGET_SOC=rk356x \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=arrch64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_ASAN=off \
    -DDISABLE_RGA=off \
    -DCMAKE_INSTALL_PREFIX=./rknn_yolo11_infer_patch
make -j4
make install
```

**依赖的 `CMake` 目标:**
- `imageutils`
- `fileutils`
- `imagedrawing` 

**依赖的库:**
- `Threads::Threads`
