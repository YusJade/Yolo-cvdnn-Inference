#include "yolov5-lite.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <cassert>
#include <fstream>
#include <ios>
#include <memory>
#include <set>
#include <string>

#include <absl/strings/str_format.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "3rdparty/timer/easy_timer.h"
#include "iyolo.h"
#include "rknn_api.h"
#include "utils/common.h"
#include "utils/image_utils.h"

using namespace rknn_yolo_inference;

// int post_process(rknn_app_context_t *app_ctx, void *outputs,
//                  letterbox_t *letter_box, float conf_threshold,
//                  float nms_threshold, object_detect_result_list *od_results);
// char *coco_cls_to_name(int cls_id);
// int init_post_process();

constexpr int Yolov5Lite::anchor[3][6];

Yolov5Lite::Yolov5Lite(std::string model, std::string label_path) {
  memset(&rknn_app_ctx_, 0, sizeof(rknn_app_context_t));
  InitModel(model);
  InitPostProcess(label_path);
  spdlog::info("successfully init rknn model :>");
}

// 按照 rknn_app_context_t 的定义，使用 rknn_query 装载配置
void Yolov5Lite::InitModel(std::string model_path) {
  // 记录调用 rknn api 的调用结果
  int ret;

  int model_len = 0;
  rknn_context ctx = 0;

  // 加载 rknn 模型文件
  std::ifstream model_file(model_path, std::ios::binary | std::ios::ate);
  if (!model_file.is_open())
    spdlog::critical("failed to load model file!({})", model_path);
  assert(model_file.is_open());

  model_len = static_cast<int>(model_file.tellg());
  auto model_data = std::make_unique<char[]>(model_len);
  model_file.seekg(0, std::ios::beg);
  model_file.read(model_data.get(), model_len);
  spdlog::debug("model_data={}, model_len={}", model_data.get(), model_len);
  ret = rknn_init(&ctx, (void *)model_data.get(), model_len, 0, NULL);
  model_data.release();

  if (ret < 0) spdlog::critical("failed to execute rknn_init!");
  assert(ret >= 0);

  // 获取模型的输入输出层数
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC)
    spdlog::critical(
        "failed to execute rknn_query(cmd: RKNN_QUERY_IN_OUT_NUM, ret: {})",
        ret);
  assert(ret == RKNN_SUCC);
  spdlog::info("rknn model input num: {}, output num: {}", io_num.n_input,
               io_num.n_output);

  // 获取模型输入信息
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
      spdlog::critical(
          "failed to execute rknn_query(cmd: RKNN_QUERY_INPUT_ATTR, ret: {})",
          ret);
    assert(ret == RKNN_SUCC);

    auto &attr = input_attrs[i];
    spdlog::info(
        "input_tensor{} index={}, name={}, n_dims={}, dims=[{}, {}, {}, {}], "
        "n_elems={}, "
        "size={}, fmt={}, type={}, qnt_type={}, "
        "zp={}, scale={}",
        i, attr.index, attr.name, attr.n_dims, attr.dims[3], attr.dims[2],
        attr.dims[1], attr.dims[0], attr.n_elems, attr.size,
        get_format_string(attr.fmt), get_type_string(attr.type),
        get_qnt_type_string(attr.qnt_type), attr.zp, attr.scale);
  }

  // 获取模型输出信息
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                     sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
      spdlog::critical(
          "failed to execute rknn_query(cmd: RKNN_QUERY_OUTPUT_ATTR, ret: {})",
          ret);
    assert(ret == RKNN_SUCC);

    auto &attr = output_attrs[i];
    spdlog::info(
        "output_tensor{} index={}, name={}, n_dims={}, dims=[{}, {}, {}, {}], "
        "n_elems={}, "
        "size={}, fmt={}, type={}, qnt_type={}, "
        "zp={}, scale={}",
        i, attr.index, attr.name, attr.n_dims, attr.dims[3], attr.dims[2],
        attr.dims[1], attr.dims[0], attr.n_elems, attr.size,
        get_format_string(attr.fmt), get_type_string(attr.type),
        get_qnt_type_string(attr.qnt_type), attr.zp, attr.scale);
  }

  // 设置 context
  rknn_app_ctx_.rknn_ctx = ctx;
  // 检查是否为量化模型
  if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
      output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
    rknn_app_ctx_.is_quant = true;
  } else {
    rknn_app_ctx_.is_quant = false;
  }

  rknn_app_ctx_.io_num = io_num;
  rknn_app_ctx_.input_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
  memcpy(rknn_app_ctx_.input_attrs, input_attrs,
         io_num.n_input * sizeof(rknn_tensor_attr));
  rknn_app_ctx_.output_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
  memcpy(rknn_app_ctx_.output_attrs, output_attrs,
         io_num.n_output * sizeof(rknn_tensor_attr));

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    spdlog::info("model is NCHW input fmt\n");
    rknn_app_ctx_.model_channel = input_attrs[0].dims[1];
    rknn_app_ctx_.model_height = input_attrs[0].dims[2];
    rknn_app_ctx_.model_width = input_attrs[0].dims[3];
  } else {
    spdlog::info("model is NHWC input fmt\n");
    rknn_app_ctx_.model_height = input_attrs[0].dims[1];
    rknn_app_ctx_.model_width = input_attrs[0].dims[2];
    rknn_app_ctx_.model_channel = input_attrs[0].dims[3];
  }
  spdlog::info("model input height={}, width={}, channel={}\n",
               rknn_app_ctx_.model_height, rknn_app_ctx_.model_width,
               rknn_app_ctx_.model_channel);
}

// 完成一次推理需要：
// 装载 rknn_input -> rknn_inputs_set -> rknn_outputs_get -> post_process
DetectResult Yolov5Lite::Detect(cv::Mat img0) {
  spdlog::debug("detect...");
  int ret;
  image_buffer_t dst_img;
  letterbox_t letter_box;

  rknn_input inputs[rknn_app_ctx_.io_num.n_input];
  rknn_output outputs[rknn_app_ctx_.io_num.n_output];
  const float nms_threshold = NMS_THRESH;       // 默认的NMS阈值
  const float box_conf_threshold = BOX_THRESH;  // 默认的置信度阈值
  int bg_color = 114;
  TIMER timer;
  timer.indent_set("");

  object_detect_result_list *od_results = new object_detect_result_list;

  image_buffer_t img;
  cv::Mat det_img = img0.clone();
  cv::cvtColor(det_img, det_img, cv::COLOR_BGR2RGB);
  img.format = IMAGE_FORMAT_RGB888;
  img.width = det_img.cols;
  img.height = det_img.rows;
  img.width_stride = det_img.step;
  img.height_stride = det_img.rows;
  img.virt_addr = det_img.data;
  img.size = det_img.total() * det_img.elemSize();
  img.fd = -1;
  // if ((!rknn_app_ctx_) || !(img) || (!od_results)) {
  //   return -1;
  // }
  spdlog::debug("memset...");
  // TODO:  memset(&od_results, 0x00, sizeof(*od_results)); 'memset' will always
  // overflow; destination buffer has size 8, but size argument is 3080
  memset(od_results, 0x00, sizeof(*od_results));
  spdlog::debug("memseted od_results");
  memset(&letter_box, 0, sizeof(letterbox_t));
  spdlog::debug("memseted letter_box");
  memset(&dst_img, 0, sizeof(image_buffer_t));
  spdlog::debug("memseted dst_img");
  spdlog::debug("memseting inputs and outputs");
  memset(inputs, 0, sizeof(inputs));
  memset(outputs, 0, sizeof(outputs));
  spdlog::debug("memseted inputs and outputs");

  spdlog::debug("preprocessing...");
  // Pre Process
  dst_img.width = rknn_app_ctx_.model_width;
  dst_img.height = rknn_app_ctx_.model_height;
  dst_img.format = IMAGE_FORMAT_RGB888;
  dst_img.size = get_image_size(&dst_img);
  dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
  if (dst_img.virt_addr == NULL) {
    printf("malloc buffer size:%d fail!\n", dst_img.size);
    // return -1; TODO
  }
  assert(ret >= 0);
  // letterbox
  timer.tik();
  ret = convert_image_with_letterbox(&img, &dst_img, &letter_box, bg_color);
  if (ret < 0) {
    printf("convert_image_with_letterbox fail! ret=%d\n", ret);

    // TODO:
    // return -1;
  }
  assert(ret >= 0);
  timer.tok();
  timer.print_time("convert_image_with_letterbox");
  spdlog::debug("1");
  // Set Input Data
  inputs[0].index = 0;
  spdlog::debug("2");
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  spdlog::debug("3");
  inputs[0].size = rknn_app_ctx_.model_width * rknn_app_ctx_.model_height *
                   rknn_app_ctx_.model_channel;
  spdlog::debug("4");
  inputs[0].buf = dst_img.virt_addr;
  spdlog::debug("5");
  timer.tik();
  spdlog::debug("setting rknn_inputs_sets");
  ret = rknn_inputs_set(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_input,
                        inputs);
  spdlog::debug("executed rknn_inputs_sets");
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    // return -1; TODO
  }
  assert(ret >= 0);
  timer.tok();
  timer.print_time("rknn_inputs_set");

  // Run
  timer.tik();
  ret = rknn_run(rknn_app_ctx_.rknn_ctx, nullptr);
  if (ret < 0) {
    printf("rknn_run fail! ret=%d\n", ret);
    // return -1; TODO
  }
  assert(ret >= 0);
  timer.tok();
  timer.print_time("rknn_run");

  // Get Output
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < rknn_app_ctx_.io_num.n_output; i++) {
    outputs[i].index = i;
    outputs[i].want_float = (!rknn_app_ctx_.is_quant);
  }

  timer.tik();
  ret = rknn_outputs_get(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_output,
                         outputs, NULL);
  if (ret < 0) {
    printf("rknn_outputs_get fail! ret=%d\n", ret);
    goto out;
  }
  timer.tok();
  timer.print_time("rknn_outputs_get");

  spdlog::debug("postprocessing...");
  // Post Process
  timer.tik();
  PostProcess(&rknn_app_ctx_, outputs, &letter_box, box_conf_threshold,
              nms_threshold, od_results);
  timer.tok();
  timer.print_time("post_process");

  // Remeber to release rknn output
  rknn_outputs_release(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_output,
                       outputs);

out:
  if (dst_img.virt_addr != NULL) {
    free(dst_img.virt_addr);
  }

  DetectResult res;
  for (int idx = 0; idx < od_results->count; idx++) {
    // res.img = img0;
    auto &od = od_results->results[idx];
    cv::Rect2f box(od.box.left, od.box.top, od.box.right - od.box.left,
                   od.box.bottom - od.box.top);
    DetectItem res_item{box, od.cls_id, ClsIdtoName(od.cls_id), od.prop};
    res.items.push_back(res_item);

    cv::rectangle(det_img, cv::Rect(box.x, box.y, box.width, box.height),
                  cv::Scalar(0, 255, 0));
    cv::putText(det_img, ClsIdtoName(od.cls_id), cv::Point(box.x, box.y + 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    spdlog::debug("detected box: {} @ ({} {} {} {}) {}", ClsIdtoName(od.cls_id),
                  od.box.left, od.box.top, od.box.right, od.box.bottom,
                  od.prop);
  }

  cv::Mat frame;
  cv::cvtColor(det_img, frame, cv::COLOR_RGB2BGR);
  Notify(frame);
  res.img = frame;

  return res;
}

int Yolov5Lite::Clamp(float val, int min, int max) {
  return val > min ? (val < max ? val : max) : min;
}

float Yolov5Lite::CalculateOverlap(float xmin0, float ymin0, float xmax0,
                                   float ymax0, float xmin1, float ymin1,
                                   float xmax1, float ymax1) {
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) +
            (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

int Yolov5Lite::NMS(int validCount, std::vector<float> &outputLocations,
                    std::vector<int> classIds, std::vector<int> &order,
                    int filterId, float threshold) {
  for (int i = 0; i < validCount; ++i) {
    int n = order[i];
    if (n == -1 || classIds[n] != filterId) {
      continue;
    }
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[m] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1,
                                   xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

int Yolov5Lite::QuickSortIndiceInverse(std::vector<float> &input, int left,
                                       int right, std::vector<int> &indices) {
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right) {
    key_index = indices[left];
    key = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    QuickSortIndiceInverse(input, left, low - 1, indices);
    QuickSortIndiceInverse(input, low + 1, right, indices);
  }
  return low;
}

int32_t Yolov5Lite::__clip(float val, float min, float max) {
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

int8_t Yolov5Lite::QntFP32toAffine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

uint8_t Yolov5Lite::QntFP32toAffineUInt8(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
  return res;
}

float Yolov5Lite::DeqntAffinetoFP32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}
float Yolov5Lite::DeqntAffineUInt8toFP32(uint8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

int Yolov5Lite::ProcessUInt8(uint8_t *input, int *anchor, int grid_h,
                             int grid_w, int height, int width, int stride,
                             std::vector<float> &boxes,
                             std::vector<float> &objProbs,
                             std::vector<int> &classId, float threshold,
                             int32_t zp, float scale) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  uint8_t thres_u8 = QntFP32toAffineUInt8(threshold, zp, scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        uint8_t box_confidence =
            input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_u8) {
          int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          uint8_t *in_ptr = input + offset;
          float box_x =
              (DeqntAffineUInt8toFP32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float box_y =
              (DeqntAffineUInt8toFP32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float box_w =
              (DeqntAffineUInt8toFP32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float box_h =
              (DeqntAffineUInt8toFP32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          uint8_t maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            uint8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs > thres_u8) {
            objProbs.push_back(
                (DeqntAffineUInt8toFP32(maxClassProbs, zp, scale)) *
                (DeqntAffineUInt8toFP32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

int Yolov5Lite::ProcessInt8(int8_t *input, int *anchor, int grid_h, int grid_w,
                            int height, int width, int stride,
                            std::vector<float> &boxes,
                            std::vector<float> &objProbs,
                            std::vector<int> &classId, float threshold,
                            int32_t zp, float scale) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  int8_t thres_i8 = QntFP32toAffine(threshold, zp, scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        int8_t box_confidence =
            input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_i8) {
          int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          int8_t *in_ptr = input + offset;
          float box_x = (DeqntAffinetoFP32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float box_y =
              (DeqntAffinetoFP32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float box_w =
              (DeqntAffinetoFP32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float box_h =
              (DeqntAffinetoFP32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs > thres_i8) {
            objProbs.push_back((DeqntAffinetoFP32(maxClassProbs, zp, scale)) *
                               (DeqntAffinetoFP32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

int Yolov5Lite::ProcessFP32(float *input, int *anchor, int grid_h, int grid_w,
                            int height, int width, int stride,
                            std::vector<float> &boxes,
                            std::vector<float> &objProbs,
                            std::vector<int> &classId, float threshold) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;

  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        float box_confidence =
            input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= threshold) {
          int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          float *in_ptr = input + offset;
          float box_x = *in_ptr * 2.0 - 0.5;
          float box_y = in_ptr[grid_len] * 2.0 - 0.5;
          float box_w = in_ptr[2 * grid_len] * 2.0;
          float box_h = in_ptr[3 * grid_len] * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          float maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            float prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs > threshold) {
            objProbs.push_back(maxClassProbs * box_confidence);
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

int Yolov5Lite::PostProcess(rknn_app_context_t *app_ctx, void *outputs,
                            letterbox_t *letter_box, float conf_threshold,
                            float nms_threshold,
                            object_detect_result_list *od_results) {
  rknn_output *_outputs = (rknn_output *)outputs;

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;
  int validCount = 0;
  int stride = 0;
  int grid_h = 0;
  int grid_w = 0;
  int model_in_w = app_ctx->model_width;
  int model_in_h = app_ctx->model_height;

  TIMER timer;
  timer.indent_set("");

  memset(od_results, 0, sizeof(object_detect_result_list));

  // spdlog::debug("process");
  for (int i = 0; i < 3; i++) {
    grid_h = app_ctx->output_attrs[i].dims[2];
    grid_w = app_ctx->output_attrs[i].dims[3];
    stride = model_in_h / grid_h;
    timer.tik();
    if (app_ctx->is_quant) {
      validCount += ProcessInt8((int8_t *)_outputs[i].buf, (int *)anchor[i],
                                grid_h, grid_w, model_in_h, model_in_w, stride,
                                filterBoxes, objProbs, classId, conf_threshold,
                                app_ctx->output_attrs[i].zp,
                                app_ctx->output_attrs[i].scale);
    } else {
      validCount += ProcessFP32((float *)_outputs[i].buf, (int *)anchor[i],
                                grid_h, grid_w, model_in_h, model_in_w, stride,
                                filterBoxes, objProbs, classId, conf_threshold);
    }
    timer.tok();
    timer.print_time("process_i8/process_fp32");
  }

  // no object detect
  if (validCount <= 0) {
    return 0;
  }
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }
  timer.tik();
  QuickSortIndiceInverse(objProbs, 0, validCount - 1, indexArray);
  timer.tok();
  timer.print_time("quick sort");

  std::set<int> class_set(std::begin(classId), std::end(classId));

  timer.tik();
  for (auto c : class_set) {
    TIMER nms_timer;
    nms_timer.indent_set("");
    nms_timer.tik();
    NMS(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    nms_timer.tok();
    nms_timer.print_time(absl::StrFormat("nms for %s", ClsIdtoName(c)).c_str());
  }
  timer.tok();
  timer.print_time("nms");

  int last_count = 0;
  od_results->count = 0;

  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
    float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    od_results->results[last_count].box.left =
        (int)(Clamp(x1, 0, model_in_w) / letter_box->scale);
    od_results->results[last_count].box.top =
        (int)(Clamp(y1, 0, model_in_h) / letter_box->scale);
    od_results->results[last_count].box.right =
        (int)(Clamp(x2, 0, model_in_w) / letter_box->scale);
    od_results->results[last_count].box.bottom =
        (int)(Clamp(y2, 0, model_in_h) / letter_box->scale);
    od_results->results[last_count].prop = obj_conf;
    od_results->results[last_count].cls_id = id;
    last_count++;
  }
  od_results->count = last_count;
  return 0;
}

// 加载标签
int Yolov5Lite::InitPostProcess(std::string label_path) {
  std::ifstream file_stream(label_path);
  std::string label_name;
  labels_.clear();
  while (std::getline(file_stream, label_name)) {
    labels_.push_back(label_name);
  }
  return 0;
}

const char *Yolov5Lite::ClsIdtoName(int cls_id) {
  if (cls_id >= OBJ_CLASS_NUM) {
    return "null";
  }

  return labels_[cls_id].c_str();
}

void Yolov5Lite::DeinitPostProcess() {
  // for (int i = 0; i < OBJ_CLASS_NUM; i++) {
  //   if (labels[i] != nullptr) {
  //     free(labels[i]);
  //     labels[i] = nullptr;
  //   }
  // }
}
