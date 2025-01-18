#include "yolov5.h"

#include <cstdint>
#include <fstream>

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "common.h"
#include "image_utils.h"
#include "iyolo.h"
#include "quantization.h"

using namespace rknn_yolo_inference;

Yolov5::Yolov5(std::string model, std::string label_path) {
  memset(&rknn_app_ctx_, 0, sizeof(rknn_app_context_t));
  InitModel(model);
  InitPostProcess(label_path);
  spdlog::info("successfully init rknn model :>");
}

// 按照 rknn_app_context_t 的定义，使用 rknn_query 装载配置
void Yolov5::InitModel(std::string model_path) {
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
DetectResult Yolov5::Detect(cv::Mat img0) {
  spdlog::debug("detect...");
  int ret;

  rknn_input inputs[rknn_app_ctx_.io_num.n_input];
  rknn_output outputs[rknn_app_ctx_.io_num.n_output];
  memset(inputs, 0, sizeof(inputs));
  memset(outputs, 0, sizeof(outputs));
  // const float nms_threshold = NMS_THRESH;       // 默认的NMS阈值
  // const float box_conf_threshold = BOX_THRESH;  // 默认的置信度阈值

  cv::Mat rgb_img = img0.clone();
  cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);
  image_buffer_t src_img_buf;
  src_img_buf.format = IMAGE_FORMAT_RGB888;
  src_img_buf.width = rgb_img.cols;
  src_img_buf.height = rgb_img.rows;
  src_img_buf.width_stride = rgb_img.step;
  src_img_buf.height_stride = rgb_img.rows;
  src_img_buf.virt_addr = rgb_img.data;
  src_img_buf.size = rgb_img.total() * rgb_img.elemSize();
  src_img_buf.fd = -1;

  image_buffer_t dst_img_buf;
  memset(&dst_img_buf, 0, sizeof(image_buffer_t));

  dst_img_buf.width = rknn_app_ctx_.model_width;
  dst_img_buf.height = rknn_app_ctx_.model_height;
  dst_img_buf.format = IMAGE_FORMAT_RGB888;
  dst_img_buf.size = get_image_size(&dst_img_buf);
  dst_img_buf.virt_addr = (unsigned char *)malloc(dst_img_buf.size);
  if (dst_img_buf.virt_addr == NULL)
    spdlog::critical("malloc buffer size:%d fail!\n", dst_img_buf.size);
  assert(dst_img_buf.virt_addr != NULL);

  letterbox_t letterbox;
  memset(&letterbox, 0, sizeof(letterbox_t));
  int bg_color = 114;
  ret = convert_image_with_letterbox(&src_img_buf, &dst_img_buf, &letterbox,
                                     bg_color);
  if (ret < 0)
    spdlog::critical("convert_image_with_letterbox fail! ret=%d\n", ret);
  assert(ret >= 0);

  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].size = rknn_app_ctx_.model_width * rknn_app_ctx_.model_height *
                   rknn_app_ctx_.model_channel;
  inputs[0].buf = dst_img_buf.virt_addr;
  spdlog::info("inputs inited");

  ret = rknn_inputs_set(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_input,
                        inputs);
  spdlog::info("rknn_inputs_set");
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    // return -1; TODO
  }
  assert(ret >= 0);

  ret = rknn_run(rknn_app_ctx_.rknn_ctx, nullptr);
  spdlog::info("rknn_run");
  if (ret < 0) {
    printf("rknn_run fail! ret=%d\n", ret);
    // return -1; TODO
  }
  assert(ret >= 0);

  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < rknn_app_ctx_.io_num.n_output; i++) {
    outputs[i].index = i;
    outputs[i].want_float = (!rknn_app_ctx_.is_quant);
  }

  ret = rknn_outputs_get(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_output,
                         outputs, NULL);
  spdlog::info("rknn_outputs_get");
  if (ret < 0) {
    printf("rknn_outputs_get fail! ret=%d\n", ret);
  }
  assert(ret >= 0);

  DetectResult result;
  // TODO
  int OBJ_CLASS_NUM = 80;
  int PROP_BOX_SIZE = OBJ_CLASS_NUM + 5;
  int8_t *output_buf = (int8_t *)outputs[0].buf;
  // TODO: 置信度阈值需要量化为 int8_t
  float thres_i8 = QntFP32toAffine(0.2, rknn_app_ctx_.output_attrs->zp,
                                   rknn_app_ctx_.output_attrs->scale);
  spdlog::info("prob_box_count={}, thres_i8={}",
               rknn_app_ctx_.output_attrs->dims[1], thres_i8);
  for (int idx = 0; idx < rknn_app_ctx_.output_attrs->dims[1]; idx++) {
    int8_t box_confidence = output_buf[idx * PROP_BOX_SIZE + OBJ_CLASS_NUM + 4];
    spdlog::debug(
        "prob_box # {} {} {} {} {}",
        DeqntAffineUInt8toFP32(
            output_buf[idx * PROP_BOX_SIZE + OBJ_CLASS_NUM + 0],
            rknn_app_ctx_.output_attrs->zp, rknn_app_ctx_.output_attrs->scale),
        DeqntAffineUInt8toFP32(
            output_buf[idx * PROP_BOX_SIZE + OBJ_CLASS_NUM + 1],
            rknn_app_ctx_.output_attrs->zp, rknn_app_ctx_.output_attrs->scale),
        DeqntAffineUInt8toFP32(
            output_buf[idx * PROP_BOX_SIZE + OBJ_CLASS_NUM + 2],
            rknn_app_ctx_.output_attrs->zp, rknn_app_ctx_.output_attrs->scale),
        DeqntAffineUInt8toFP32(
            output_buf[idx * PROP_BOX_SIZE + OBJ_CLASS_NUM + 3],
            rknn_app_ctx_.output_attrs->zp, rknn_app_ctx_.output_attrs->scale),
        DeqntAffineUInt8toFP32(
            output_buf[idx * PROP_BOX_SIZE + OBJ_CLASS_NUM + 4],
            rknn_app_ctx_.output_attrs->zp, rknn_app_ctx_.output_attrs->scale)

    );
    // float box_conf_fp32 =
    //     DeqntAffinetoFP32(box_confidence, rknn_app_ctx_.output_attrs->zp,
    //                       rknn_app_ctx_.output_attrs->scale);

    if (box_confidence < thres_i8) continue;
    spdlog::debug("box_confidence={} (thres={})", box_confidence, thres_i8);
    int max_class_id = 0;
    float max_class_prob = 0.0;
    for (int cls_id = 0; cls_id < OBJ_CLASS_NUM; cls_id++) {
      spdlog::debug("cls{} prob={}", cls_id,
                    output_buf[cls_id + idx * PROP_BOX_SIZE]);
      if (output_buf[cls_id + idx * PROP_BOX_SIZE] > max_class_prob) {
        max_class_prob = output_buf[cls_id + idx * PROP_BOX_SIZE];
        max_class_id = cls_id;
      }
    }
    int box_x = output_buf[0 + OBJ_CLASS_NUM + idx * PROP_BOX_SIZE];
    int box_y = output_buf[1 + OBJ_CLASS_NUM + idx * PROP_BOX_SIZE];
    int box_width = output_buf[2 + OBJ_CLASS_NUM + idx * PROP_BOX_SIZE];
    int box_height = output_buf[3 + OBJ_CLASS_NUM + idx * PROP_BOX_SIZE];

    cv::Rect2f box(box_x, box_y, box_width, box_height);
    DetectItem item{box, max_class_id, labels_[max_class_id], max_class_prob};
    result.items.push_back(item);

    cv::rectangle(rgb_img, cv::Rect(box.x, box.y, box.width, box.height),
                  cv::Scalar(0, 255, 0));
    cv::putText(rgb_img, item.class_label, cv::Point(box.x, box.y + 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    spdlog::debug("detected box: {} @ ({} {} {} {}) {}", item.class_label,
                  box.x, box.y, box.width, box.height, item.class_index);
  }

  cv::Mat frame;
  cv::cvtColor(rgb_img, frame, cv::COLOR_RGB2BGR);
  Notify(frame);
  result.img = frame;

  rknn_outputs_release(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_output,
                       outputs);
  if (dst_img_buf.virt_addr != NULL) {
    free(dst_img_buf.virt_addr);
  }

  return result;
}

// 加载标签
int Yolov5::InitPostProcess(std::string label_path) {
  std::ifstream file_stream(label_path);
  std::string label_name;
  labels_.clear();
  while (std::getline(file_stream, label_name)) {
    labels_.push_back(label_name);
  }
  // OBJ_CLASS_NUM = label_name.size();
  return 0;
}

cv::Mat Yolov5::Letterbox(const cv::Mat &src, const cv::Size &target_size) {
  // 获取原始图像的宽高
  int src_width = src.cols;
  int src_height = src.rows;

  // 计算缩放比例
  float scale = std::min(static_cast<float>(target_size.width) / src_width,
                         static_cast<float>(target_size.height) / src_height);

  // 计算缩放后的图像尺寸
  int new_width = static_cast<int>(src_width * scale);
  int new_height = static_cast<int>(src_height * scale);

  // 缩放图像
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_width, new_height));

  // 创建目标图像，初始化为黑色
  cv::Mat dst =
      cv::Mat::zeros(target_size.height, target_size.width, src.type());

  // 计算图像在目标图像中的位置
  int dx = (target_size.width - new_width) / 2;
  int dy = (target_size.height - new_height) / 2;

  // 将缩放后的图像复制到目标图像的中心
  resized.copyTo(dst(cv::Rect(dx, dy, new_width, new_height)));

  return dst;
}